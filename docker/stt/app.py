"""ScrollKeeper speech-to-text: Parakeet-TDT 0.6B (sherpa-onnx, CPU) behind an OpenAI-style API.

POST /v1/audio/transcriptions accepts anything ffmpeg can decode, including whole per-speaker
tracks of several hours. Silero VAD cuts the track into speech regions, each region is decoded
separately, and word timestamps are offset back to track time.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import sherpa_onnx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from pipeline import SAMPLE_RATE, Progress, RegionCutter, build_response, transcribe_track

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("scrollkeeper.stt")

MODEL_DIR = Path(os.getenv("STT_MODEL_DIR", "/models/parakeet-tdt-0.6b-v2-int8"))
VAD_MODEL = Path(os.getenv("STT_VAD_MODEL", "/models/silero_vad.onnx"))
THREADS = int(os.getenv("STT_THREADS", "3"))
MAX_SPEECH_SECONDS = float(os.getenv("STT_MAX_SPEECH_SECONDS", "20"))
READ_BYTES = SAMPLE_RATE * 2 * 10  # ten seconds of s16le mono per read

_recognizer: sherpa_onnx.OfflineRecognizer | None = None
# One transcription at a time: a second concurrent decode would only compete for the same cores.
_busy = threading.Lock()


def load_recognizer() -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(MODEL_DIR / "encoder.int8.onnx"),
        decoder=str(MODEL_DIR / "decoder.int8.onnx"),
        joiner=str(MODEL_DIR / "joiner.int8.onnx"),
        tokens=str(MODEL_DIR / "tokens.txt"),
        num_threads=THREADS,
        model_type="nemo_transducer",
        decoding_method="greedy_search",
    )


def new_cutter() -> RegionCutter:
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(VAD_MODEL)
    config.silero_vad.max_speech_duration = MAX_SPEECH_SECONDS
    config.sample_rate = SAMPLE_RATE
    config.num_threads = 1
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=MAX_SPEECH_SECONDS + 30)
    return RegionCutter(
        vad,
        concat=np.concatenate,
        window_size=config.silero_vad.window_size,
        keep_samples=int(SAMPLE_RATE * (MAX_SPEECH_SECONDS + 30)),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _recognizer
    started = time.monotonic()
    _recognizer = load_recognizer()
    new_cutter()  # fail at startup, not on the first request, if the VAD model is missing
    log.info("Loaded Parakeet from %s with %s threads in %.1fs", MODEL_DIR, THREADS, time.monotonic() - started)
    yield


app = FastAPI(title="ScrollKeeper STT", lifespan=lifespan)


def decode_pcm(path: Path) -> Iterator[np.ndarray]:
    """Stream the file as 16 kHz mono float32 chunks via ffmpeg, without holding it all in memory."""
    process = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None and process.stderr is not None
    try:
        leftover = b""
        while block := process.stdout.read(READ_BYTES):
            block = leftover + block
            usable = len(block) - len(block) % 2
            leftover = block[usable:]
            yield np.frombuffer(block[:usable], dtype=np.int16).astype(np.float32) / 32768.0
        stderr = process.stderr.read().decode(errors="replace").strip()
        if process.wait() != 0:
            raise HTTPException(status_code=400, detail=f"Could not decode audio: {stderr[-500:]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def decode_region(samples: np.ndarray):
    assert _recognizer is not None
    stream = _recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    _recognizer.decode_stream(stream)
    return stream.result


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if _recognizer is not None else "loading",
        "model": MODEL_DIR.name,
        "threads": THREADS,
        "busy": _busy.locked(),
    }


@app.post("/v1/audio/transcriptions", response_model=None)
def transcribe(
    file: UploadFile = File(...),
    model: str = Form(""),
    response_format: str = Form("json"),
    timestamp_granularities: list[str] = Form(default=[], alias="timestamp_granularities[]"),
    prompt: str = Form(""),
) -> dict[str, object] | PlainTextResponse:
    """OpenAI-compatible transcription. `model` and `prompt` are accepted and ignored: the
    benchmark (#3) found Parakeet hotwords unusable, so names are fixed downstream."""
    if response_format not in {"json", "text", "verbose_json"}:
        raise HTTPException(status_code=400, detail=f"Unsupported response_format {response_format!r}")
    name = file.filename or "audio"
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix or ".audio") as handle:
        shutil.copyfileobj(file.file, handle)
        handle.flush()
        if not _busy.acquire(blocking=False):
            log.info("Waiting for the previous transcription to finish before %s", name)
            _busy.acquire()
        try:
            started = time.monotonic()

            def report(progress: Progress) -> None:
                log.info(
                    "%s: %.1f min of audio, %s segments, %.0fs elapsed (%.1fx realtime)",
                    name,
                    progress.audio_seconds / 60,
                    progress.regions,
                    progress.elapsed_seconds,
                    progress.audio_seconds / max(progress.elapsed_seconds, 1e-6),
                )

            segments, duration = transcribe_track(
                decode_pcm(Path(handle.name)), new_cutter(), decode_region, on_progress=report
            )
        finally:
            _busy.release()
    elapsed = time.monotonic() - started
    log.info(
        "%s: transcribed %.1f min in %.1fs (%.1fx realtime), %s segments",
        name,
        duration / 60,
        elapsed,
        duration / max(elapsed, 1e-6),
        len(segments),
    )
    result = build_response(segments, duration, response_format, timestamp_granularities)
    if isinstance(result, str):
        return PlainTextResponse(result)
    return result
