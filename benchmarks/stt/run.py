"""Benchmark one STT engine at one thread count on a 16 kHz mono WAV file.

Run each engine/thread combination as its own process so peak RSS is clean:

    python benchmarks/stt/run.py --engine faster-whisper --model small.en \
        --threads 2 --audio slice.wav --reference slice.txt --glossary names.txt

Prepare the audio with:  ffmpeg -i in.ogg -ac 1 -ar 16000 slice.wav
Prints one JSON object with realtime factor, peak RSS, WER and glossary recall,
and writes the engine's words/timestamps (when provided) to --dump for manual
inspection. Engines are imported lazily; install only the one you are testing.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metrics import glossary_recall, realtime_factor, word_error_rate  # noqa: E402

CHUNK_SECONDS = 30

# Word = (text, start_seconds, end_seconds)
Word = tuple[str, float, float]


def audio_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1:
            raise SystemExit("audio must be 16 kHz mono WAV (see module docstring)")
        return wav.getnframes() / wav.getframerate()


def read_samples(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def load_faster_whisper(model: str, threads: int):
    from faster_whisper import WhisperModel

    engine = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)

    def transcribe(path: Path) -> tuple[str, list[Word]]:
        segments, _ = engine.transcribe(read_samples(path), word_timestamps=True, vad_filter=True)
        words: list[Word] = []
        for segment in segments:  # lazy generator: consuming it is the decode
            words.extend((w.word.strip(), w.start, w.end) for w in segment.words or [])
        return " ".join(w[0] for w in words), words

    return transcribe


def load_onnx_asr(model: str, threads: int):
    import onnx_asr
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    engine = onnx_asr.load_model(model, quantization="int8", sess_options=options)

    def transcribe(path: Path) -> tuple[str, list[Word]]:
        # Parakeet fails on inputs of many minutes (attention shape error), so
        # decode in fixed chunks like the real service will. Token timestamps are
        # not exposed uniformly across onnx-asr versions.
        samples = read_samples(path)
        step = 16000 * CHUNK_SECONDS
        texts = [str(engine.recognize(samples[i : i + step])) for i in range(0, len(samples), step)]
        return " ".join(texts), []

    return transcribe


def load_sherpa_onnx(model: str, threads: int):
    """`model` is a directory holding encoder/decoder/joiner .int8.onnx and tokens.txt."""
    import sherpa_onnx

    root = Path(model)
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(root / "encoder.int8.onnx"),
        decoder=str(root / "decoder.int8.onnx"),
        joiner=str(root / "joiner.int8.onnx"),
        tokens=str(root / "tokens.txt"),
        num_threads=threads,
        model_type="nemo_transducer",
        decoding_method="greedy_search",
    )

    def transcribe(path: Path) -> tuple[str, list[Word]]:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, read_samples(path))
        recognizer.decode_stream(stream)
        result = stream.result
        words: list[Word] = [
            (token, start, start) for token, start in zip(result.tokens, result.timestamps)
        ]
        return result.text, words

    return transcribe


ENGINES = {
    "faster-whisper": load_faster_whisper,
    "onnx-asr": load_onnx_asr,
    "sherpa-onnx": load_sherpa_onnx,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--engine", choices=sorted(ENGINES), required=True)
    parser.add_argument("--model", required=True, help="model name, or model directory for sherpa-onnx")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="hand-corrected transcript text for --audio")
    parser.add_argument("--glossary", type=Path, help="one name/term per line")
    parser.add_argument("--dump", type=Path, help="write hypothesis and word timestamps here (JSON)")
    args = parser.parse_args()

    seconds = audio_seconds(args.audio)
    transcribe = ENGINES[args.engine](args.model, args.threads)

    started = time.perf_counter()
    text, words = transcribe(args.audio)
    wall = time.perf_counter() - started

    report: dict[str, object] = {
        "engine": args.engine,
        "model": args.model,
        "threads": args.threads,
        "audio_seconds": round(seconds, 1),
        "wall_seconds": round(wall, 1),
        "realtime_factor": round(realtime_factor(seconds, wall), 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "words_with_timestamps": len(words),
    }
    if args.reference:
        reference = args.reference.read_text(encoding="utf-8")
        result = word_error_rate(reference, text)
        report["wer"] = round(result.wer, 4)
        report["wer_errors"] = {
            "substitutions": result.substitutions,
            "deletions": result.deletions,
            "insertions": result.insertions,
            "reference_words": result.reference_words,
        }
        if args.glossary:
            terms = [t.strip() for t in args.glossary.read_text(encoding="utf-8").splitlines() if t.strip()]
            scores = glossary_recall(terms, reference, text)
            hits = sum(h for h, _ in scores.values())
            total = sum(n for _, n in scores.values())
            report["glossary_recall"] = round(hits / total, 4) if total else None
            report["glossary_detail"] = {t: f"{h}/{n}" for t, (h, n) in scores.items()}
    if args.dump:
        args.dump.write_text(json.dumps({"text": text, "words": words}, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
