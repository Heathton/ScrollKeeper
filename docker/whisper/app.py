from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import ctranslate2
from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel


app = FastAPI(title="ScrollKeeper Whisper")
log = logging.getLogger("scrollkeeper.whisper")
_MODEL_ACTUAL_DEVICE: str | None = None


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def should_use_vad_filter() -> bool:
    return _read_bool_env("WHISPER_VAD_FILTER", True)


@lru_cache(maxsize=1)
def runtime_probe() -> tuple[str, str, int]:
    requested_device = os.getenv("WHISPER_DEVICE", "cpu").strip().lower()
    cuda_device_count = ctranslate2.get_cuda_device_count() if requested_device == "cuda" else 0
    effective_device = requested_device
    if requested_device == "cuda" and cuda_device_count < 1:
        log.warning(
            "WHISPER_DEVICE=cuda requested but no CUDA devices detected; falling back to cpu.",
        )
        effective_device = "cpu"
    return requested_device, effective_device, cuda_device_count


@app.on_event("startup")
def startup_probe() -> None:
    requested_device, effective_device, cuda_device_count = runtime_probe()
    log.info(
        "Whisper runtime probe: requested_device=%s effective_device=%s cuda_device_count=%s vad_filter=%s",
        requested_device,
        effective_device,
        cuda_device_count,
        should_use_vad_filter(),
    )


@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    global _MODEL_ACTUAL_DEVICE
    model_name = os.getenv("WHISPER_MODEL", "base.en")
    requested_device, effective_device, cuda_device_count = runtime_probe()
    compute_type = os.getenv(
        "WHISPER_COMPUTE_TYPE",
        "float16" if effective_device == "cuda" else "int8",
    )
    model = WhisperModel(model_name, device=effective_device, compute_type=compute_type)
    _MODEL_ACTUAL_DEVICE = str(getattr(model.model, "device", effective_device))
    log.info(
        "Loaded Whisper model=%s requested_device=%s effective_device=%s actual_device=%s compute_type=%s cuda_device_count=%s",
        model_name,
        requested_device,
        effective_device,
        _MODEL_ACTUAL_DEVICE,
        compute_type,
        cuda_device_count,
    )
    return model


@app.get("/health")
def health() -> dict[str, str]:
    requested_device, effective_device, cuda_device_count = runtime_probe()
    response = {
        "status": "ok",
        "requested_device": requested_device,
        "effective_device": effective_device,
        "cuda_device_count": str(cuda_device_count),
        "vad_filter": "true" if should_use_vad_filter() else "false",
    }
    if _MODEL_ACTUAL_DEVICE is not None:
        response["actual_device"] = _MODEL_ACTUAL_DEVICE
    return response


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    temp_path = Path("/tmp") / f"scrollkeeper_upload{suffix}"
    contents = await file.read()
    temp_path.write_bytes(contents)
    try:
        segments, _ = get_model().transcribe(str(temp_path), vad_filter=should_use_vad_filter())
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
