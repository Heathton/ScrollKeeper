from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel


app = FastAPI(title="ScrollKeeper Whisper")


@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    model_name = os.getenv("WHISPER_MODEL", "base.en")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    return WhisperModel(model_name, device=device, compute_type=compute_type)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    temp_path = Path("/tmp") / f"scrollkeeper_upload{suffix}"
    contents = await file.read()
    temp_path.write_bytes(contents)
    try:
        segments, _ = get_model().transcribe(str(temp_path), vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
