"""Download and verify the speech-to-text models at image build time.

Usage: python fetch_models.py /models
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
PARAKEET = (
    f"{RELEASES}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
    "157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad",
)
SILERO_VAD = (
    f"{RELEASES}/silero_vad.onnx",
    "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
)
PARAKEET_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def download(url: str, sha256: str, target: Path) -> None:
    print(f"Downloading {url}", flush=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, target.open("wb") as out:
        while block := response.read(1 << 20):
            digest.update(block)
            out.write(block)
    if digest.hexdigest() != sha256:
        raise SystemExit(f"Checksum mismatch for {url}: {digest.hexdigest()}")


def main(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    download(*SILERO_VAD, root / "silero_vad.onnx")

    archive = root / "parakeet.tar.bz2"
    download(*PARAKEET, archive)
    model_dir = root / "parakeet-tdt-0.6b-v2-int8"
    model_dir.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.isfile() and name in PARAKEET_FILES:
                with tar.extractfile(member) as src, (model_dir / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    archive.unlink()
    missing = [name for name in PARAKEET_FILES if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Archive is missing {missing}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
