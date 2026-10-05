"""Text embeddings computed in-process on CPU with ONNX Runtime (#8).

Model files are downloaded once from Hugging Face at a pinned revision, checked against their
SHA-256, and kept under the model directory (on the data volume by default), so later starts
need no network. Queries and documents are formatted differently, as each model expects: the
query side carries the retrieval instruction.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


log = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = (60, 300)


@dataclass(frozen=True, slots=True)
class EmbeddingModelSpec:
    name: str
    repo: str
    revision: str
    # Published path in the repo -> SHA-256 of its content.
    files: dict[str, str]
    onnx_file: str
    dim: int
    # "sentence_embedding": the graph returns the pooled vector; "last_token": pool the last
    # position of `last_hidden_state` (decoder-style models).
    pooling: str
    query_format: str
    document_format: str
    max_tokens: int
    # Below this cosine similarity, the best vector match is treated as unrelated to the question.
    min_similarity: float


EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    spec.name: spec
    for spec in (
        # Gemma license (https://ai.google.dev/gemma/terms). int8, 768 dimensions, ~310 MB.
        EmbeddingModelSpec(
            name="embeddinggemma-300m",
            repo="onnx-community/embeddinggemma-300m-ONNX",
            revision="5090578d9565bb06545b4552f76e6bc2c93e4a66",
            files={
                "onnx/model_quantized.onnx": "172efde319fe1542dc41f31be6154910b05b78f7a861c265c4600eec906bd6d8",
                "onnx/model_quantized.onnx_data": "705626e28e4c23c82ade34566b4197d97f534c12275fa406dfb71e9937d388c0",
                "tokenizer.json": "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47",
            },
            onnx_file="onnx/model_quantized.onnx",
            dim=768,
            pooling="sentence_embedding",
            query_format="task: search result | query: {text}",
            document_format="title: {title} | text: {text}",
            max_tokens=2048,
            min_similarity=0.3,
        ),
        # Apache-2.0. int8, 1024 dimensions, ~615 MB; about 2.5x slower than EmbeddingGemma on CPU.
        EmbeddingModelSpec(
            name="qwen3-embedding-0.6b",
            repo="onnx-community/Qwen3-Embedding-0.6B-ONNX",
            revision="c25a394dd583836952667c12f008335071b3f43d",
            files={
                "onnx/model_int8.onnx": "6d0ea863f78b4a84afa3c7fcba1ec341572b5e28121aef77b7092b1dfdf679c7",
                "tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
            },
            onnx_file="onnx/model_int8.onnx",
            dim=1024,
            pooling="last_token",
            query_format=(
                "Instruct: Given a question about a tabletop campaign, retrieve notes that answer it\nQuery:{text}"
            ),
            document_format="{title}\n{text}",
            max_tokens=2048,
            min_similarity=0.4,
        ),
    )
}
DEFAULT_EMBED_MODEL = "embeddinggemma-300m"


class LocalEmbedder:
    """Embeds queries and documents with a local ONNX model. Blocking: call from a worker thread.

    Calls are serialized, so a question waits for at most one document embedding rather than
    competing with it for the same cores.
    """

    def __init__(self, spec: EmbeddingModelSpec, model_dir: Path, threads: int = 2) -> None:
        self.spec = spec
        self.model_dir = model_dir
        self.threads = max(1, threads)
        self._lock = threading.Lock()
        self._session: Any = None
        self._tokenizer: Any = None

    @property
    def files_dir(self) -> Path:
        return self.model_dir / f"{self.spec.name}-{self.spec.revision[:12]}"

    @property
    def ready(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        """Download the model if needed and load it. Safe to call again after a failure."""
        with self._lock:
            if self._session is not None:
                return
            fetch_model(self.spec, self.files_dir)
            import onnxruntime
            from tokenizers import Tokenizer

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            self._tokenizer = Tokenizer.from_file(str(self.files_dir / "tokenizer.json"))
            self._session = onnxruntime.InferenceSession(
                str(self.files_dir / self.spec.onnx_file), options, providers=["CPUExecutionProvider"]
            )
            log.info("Loaded embedding model %s (%s threads)", self.spec.name, self.threads)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(self.spec.query_format.format(text=text.strip()))

    def embed_document(self, title: str, text: str) -> list[float]:
        return self._embed(self.spec.document_format.format(title=title.strip() or "none", text=text.strip()))

    def _embed(self, text: str) -> list[float]:
        import numpy as np

        self.load()
        with self._lock:
            ids = truncate_ids(self._tokenizer.encode(text).ids, self.spec.max_tokens)
            input_ids = np.array([ids], dtype=np.int64)
            feed = {"input_ids": input_ids, "attention_mask": np.ones_like(input_ids)}
            for model_input in self._session.get_inputs():
                if model_input.name == "position_ids":
                    feed["position_ids"] = np.arange(len(ids), dtype=np.int64)[None, :]
                elif model_input.name.startswith("past_key_values"):
                    # Decoder exports take a KV cache; an empty one means "no earlier tokens".
                    _, heads, _, head_dim = model_input.shape
                    feed[model_input.name] = np.zeros((1, heads, 0, head_dim), dtype=np.float32)
            if self.spec.pooling == "sentence_embedding":
                vector = self._session.run(["sentence_embedding"], feed)[0][0]
            else:
                vector = self._session.run(["last_hidden_state"], feed)[0][0, -1]
        norm = float(np.linalg.norm(vector))
        return [float(x) / norm for x in vector] if norm else [float(x) for x in vector]


def truncate_ids(ids: list[int], max_tokens: int) -> list[int]:
    """Cut a long input to `max_tokens`, keeping the final token (the end-of-text marker that
    last-token pooling reads)."""
    if len(ids) <= max_tokens:
        return list(ids)
    return list(ids[: max_tokens - 1]) + [ids[-1]]


def fetch_model(spec: EmbeddingModelSpec, target: Path) -> None:
    """Download any missing model file into `target`, verifying each file's SHA-256."""
    for path, sha256 in spec.files.items():
        destination = target / path
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{spec.repo}/resolve/{spec.revision}/{path}"
        log.info("Downloading embedding model file %s", url)
        partial = destination.with_name(destination.name + ".part")
        digest = hashlib.sha256()
        try:
            with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1 << 20):
                        handle.write(block)
                        digest.update(block)
            if digest.hexdigest() != sha256:
                raise RuntimeError(f"Checksum mismatch for {spec.repo}/{path}: got {digest.hexdigest()}")
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    """`python -m scrollkeeper.embeddings <model-dir> [model]`: pre-download a model (for a
    pre-provisioned or read-only `SCROLLKEEPER_EMBED_MODEL_DIR`)."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit(f"usage: python -m scrollkeeper.embeddings <model-dir> [{'|'.join(EMBEDDING_MODELS)}]")
    logging.basicConfig(level=logging.INFO)
    spec = EMBEDDING_MODELS[args[1] if len(args) > 1 else DEFAULT_EMBED_MODEL]
    embedder = LocalEmbedder(spec, Path(args[0]))
    fetch_model(spec, embedder.files_dir)
    print(embedder.files_dir)


if __name__ == "__main__":
    main()
