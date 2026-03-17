from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    discord_bot_token: str
    command_prefix: str
    data_dir: Path
    bot_name: str
    docker_network: str
    whisper_image: str
    whisper_container: str
    whisper_port: int
    whisper_model: str
    ollama_image: str
    ollama_container: str
    ollama_port: int
    ollama_model: str
    ollama_embed_model: str
    ollama_idle_timeout: int
    gpu_policy: str
    enable_gpu: bool

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        data_dir = Path(os.getenv("SCROLLKEEPER_DATA_DIR", "./data")).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            discord_bot_token=os.getenv("DISCORD_BOT_TOKEN", ""),
            command_prefix=os.getenv("DISCORD_COMMAND_PREFIX", "!"),
            data_dir=data_dir,
            bot_name=os.getenv("SCROLLKEEPER_BOT_NAME", "ScrollKeeper"),
            docker_network=os.getenv("SCROLLKEEPER_DOCKER_NETWORK", "scrollkeeper-net"),
            whisper_image=os.getenv("SCROLLKEEPER_WHISPER_IMAGE", "scrollkeeper-whisper:latest"),
            whisper_container=os.getenv("SCROLLKEEPER_WHISPER_CONTAINER", "scrollkeeper-whisper"),
            whisper_port=int(os.getenv("SCROLLKEEPER_WHISPER_PORT", "9000")),
            whisper_model=os.getenv("SCROLLKEEPER_WHISPER_MODEL", "small.en"),
            ollama_image=os.getenv("SCROLLKEEPER_OLLAMA_IMAGE", "ollama/ollama:latest"),
            ollama_container=os.getenv("SCROLLKEEPER_OLLAMA_CONTAINER", "scrollkeeper-ollama"),
            ollama_port=int(os.getenv("SCROLLKEEPER_OLLAMA_PORT", "11434")),
            ollama_model=os.getenv("SCROLLKEEPER_OLLAMA_MODEL", "qwen3.5:9b"),
            ollama_embed_model=os.getenv(
                "SCROLLKEEPER_OLLAMA_EMBED_MODEL",
                "qwen3-embedding:4b",
            ),
            ollama_idle_timeout=int(os.getenv("SCROLLKEEPER_OLLAMA_IDLE_TIMEOUT", "0")),
            gpu_policy=os.getenv("SCROLLKEEPER_GPU_POLICY", "concurrent").strip().lower(),
            enable_gpu=os.getenv("SCROLLKEEPER_ENABLE_GPU", "true").strip().lower() in {"1", "true", "yes", "on"},
        )

    def validate(self) -> None:
        missing = []
        if not self.discord_bot_token:
            missing.append("DISCORD_BOT_TOKEN")
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing required environment variables: {joined}")
        if self.gpu_policy not in {"concurrent", "serialize"}:
            raise RuntimeError("SCROLLKEEPER_GPU_POLICY must be either 'concurrent' or 'serialize'.")
