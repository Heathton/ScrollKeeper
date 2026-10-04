from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _optional_env(name: str) -> str:
    return os.getenv(name, "").strip()


@dataclass(slots=True)
class Settings:
    discord_bot_token: str
    command_prefix: str
    data_dir: Path
    bot_name: str
    stt_base_url: str
    stt_model: str
    stt_timeout_seconds: int
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_timeout_seconds: int
    embed_base_url: str
    embed_model: str
    wait_notice_seconds: int
    health_port: int

    @classmethod
    def load(cls) -> "Settings":
        # A .env file is optional (local dev); in Kubernetes everything comes from the environment.
        load_dotenv()
        data_dir = Path(os.getenv("SCROLLKEEPER_DATA_DIR", "./data")).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        llm_base_url = _optional_env("SCROLLKEEPER_LLM_BASE_URL").rstrip("/")
        return cls(
            discord_bot_token=os.getenv("DISCORD_BOT_TOKEN", ""),
            command_prefix=os.getenv("DISCORD_COMMAND_PREFIX", "!"),
            data_dir=data_dir,
            bot_name=os.getenv("SCROLLKEEPER_BOT_NAME", "ScrollKeeper"),
            stt_base_url=_optional_env("SCROLLKEEPER_STT_BASE_URL").rstrip("/"),
            stt_model=os.getenv("SCROLLKEEPER_STT_MODEL", "whisper-1").strip(),
            stt_timeout_seconds=int(os.getenv("SCROLLKEEPER_STT_TIMEOUT_SECONDS", "600")),
            llm_base_url=llm_base_url,
            llm_model=_optional_env("SCROLLKEEPER_LLM_MODEL"),
            llm_api_key=_optional_env("SCROLLKEEPER_LLM_API_KEY"),
            llm_timeout_seconds=int(os.getenv("SCROLLKEEPER_LLM_TIMEOUT_SECONDS", "900")),
            embed_base_url=_optional_env("SCROLLKEEPER_EMBED_BASE_URL").rstrip("/") or llm_base_url,
            embed_model=_optional_env("SCROLLKEEPER_EMBED_MODEL"),
            wait_notice_seconds=int(os.getenv("SCROLLKEEPER_WAIT_NOTICE_SECONDS", "20")),
            health_port=int(os.getenv("SCROLLKEEPER_HEALTH_PORT", "8080")),
        )

    def validate(self) -> None:
        required = {
            "DISCORD_BOT_TOKEN": self.discord_bot_token,
            "SCROLLKEEPER_STT_BASE_URL": self.stt_base_url,
            "SCROLLKEEPER_LLM_BASE_URL": self.llm_base_url,
            "SCROLLKEEPER_LLM_MODEL": self.llm_model,
            "SCROLLKEEPER_EMBED_MODEL": self.embed_model,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing required environment variables: {joined}")
