from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _optional_env(name: str) -> str:
    return os.getenv(name, "").strip()


def _json_object_env(name: str) -> dict:
    raw = _optional_env(name)
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return value


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
    extract_chunk_chars: int = 24000
    wiki_export: bool = True
    llm_extra_body: dict = field(default_factory=dict)
    spool_dir: Path | None = None
    audio_retention_days: int = 0

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
            stt_model=os.getenv("SCROLLKEEPER_STT_MODEL", "parakeet-tdt-0.6b-v2").strip(),
            # One request transcribes a whole speaker track, so this must cover the longest track.
            stt_timeout_seconds=int(os.getenv("SCROLLKEEPER_STT_TIMEOUT_SECONDS", "7200")),
            llm_base_url=llm_base_url,
            llm_model=_optional_env("SCROLLKEEPER_LLM_MODEL"),
            llm_api_key=_optional_env("SCROLLKEEPER_LLM_API_KEY"),
            llm_timeout_seconds=int(os.getenv("SCROLLKEEPER_LLM_TIMEOUT_SECONDS", "900")),
            embed_base_url=_optional_env("SCROLLKEEPER_EMBED_BASE_URL").rstrip("/") or llm_base_url,
            embed_model=_optional_env("SCROLLKEEPER_EMBED_MODEL"),
            wait_notice_seconds=int(os.getenv("SCROLLKEEPER_WAIT_NOTICE_SECONDS", "20")),
            health_port=int(os.getenv("SCROLLKEEPER_HEALTH_PORT", "8080")),
            extract_chunk_chars=int(os.getenv("SCROLLKEEPER_EXTRACT_CHUNK_CHARS", "24000")),
            wiki_export=os.getenv("SCROLLKEEPER_WIKI_EXPORT", "1").strip().lower() not in {"0", "false", "no", "off"},
            llm_extra_body=_json_object_env("SCROLLKEEPER_LLM_EXTRA_BODY"),
            spool_dir=Path(spool).resolve() if (spool := _optional_env("SCROLLKEEPER_SPOOL_DIR")) else None,
            audio_retention_days=int(os.getenv("SCROLLKEEPER_AUDIO_RETENTION_DAYS", "0")),
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
