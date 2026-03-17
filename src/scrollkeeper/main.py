from __future__ import annotations

from .bot import build_bot
from .config import Settings


def main() -> None:
    settings = Settings.load()
    settings.validate()
    bot = build_bot(settings)
    bot.run(settings.discord_bot_token)


if __name__ == "__main__":
    main()
