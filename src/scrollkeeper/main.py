from __future__ import annotations

import logging
import sys

from .bot import build_bot
from .config import Settings
from .health import start_health_server


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
    logging.getLogger("scrollkeeper").setLevel(logging.INFO)
    logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
    logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)
    settings = Settings.load()
    settings.validate()
    bot = build_bot(settings)
    start_health_server(bot.heartbeat, settings.health_port)
    bot.run(settings.discord_bot_token)


if __name__ == "__main__":
    main()
