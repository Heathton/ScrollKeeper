from __future__ import annotations

import logging
from collections import defaultdict


log = logging.getLogger(__name__)
_PATCH_APPLIED = False
_MAX_CONSECUTIVE_OPUS_ERRORS = 3


def apply_voice_recv_compatibility_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    try:
        import discord
        from discord.ext.voice_recv.router import PacketRouter
    except ModuleNotFoundError:
        return

    original_do_run = PacketRouter._do_run

    def _patched_do_run(self: PacketRouter) -> None:
        if not hasattr(self, "_scrollkeeper_opus_error_counts"):
            self._scrollkeeper_opus_error_counts = defaultdict(int)

        while not self._end_thread.is_set():
            self.waiter.wait()
            with self._lock:
                for decoder in tuple(self.waiter.items):
                    ssrc = getattr(decoder, "ssrc", "unknown")
                    try:
                        data = decoder.pop_data()
                    except discord.opus.OpusError as exc:
                        self._scrollkeeper_opus_error_counts[ssrc] += 1
                        error_count = self._scrollkeeper_opus_error_counts[ssrc]
                        if error_count >= _MAX_CONSECUTIVE_OPUS_ERRORS:
                            log.warning(
                                "Quarantining voice SSRC %s after %s consecutive Opus decode failures: %s",
                                ssrc,
                                error_count,
                                exc,
                            )
                            self.destroy_decoder(ssrc)
                            self._scrollkeeper_opus_error_counts.pop(ssrc, None)
                            continue

                        log.warning(
                            "Dropping undecodable voice packet for ssrc=%s after OpusError (%s/%s): %s",
                            ssrc,
                            error_count,
                            _MAX_CONSECUTIVE_OPUS_ERRORS,
                            exc,
                        )
                        decoder.reset()
                        continue

                    if data is not None:
                        self._scrollkeeper_opus_error_counts.pop(ssrc, None)
                        self.sink.write(data.source, data)

    if PacketRouter._do_run is original_do_run:
        PacketRouter._do_run = _patched_do_run

    _PATCH_APPLIED = True
