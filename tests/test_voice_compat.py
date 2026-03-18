from __future__ import annotations

import importlib
import sys
import threading
import types
import unittest
from unittest.mock import patch


class FakeOpusError(Exception):
    pass


class FakePacketRouter:
    def destroy_decoder(self, ssrc: int) -> None:
        self.destroyed_ssrcs.append(ssrc)

    def _do_run(self) -> None:
        raise AssertionError("compat patch was not applied")


class FakeWaiter:
    def __init__(self, items, end_event: threading.Event) -> None:
        self.items = items
        self._end_event = end_event

    def wait(self) -> None:
        self._end_event.set()


class FakeDecoder:
    def __init__(self, result=None, exc: Exception | None = None, ssrc: int = 1234) -> None:
        self.result = result
        self.exc = exc
        self.ssrc = ssrc
        self.reset_calls = 0

    def pop_data(self):
        if self.exc is not None:
            raise self.exc
        return self.result

    def reset(self) -> None:
        self.reset_calls += 1


class FakeSink:
    def __init__(self) -> None:
        self.writes: list[tuple[object, object]] = []

    def write(self, source, data) -> None:
        self.writes.append((source, data))


class VoiceCompatTests(unittest.TestCase):
    def test_apply_patch_drops_corrupted_packets_without_stopping_router(self) -> None:
        fake_discord = types.ModuleType("discord")
        fake_discord.opus = types.SimpleNamespace(OpusError=FakeOpusError)

        fake_router_module = types.ModuleType("discord.ext.voice_recv.router")
        fake_router_module.PacketRouter = FakePacketRouter

        with patch.dict(
            sys.modules,
            {
                "discord": fake_discord,
                "discord.ext": types.ModuleType("discord.ext"),
                "discord.ext.voice_recv": types.ModuleType("discord.ext.voice_recv"),
                "discord.ext.voice_recv.router": fake_router_module,
            },
        ):
            import scrollkeeper.voice_compat as voice_compat

            voice_compat = importlib.reload(voice_compat)
            voice_compat.apply_voice_recv_compatibility_patch()

            end_event = threading.Event()
            sink = FakeSink()
            good_data = types.SimpleNamespace(source="user", pcm=b"pcm")
            bad_decoder = FakeDecoder(exc=FakeOpusError("corrupted stream"), ssrc=7)
            good_decoder = FakeDecoder(result=good_data, ssrc=8)
            router = FakePacketRouter()
            router.destroyed_ssrcs = []
            router._end_thread = end_event
            router.waiter = FakeWaiter([bad_decoder, good_decoder], end_event)
            router._lock = threading.RLock()
            router.sink = sink

            router._do_run()

            self.assertEqual(bad_decoder.reset_calls, 1)
            self.assertEqual(sink.writes, [("user", good_data)])
            self.assertEqual(router.destroyed_ssrcs, [])

    def test_apply_patch_quarantines_ssrc_after_repeated_opus_errors(self) -> None:
        fake_discord = types.ModuleType("discord")
        fake_discord.opus = types.SimpleNamespace(OpusError=FakeOpusError)

        fake_router_module = types.ModuleType("discord.ext.voice_recv.router")
        fake_router_module.PacketRouter = FakePacketRouter

        with patch.dict(
            sys.modules,
            {
                "discord": fake_discord,
                "discord.ext": types.ModuleType("discord.ext"),
                "discord.ext.voice_recv": types.ModuleType("discord.ext.voice_recv"),
                "discord.ext.voice_recv.router": fake_router_module,
            },
        ):
            import scrollkeeper.voice_compat as voice_compat

            voice_compat = importlib.reload(voice_compat)
            voice_compat.apply_voice_recv_compatibility_patch()

            router = FakePacketRouter()
            router.destroyed_ssrcs = []
            router._lock = threading.RLock()
            router.sink = FakeSink()
            decoder = FakeDecoder(exc=FakeOpusError("corrupted stream"), ssrc=77)

            for _ in range(3):
                end_event = threading.Event()
                router._end_thread = end_event
                router.waiter = FakeWaiter([decoder], end_event)
                router._do_run()

            self.assertEqual(decoder.reset_calls, 2)
            self.assertEqual(router.destroyed_ssrcs, [77])


if __name__ == "__main__":
    unittest.main()
