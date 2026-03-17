from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from scrollkeeper.config import Settings
from scrollkeeper.service_manager import DockerServiceManager


class FakeServiceManager(DockerServiceManager):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.ensure_ollama_calls = 0
        self.unload_calls: list[list[str]] = []
        self.stopped_containers: list[str] = []

    def _ensure_ollama_running_sync(self) -> None:
        self.ensure_ollama_calls += 1

    def _unload_ollama_models_sync(self, model_names: list[str]) -> None:
        self.unload_calls.append(model_names)

    def _stop_container(self, name: str) -> None:
        self.stopped_containers.append(name)

    def _ensure_whisper_running_sync(self) -> None:
        return


def build_settings(gpu_policy: str = "concurrent", idle_timeout: int = 0) -> Settings:
    return Settings(
        discord_bot_token="token",
        command_prefix="!",
        data_dir=Path("/tmp/scrollkeeper-tests"),
        bot_name="ScrollKeeper",
        docker_network="scrollkeeper-net",
        whisper_image="scrollkeeper-whisper:latest",
        whisper_container="scrollkeeper-whisper",
        whisper_port=9000,
        whisper_model="small.en",
        ollama_image="ollama/ollama:latest",
        ollama_container="scrollkeeper-ollama",
        ollama_port=11434,
        ollama_model="qwen3.5:9b",
        ollama_embed_model="qwen3-embedding:4b",
        ollama_idle_timeout=idle_timeout,
        gpu_policy=gpu_policy,
        enable_gpu=True,
    )


class ServiceManagerPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_for_transcription_concurrent_policy_does_not_unload(self) -> None:
        manager = FakeServiceManager(build_settings(gpu_policy="concurrent", idle_timeout=0))
        await manager.prepare_for_transcription()
        self.assertEqual(manager.ensure_ollama_calls, 0)
        self.assertEqual(manager.unload_calls, [])

    async def test_prepare_for_transcription_serialize_policy_unloads_models(self) -> None:
        manager = FakeServiceManager(build_settings(gpu_policy="serialize", idle_timeout=0))
        await manager.prepare_for_transcription()
        self.assertEqual(manager.ensure_ollama_calls, 1)
        self.assertEqual(len(manager.unload_calls), 1)
        self.assertEqual(
            manager.unload_calls[0],
            [manager.settings.ollama_model, manager.settings.ollama_embed_model],
        )

    async def test_idle_timeout_zero_disables_shutdown_task(self) -> None:
        manager = FakeServiceManager(build_settings(gpu_policy="concurrent", idle_timeout=0))
        await manager.ensure_ollama_running()
        self.assertIsNone(manager._ollama_shutdown_task)

    async def test_idle_timeout_positive_stops_ollama_after_inactivity(self) -> None:
        manager = FakeServiceManager(build_settings(gpu_policy="concurrent", idle_timeout=1))
        await manager.ensure_ollama_running()
        self.assertIsNotNone(manager._ollama_shutdown_task)
        await asyncio.sleep(16)
        self.assertIn(manager.settings.ollama_container, manager.stopped_containers)


if __name__ == "__main__":
    unittest.main()
