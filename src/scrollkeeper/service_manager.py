from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import Settings


@dataclass(slots=True)
class ContainerSpec:
    name: str
    image: str
    port: int


class DockerServiceManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.whisper = ContainerSpec(
            name=settings.whisper_container,
            image=settings.whisper_image,
            port=settings.whisper_port,
        )
        self.ollama = ContainerSpec(
            name=settings.ollama_container,
            image=settings.ollama_image,
            port=settings.ollama_port,
        )
        self._ensure_whisper_lock = threading.Lock()
        self._ensure_ollama_lock = threading.Lock()
        self._ensure_network_lock = threading.Lock()
        self._ensure_whisper_image_lock = threading.Lock()
        self._model_pull_lock = threading.Lock()
        self._ollama_last_used = 0.0
        self._ollama_shutdown_task: asyncio.Task | None = None

    @property
    def whisper_base_url(self) -> str:
        return f"http://{self.whisper.name}:{self.whisper.port}"

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.ollama.name}:{self.ollama.port}"

    async def ensure_whisper_running(self) -> None:
        await asyncio.to_thread(self._ensure_whisper_running_sync)

    async def stop_whisper(self) -> None:
        await asyncio.to_thread(self._stop_container, self.whisper.name)

    async def ensure_ollama_running(self) -> None:
        await asyncio.to_thread(self._ensure_ollama_running_sync)
        self.mark_ollama_used()

    async def stop_ollama(self) -> None:
        await asyncio.to_thread(self._stop_container, self.ollama.name)

    async def prepare_for_transcription(self) -> None:
        if self.settings.gpu_policy != "serialize":
            return
        await self.ensure_ollama_running()
        await asyncio.to_thread(
            self._unload_ollama_models_sync,
            [self.settings.ollama_model, self.settings.ollama_embed_model],
        )

    async def recover_after_transcription(self) -> None:
        if self.settings.gpu_policy != "serialize":
            return
        await self.ensure_ollama_running()

    def mark_ollama_used(self) -> None:
        self._ollama_last_used = time.time()
        if self.settings.ollama_idle_timeout <= 0:
            return
        loop = asyncio.get_running_loop()
        if self._ollama_shutdown_task is None or self._ollama_shutdown_task.done():
            self._ollama_shutdown_task = loop.create_task(self._ollama_idle_monitor())

    async def shutdown(self) -> None:
        if self._ollama_shutdown_task:
            self._ollama_shutdown_task.cancel()
        await self.stop_whisper()
        await self.stop_ollama()

    async def _ollama_idle_monitor(self) -> None:
        while True:
            await asyncio.sleep(15)
            idle_for = time.time() - self._ollama_last_used
            if idle_for >= self.settings.ollama_idle_timeout:
                await self.stop_ollama()
                return

    def _ensure_whisper_running_sync(self) -> None:
        with self._ensure_whisper_lock:
            self._ensure_network_sync()
            self._ensure_whisper_image_sync()
            if self._is_container_running(self.whisper.name):
                self._wait_for_http(f"{self.whisper_base_url}/health", timeout_seconds=120)
                return
            command = [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self.whisper.name,
                "--network",
                self.settings.docker_network,
            ]
            if self.settings.enable_gpu:
                command.extend(["--gpus", "all"])
            command.extend(
                [
                    "-e",
                    f"WHISPER_MODEL={self.settings.whisper_model}",
                    "-e",
                    f"WHISPER_DEVICE={'cuda' if self.settings.enable_gpu else 'cpu'}",
                    self.whisper.image,
                ]
            )
            self._run_container(command)
            self._wait_for_http(f"{self.whisper_base_url}/health", timeout_seconds=300)

    def _ensure_ollama_running_sync(self) -> None:
        with self._ensure_ollama_lock:
            self._ensure_network_sync()
            if not self._is_container_running(self.ollama.name):
                command = [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    "--name",
                    self.ollama.name,
                    "--network",
                    self.settings.docker_network,
                ]
                if self.settings.enable_gpu:
                    command.extend(["--gpus", "all"])
                command.extend(
                    [
                        "-v",
                        "scrollkeeper_ollama:/root/.ollama",
                        self.ollama.image,
                    ]
                )
                self._run_container(command)
            self._wait_for_http(f"{self.ollama_base_url}/api/version", timeout_seconds=300)
            self._ensure_ollama_model_sync(self.settings.ollama_model)
            self._ensure_ollama_model_sync(self.settings.ollama_embed_model)

    def _ensure_ollama_model_sync(self, model: str) -> None:
        with self._model_pull_lock:
            if self._ollama_model_exists(model):
                return
            payload = json.dumps({"name": model}).encode("utf-8")
            self._http_post(f"{self.ollama_base_url}/api/pull", payload, timeout_seconds=3600)

    def _unload_ollama_models_sync(self, model_names: list[str]) -> None:
        for model in model_names:
            payload = json.dumps(
                {
                    "model": model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                }
            ).encode("utf-8")
            try:
                self._http_post(f"{self.ollama_base_url}/api/generate", payload, timeout_seconds=120)
            except Exception:
                # Best effort for VRAM relief: some models/endpoints may ignore unload requests.
                continue

    def _ensure_whisper_image_sync(self) -> None:
        with self._ensure_whisper_image_lock:
            if self._image_exists(self.whisper.image):
                return
            project_root = Path(__file__).resolve().parents[2]
            dockerfile = project_root / "docker" / "whisper" / "Dockerfile"
            context = project_root / "docker" / "whisper"
            self._run_container(
                [
                    "docker",
                    "build",
                    "-t",
                    self.whisper.image,
                    "-f",
                    str(dockerfile),
                    str(context),
                ]
            )

    def _ensure_network_sync(self) -> None:
        with self._ensure_network_lock:
            result = subprocess.run(
                ["docker", "network", "inspect", self.settings.docker_network],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                return
            self._run_container(["docker", "network", "create", self.settings.docker_network])

    def _ollama_model_exists(self, model: str) -> bool:
        payload = self._http_get(f"{self.ollama_base_url}/api/tags", timeout_seconds=60)
        models = json.loads(payload.decode("utf-8")).get("models", [])
        names = {item.get("name") for item in models}
        return model in names

    def _image_exists(self, image: str) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def _is_container_running(self, name: str) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _stop_container(self, name: str) -> None:
        subprocess.run(
            ["docker", "stop", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _run_container(self, command: list[str]) -> None:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"Command failed: {' '.join(command)}")

    def _wait_for_http(self, url: str, timeout_seconds: int) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                with urlopen(url, timeout=5) as response:
                    if response.status < 500:
                        return
            except URLError:
                time.sleep(2)
        raise RuntimeError(f"Service did not become ready in time: {url}")

    def _http_get(self, url: str, timeout_seconds: int) -> bytes:
        with urlopen(url, timeout=timeout_seconds) as response:
            return response.read()

    def _http_post(self, url: str, payload: bytes, timeout_seconds: int) -> bytes:
        request = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
