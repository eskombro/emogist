"""Load the model only while people are using it.

The model runs in its own llama-server process. Stopping that process is what reliably gives
all its memory back. A warm-up holds the model briefly, and only real sentences extend the hold.
"""

import asyncio
import json
import logging
import os
import socket
import tempfile
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psutil

log = logging.getLogger("emogist")

SYSTEM_PROMPT = "Convert the sentence into a short emoji sentence that keeps its gist. Answer with emojis only."
WARM_SENTENCE = "I can do it now."  # read through the model before reporting it as ready


def build_prompt(sentence: str) -> str:
    """Same text as the training chat template, with thinking turned off."""
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{sentence}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class Settings:
    model_path: Path = field(default_factory=lambda: Path(os.environ.get(
        "EMOGIST_MODEL", Path(__file__).parent / "models" / "emogist-qwen2b-Q8_0.gguf")))
    server_binary: str = field(default_factory=lambda: os.environ.get("EMOGIST_LLAMA_SERVER", "llama-server"))
    port: int = field(default_factory=lambda: _env_int("EMOGIST_MODEL_PORT", 18090))
    threads: int = field(default_factory=lambda: _env_int("EMOGIST_THREADS", 3))
    gpu_layers: int = field(default_factory=lambda: _env_int("EMOGIST_GPU_LAYERS", 0))
    warmup_hold: float = field(default_factory=lambda: _env_float("EMOGIST_WARMUP_HOLD", 90))
    sentence_hold: float = field(default_factory=lambda: _env_float("EMOGIST_SENTENCE_HOLD", 180))
    min_available_gb: float = field(default_factory=lambda: _env_float("EMOGIST_MIN_AVAILABLE_GB", 3))
    load_timeout: float = field(default_factory=lambda: _env_float("EMOGIST_LOAD_TIMEOUT", 60))
    generate_timeout: float = field(default_factory=lambda: _env_float("EMOGIST_GENERATE_TIMEOUT", 30))
    max_pending: int = field(default_factory=lambda: _env_int("EMOGIST_MAX_PENDING", 8))
    max_warmup_loads_per_hour: int = field(default_factory=lambda: _env_int("EMOGIST_MAX_WARMUP_LOADS", 20))


class Busy(Exception):
    """The demo can't take this request right now, and trying later may work."""


class ModelFailed(Exception):
    """The model process failed to start or answer."""


class ModelManager:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.state = "unloaded"
        self._process: asyncio.subprocess.Process | None = None
        self._log_file = None
        self._transition = asyncio.Lock()  # one load or unload at a time
        self._generation = asyncio.Lock()  # one sentence at a time
        self._pending = 0
        self._expires = 0.0
        self._loaded_at = 0.0
        self._warmup_loads: deque[float] = deque()
        self._background: set[asyncio.Task] = set()
        self._idle_task: asyncio.Task | None = None
        self._client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.settings.port}")

    async def start(self) -> None:
        self._idle_task = asyncio.create_task(self._unload_when_idle())

    async def close(self) -> None:
        if self._idle_task:
            self._idle_task.cancel()
        for task in list(self._background):
            task.cancel()
        async with self._transition:
            await self._stop("shutdown")
        await self._client.aclose()

    def warmup(self) -> str:
        """Start loading without waiting. A warm-up never extends the hold of a loaded model."""
        if self.state != "unloaded":
            return self.state
        now = time.monotonic()
        while self._warmup_loads and now - self._warmup_loads[0] > 3600:
            self._warmup_loads.popleft()
        if len(self._warmup_loads) >= self.settings.max_warmup_loads_per_hour:
            log.warning("warmup_skipped reason=hourly_limit")
            return self.state
        self._warmup_loads.append(now)
        self.state = "loading"
        task = asyncio.create_task(self._warmup_task())
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return self.state

    async def _warmup_task(self) -> None:
        try:
            await self._ensure_loaded("warmup")
        except (Busy, ModelFailed) as error:
            log.warning("warmup_failed reason=%s", error)

    def queue_full(self) -> bool:
        return self._pending >= self.settings.max_pending

    async def generate(self, sentence: str, usage: dict | None = None) -> str:
        async with aclosing(self.stream(sentence, usage)) as pieces:
            return "".join([piece async for piece in pieces]).strip()

    async def stream(self, sentence: str, usage: dict | None = None) -> AsyncIterator[str]:
        """Yield the answer piece by piece and fill `usage` with token counts at the end.

        Closing it early also stops the model.
        """
        if self.queue_full():
            raise Busy("queue full")
        self._pending += 1
        try:
            async with self._generation:
                await self._ensure_loaded("sentence")
                started = time.perf_counter()
                try:
                    async with aclosing(self._stream_completion(sentence, usage)) as pieces:
                        async for piece in pieces:
                            yield piece
                finally:
                    self._expires = time.monotonic() + self.settings.sentence_hold
                log.info("sentence_done generate_ms=%d", (time.perf_counter() - started) * 1000)
        finally:
            self._pending -= 1

    async def _stream_completion(self, sentence: str, usage: dict | None = None) -> AsyncIterator[str]:
        deadline = time.monotonic() + self.settings.generate_timeout
        try:
            async with self._client.stream("POST", "/completion", timeout=self.settings.generate_timeout, json={
                    "prompt": build_prompt(sentence), "n_predict": 60, "temperature": 0, "top_k": 1,
                    "cache_prompt": False, "seed": 1, "stream": True}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if time.monotonic() > deadline:
                        raise ModelFailed("model request timed out")
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[len("data: "):])
                    if data.get("content"):
                        yield data["content"]
                    if data.get("stop"):
                        if usage is not None:
                            usage.update(input=data.get("tokens_evaluated"), output=data.get("tokens_predicted"))
                        return
        except (httpx.HTTPError, ValueError) as error:
            raise ModelFailed(f"model request failed: {type(error).__name__}") from error
        raise ModelFailed("model stream ended early")

    async def _ensure_loaded(self, trigger: str) -> None:
        async with self._transition:
            if self.state == "ready" and self._process and self._process.returncode is None:
                return
            if self._process:
                log.warning("model_process_lost")
                await self._stop("process_lost")
            await self._load(trigger)

    async def _load(self, trigger: str) -> None:
        settings = self.settings
        self.state = "loading"
        started = time.perf_counter()
        try:
            available = psutil.virtual_memory().available / 1024 ** 3
            if available < settings.min_available_gb:
                log.warning("load_refused available_gb=%.2f minimum_gb=%.2f", available, settings.min_available_gb)
                raise Busy("not enough memory")
            if not settings.model_path.exists():
                raise ModelFailed(f"model file not found: {settings.model_path}")
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", settings.port)) == 0:
                    raise ModelFailed(f"port {settings.port} is already in use")
            log_path = Path(tempfile.gettempdir()) / "emogist-llama-server.log"
            self._log_file = open(log_path, "w")
            try:
                self._process = await asyncio.create_subprocess_exec(
                    settings.server_binary, "-m", str(settings.model_path), "--host", "127.0.0.1",
                    "--port", str(settings.port), "-ngl", str(settings.gpu_layers), "-t", str(settings.threads),
                    "-tb", str(settings.threads), "-c", "512", "-np", "1", "--no-repack", "--cache-ram", "0",
                    "--no-webui", stdout=self._log_file, stderr=self._log_file)
            except OSError as error:
                raise ModelFailed(f"could not start llama-server: {error}") from error
            await asyncio.wait_for(self._wait_until_healthy(), settings.load_timeout)
            # Reads the model from disk before visitors wait on it, so it gets the load timeout.
            await self._complete(WARM_SENTENCE, settings.load_timeout)
            load_seconds = time.perf_counter() - started
            self._loaded_at = time.monotonic()
            hold = settings.sentence_hold if trigger == "sentence" else settings.warmup_hold
            self._expires = self._loaded_at + hold
            self.state = "ready"
            log.info("model_loaded trigger=%s load_ms=%d available_gb_before=%.2f",
                     trigger, load_seconds * 1000, available)
        except asyncio.TimeoutError as error:
            await self._stop("load_failed")
            raise ModelFailed("model load timed out") from error
        except BaseException:
            await self._stop("load_failed")
            raise

    async def _wait_until_healthy(self) -> None:
        while True:
            if self._process.returncode is not None:
                raise ModelFailed(f"llama-server exited with code {self._process.returncode}")
            try:
                if (await self._client.get("/health", timeout=1)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)

    async def _complete(self, sentence: str, timeout: float | None = None) -> str:
        try:
            response = await self._client.post("/completion", timeout=timeout or self.settings.generate_timeout, json={
                "prompt": build_prompt(sentence), "n_predict": 60, "temperature": 0, "top_k": 1,
                "cache_prompt": False, "seed": 1})
            response.raise_for_status()
            return response.json()["content"].strip()
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise ModelFailed(f"model request failed: {type(error).__name__}") from error

    async def _stop(self, reason: str) -> None:
        process, self._process = self._process, None
        if process and process.returncode is None:
            self.state = "unloading"
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if process:
            log.info("model_unloaded reason=%s loaded_for_s=%d", reason,
                     time.monotonic() - self._loaded_at if self._loaded_at else 0)
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        self._loaded_at = 0.0
        self.state = "unloaded"

    async def _unload_when_idle(self) -> None:
        while True:
            await asyncio.sleep(1)
            if self.state != "ready" or self._pending or time.monotonic() < self._expires:
                continue
            async with self._transition:
                if self.state == "ready" and not self._pending and time.monotonic() >= self._expires:
                    await self._stop("idle")
