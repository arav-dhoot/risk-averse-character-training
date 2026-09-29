"""Async client for OpenAI-compatible chat servers (vLLM in this repo).

Responses are cached on disk keyed by the request payload, so an interrupted
run resumes where it stopped and a repeated run costs nothing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class UnsupportedRequestError(RuntimeError):
    """A non-retryable 4xx, usually a feature the server does not support."""


@dataclass
class Endpoint:
    base_url: str
    model: str
    api_key: str | None = None

    def headers(self) -> dict[str, str]:
        key = self.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        return {"Authorization": f"Bearer {key}"}


@dataclass
class ChatClient:
    endpoint: Endpoint
    concurrency: int = 32
    max_retries: int = 6
    timeout: float = 1800.0
    cache_path: Path | None = None
    # Forwarded to the server's chat template on every chat request
    # (e.g. {"enable_thinking": False}); part of the cache key.
    chat_template_kwargs: dict | None = None

    _sem: asyncio.Semaphore = field(init=False, repr=False)
    _cache: dict[str, dict] = field(init=False, repr=False)
    _cache_lock: asyncio.Lock = field(init=False, repr=False)
    _http: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.concurrency)
        self._cache_lock = asyncio.Lock()
        self._cache = {}
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open() as f:
                for line in f:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["response"]
        self._http = httpx.AsyncClient(timeout=self.timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat(self, payload: dict, *, cache_salt: str | None = None) -> dict:
        """POST /chat/completions. `payload` must not carry `model`."""
        if self.chat_template_kwargs is not None:
            payload = {**payload, "chat_template_kwargs": dict(self.chat_template_kwargs)}
        return await self._post("/chat/completions", payload, cache_salt=cache_salt)

    async def completions(self, payload: dict) -> dict:
        """POST /completions (raw text, no chat template)."""
        return await self._post("/completions", payload)

    @staticmethod
    def _key(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    async def _post(self, route: str, payload: dict, cache_salt: str | None = None) -> dict:
        payload = {"model": self.endpoint.model, **payload}
        key_parts = {"route": route, **payload}
        if cache_salt is not None:
            key_parts["cache_salt"] = cache_salt
        key = self._key(key_parts)
        if key in self._cache:
            return self._cache[key]

        url = self.endpoint.base_url.rstrip("/") + route
        delay = 1.0
        last_err: Exception | None = None
        async with self._sem:
            for _ in range(self.max_retries):
                try:
                    resp = await self._http.post(url, json=payload, headers=self.endpoint.headers())
                except httpx.HTTPError as e:
                    last_err = e
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                if resp.status_code in RETRYABLE_STATUS:
                    last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                if resp.status_code >= 400:
                    raise UnsupportedRequestError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                await self._store(key, data)
                return data
        raise RuntimeError(f"request failed after {self.max_retries} retries: {last_err}")

    async def _store(self, key: str, response: dict) -> None:
        async with self._cache_lock:
            self._cache[key] = response
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a") as f:
                    f.write(json.dumps({"key": key, "response": response}) + "\n")


def make_client(base_url: str, model: str, *, concurrency: int = 32,
                cache_path: Path | None = None, enable_thinking: bool | None = None,
                timeout: float = 1800.0) -> ChatClient:
    """A client for `model` as served at `base_url`.

    `enable_thinking` is passed to the chat template (Qwen3-style thinking
    switch); leave it None for templates without one.
    """
    kwargs = None if enable_thinking is None else {"enable_thinking": enable_thinking}
    return ChatClient(endpoint=Endpoint(base_url=base_url, model=model),
                      concurrency=concurrency, cache_path=cache_path, timeout=timeout,
                      chat_template_kwargs=kwargs)
