"""Generation through an injected chat client, one request per situation."""
from __future__ import annotations

import asyncio
from typing import Dict, List


def build_messages(eval_prompt: str, system_prompt: str) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": eval_prompt})
    return messages


def chat_payload(eval_prompt: str, system_prompt: str, *, temperature: float, top_p: float,
                 top_k: int, seed: int, max_new_tokens: int) -> dict:
    """The chat-completions payload for one prompt. The client adds `model`."""
    return {
        "messages": build_messages(eval_prompt, system_prompt),
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "max_tokens": max_new_tokens,
    }


async def generate_openai(client, *, eval_prompts: List[str], system_prompt: str,
                          temperature: float, top_p: float, top_k: int, seed: int,
                          max_new_tokens: int) -> List[dict]:
    """One response per prompt: `{text, num_tokens, finish_reason, stop_reason}`.

    All prompts are issued at once; the client's semaphore bounds concurrency.
    """

    async def _one(eval_prompt: str) -> dict:
        payload = chat_payload(eval_prompt, system_prompt, temperature=temperature,
                               top_p=top_p, top_k=top_k, seed=seed,
                               max_new_tokens=max_new_tokens)
        resp = await client.chat(payload)
        choice = (resp.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        usage = resp.get("usage") or {}
        return {
            "text": text,
            "num_tokens": int(usage.get("completion_tokens") or 0),
            "finish_reason": choice.get("finish_reason"),
            "stop_reason": None,
        }

    return await asyncio.gather(*(_one(p) for p in eval_prompts))
