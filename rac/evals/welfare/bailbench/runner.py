"""Run BailBench through an injected chat client.

The single-turn mechanisms put the bail string in the system prompt or the
bail tool in `tools`; the two bail-prompt types run a second turn with the
well-being check, substituting a placeholder for an empty first reply.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from .loader import DATASETS, load_dataset
from .prompts import (BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE,
                      BAIL_STR_TYPE, BAIL_TOOL_TYPE, BAIL_TYPES, REFUSAL_PLACEHOLDER,
                      ablations_for, bail_prompt, bail_string_prompt, bail_tool_description,
                      bail_tool_name)
from .refusal import MinosRefusalClassifier, summarize_refusal
from .scoring import classify, summarize_bail

BAIL_PROMPT_TYPES = (BAIL_PROMPT_BAIL_FIRST_TYPE, BAIL_PROMPT_CONTINUE_FIRST_TYPE)


@dataclass
class BailBenchConfig:
    model: str
    bail_type: str = BAIL_STR_TYPE
    ablation: str = ""
    dataset: str = "bailbench"
    refusal_classifier: bool = False
    num_prompts: Optional[int] = None       # None = all
    num_rollouts: int = 10
    system_prompt: Optional[str] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: Optional[int] = None
    max_new_tokens: int = 2048
    concurrency: int = 32
    thinking: bool = False                  # recorded; the client applies it
    output: Optional[str] = None            # JSON summary path
    raw_output: Optional[str] = None        # JSONL of every response

    def __post_init__(self):
        if self.bail_type not in BAIL_TYPES:
            raise ValueError(f"bail_type must be one of {BAIL_TYPES}, got {self.bail_type!r}")
        if self.num_rollouts < 1:
            raise ValueError("num_rollouts must be >= 1")
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        allowed = ablations_for(self.bail_type)
        if self.ablation not in allowed:
            raise ValueError(f"ablation {self.ablation!r} invalid for {self.bail_type}; "
                             f"allowed: {allowed}")


@dataclass
class BailBenchResult:
    config: dict
    metrics: dict
    rows: List[dict] = field(default_factory=list)


def _bail_tool_spec(model: str, ablation: str = "") -> dict:
    return {
        "type": "function",
        "function": {
            "name": bail_tool_name(ablation),
            "description": bail_tool_description(model, ablation),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


def build_messages(cfg: BailBenchConfig, prompt: str) -> List[dict]:
    """First-turn messages for one prompt."""
    system = cfg.system_prompt
    if cfg.bail_type == BAIL_STR_TYPE:
        bail_str = bail_string_prompt(cfg.model, cfg.ablation)
        system = f"{system}\n\n{bail_str}" if system else bail_str
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return messages


def _payload(cfg: BailBenchConfig, messages: List[dict]) -> dict:
    payload = {
        "messages": messages,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_new_tokens,
    }
    if cfg.top_k and cfg.top_k > 0:
        payload["top_k"] = cfg.top_k
    if cfg.seed is not None:
        payload["seed"] = cfg.seed
    if cfg.bail_type == BAIL_TOOL_TYPE:
        payload["tools"] = [_bail_tool_spec(cfg.model, cfg.ablation)]
    return payload


def _message_of(response: dict) -> dict:
    choices = response.get("choices")
    if not choices:
        raise ValueError(f"model response has no choices: {response!r}")
    return choices[0].get("message") or {}


async def _one_rollout(cfg, client, item, rollout_idx, sem) -> dict:
    async with sem:
        messages = build_messages(cfg, item["content"])
        response = await client.chat(_payload(cfg, messages))
        message = _message_of(response)
        text = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []

        if cfg.bail_type in BAIL_PROMPT_TYPES:
            first_turn = text
            messages = messages + [
                {"role": "assistant", "content": text or REFUSAL_PLACEHOLDER},
                {"role": "user", "content": bail_prompt(cfg.bail_type, cfg.ablation)},
            ]
            response = await client.chat(_payload(cfg, messages))
            text = _message_of(response).get("content") or ""
            tool_calls = []
        else:
            first_turn = None

    verdict = classify(cfg.bail_type, text, tool_calls, ablation=cfg.ablation)
    return {
        "prompt": item["content"],
        "category": item["category"],
        "subcategory": item["subcategory"],
        "rollout": rollout_idx,
        "bail_type": cfg.bail_type,
        "ablation": cfg.ablation,
        "response": text,
        "first_turn_response": first_turn,
        "tool_calls": list(tool_calls),
        **verdict,
    }


async def run_bailbench(cfg: BailBenchConfig, client, classifier=None) -> BailBenchResult:
    """Run one arm and return overall and per-category bail rates, plus refusal
    rates when a classifier is requested."""
    items = load_dataset(cfg.dataset)
    if cfg.num_prompts is not None:
        if cfg.num_prompts < 1:
            raise ValueError("num_prompts must be >= 1")
        items = items[: cfg.num_prompts]

    sem = asyncio.Semaphore(cfg.concurrency)
    tasks = [
        _one_rollout(cfg, client, item, r, sem)
        for item in items
        for r in range(cfg.num_rollouts)
    ]
    rows = list(await asyncio.gather(*tasks))

    if cfg.refusal_classifier or classifier is not None:
        clf = classifier if classifier is not None else MinosRefusalClassifier()
        # Refusal is judged on the model's answer, so bail-prompt arms are scored
        # on the first turn rather than the well-being reply.
        pairs = [(r["prompt"], r["first_turn_response"] if r["first_turn_response"] is not None
                  else r["response"]) for r in rows]
        for row, pr in zip(rows, clf.score(pairs)):
            row["refuse_pr"] = pr

    metrics = summarize_bail(rows)
    metrics.update(summarize_refusal(rows))
    metrics["dataset"] = cfg.dataset
    metrics["ablation"] = cfg.ablation
    metrics["num_prompts"] = len(items)
    metrics["num_rollouts"] = cfg.num_rollouts
    result = BailBenchResult(config=asdict(cfg), metrics=metrics, rows=rows)

    if cfg.output:
        out = Path(cfg.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"config": result.config, "metrics": result.metrics},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
    if cfg.raw_output:
        raw = Path(cfg.raw_output)
        raw.parent.mkdir(parents=True, exist_ok=True)
        with raw.open("w", encoding="utf-8") as f:
            for row in result.rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result
