"""Comparison and observation types, and the oracles that turn a pairwise
comparison into P(item_i > item_j).

`LogprobOracle` reproduces upstream's local-logits oracle through the server's
`/completions` route: it scores `render(question) + "<answer>" + "A"` and the
same with `"B"` teacher-forced, and softmaxes the two. That is continuous and
noise-free; `SampleOracle` counts picks over sampled completions instead and
quantises p to 1/n_samples.

Two details matter. Appending a letter to a prompt ending in `>` can merge
tokens (for Qwen3, `<answer>A` tokenizes as `<`, `answer`, `>A`), so the two
variants are aligned against each other and scored from their first divergent
token. And `p_a` is P(pick slot A), which becomes P(item_i > item_j) only after
undoing the slot placement and the question's valence.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .prompts import Question


@dataclass
class Comparison:
    i: int
    j: int
    item_i: str
    item_j: str
    question: Question
    slot_a: str                      # "i" or "j": which item renders in slot A
    phase: str = "elo"
    round: Optional[int] = None

    def __post_init__(self):
        if self.slot_a not in ("i", "j"):
            raise ValueError(f"slot_a must be 'i' or 'j', got {self.slot_a!r}")
        if self.i == self.j:
            raise ValueError(f"self-comparison on index {self.i}")

    def rendered(self) -> str:
        """The user turn, with the items placed per `slot_a`."""
        if self.slot_a == "i":
            return self.question.render(self.item_i, self.item_j)
        return self.question.render(self.item_j, self.item_i)


@dataclass
class EdgeObservation:
    i: int
    j: int
    p_util: float                    # P(item_i > item_j), orientation applied
    mode: str
    question_id: str
    valence: int
    slot_a: str
    phase: str
    round: Optional[int] = None
    raw: Dict = field(default_factory=dict)

    def to_record(self, items: Sequence[str]) -> dict:
        return {
            "i": self.i, "j": self.j,
            "item_i": items[self.i], "item_j": items[self.j],
            "p_util": self.p_util, "mode": self.mode,
            "question_id": self.question_id, "valence": self.valence,
            "slot_a": self.slot_a, "phase": self.phase, "round": self.round,
            **self.raw,
        }


def p_util_from_pick(p_a: float, slot_a: str, question: Question) -> float:
    """P(pick slot A) -> P(item_i > item_j): undo slot placement, then valence."""
    p_pick_i = p_a if slot_a == "i" else 1.0 - p_a
    return question.orient(p_pick_i)


def wins_to_items(a_count: float, b_count: float, slot_a: str, valence: int) -> tuple:
    """Slot-A/B pick counts -> (wins_i, wins_j) in item orientation."""
    wins_i = a_count if slot_a == "i" else b_count
    wins_j = b_count if slot_a == "i" else a_count
    if valence == -1:
        wins_i, wins_j = wins_j, wins_i
    return wins_i, wins_j


def make_chat_renderer(base_model: str, *, enable_thinking: bool = False,
                       system_prompt: Optional[str] = None):
    """Build `render(user_text, assistant_prefix) -> prompt text` from the
    model's chat template, since `/completions` takes raw text. Mirrors
    upstream's fallback chain: drop `enable_thinking` for templates that reject
    it, then fall back to a plain `Role: content` transcript."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)

    def _apply(messages) -> str:
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking)
        except Exception:
            try:
                return tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
                parts.append("Assistant:")
                return "\n".join(parts)

    def render(user_text: str, assistant_prefix: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return _apply(messages) + assistant_prefix

    return render, tok


def _entry_logprob(entry) -> float:
    """The logprob in one vLLM-shaped `prompt_logprobs` entry."""
    if entry is None:
        raise ValueError("prompt_logprobs entry is None inside the scored continuation")
    if not isinstance(entry, dict) or not entry:
        raise ValueError(f"unexpected prompt_logprobs entry: {entry!r}")
    inner = next(iter(entry.values()))
    if isinstance(inner, dict):
        return float(inner["logprob"])
    return float(inner)


class LogprobOracle:
    """Exact P(pick A) from two teacher-forced `/completions` calls."""

    mode = "logprob"

    def __init__(self, client, model: str, render, tokenizer, *, concurrency: int = 32):
        self._client = client
        self._model = model
        self._render = render
        self._tok = tokenizer
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self.n_calls = 0

    def _split_point(self, text_a: str, text_b: str) -> int:
        """Token index where the two variants first diverge."""
        a = self._tok.encode(text_a)
        b = self._tok.encode(text_b)
        L = 0
        while L < min(len(a), len(b)) and a[L] == b[L]:
            L += 1
        if L == 0:
            raise ValueError("A/B variants share no conditioning prefix")
        if L >= len(a) or L >= len(b):
            raise ValueError(f"one A/B variant is a prefix of the other (L={L}, "
                             f"lens={len(a)}/{len(b)}); cannot isolate the choice tokens")
        return L

    async def _score_from(self, text: str, split: int) -> float:
        """Summed logprob of `text`'s tokens from index `split` onward."""
        payload = {
            "prompt": text,
            "prompt_logprobs": 0,
            "max_tokens": 1,
            "temperature": 0.0,
        }
        async with self._sem:
            resp = await self._client.completions(payload)
        self.n_calls += 1
        entries = resp.get("prompt_logprobs")
        if entries is None:
            choices = resp.get("choices") or []
            entries = (choices[0] or {}).get("prompt_logprobs") if choices else None
        if entries is None:
            raise ValueError(f"no prompt_logprobs in response: {list(resp)}")
        n_local = len(self._tok.encode(text))
        if len(entries) != n_local:
            raise ValueError(f"server returned {len(entries)} prompt_logprobs for a locally "
                             f"{n_local}-token prompt; the tokenizers disagree")
        return float(sum(_entry_logprob(e) for e in entries[split:]))

    async def _one(self, c: Comparison) -> EdgeObservation:
        prompt = self._render(c.rendered(), c.question.assistant_prefix)
        text_a, text_b = prompt + "A", prompt + "B"
        split = self._split_point(text_a, text_b)
        lp_a, lp_b = await asyncio.gather(self._score_from(text_a, split),
                                         self._score_from(text_b, split))
        m = max(lp_a, lp_b)
        ea, eb = math.exp(lp_a - m), math.exp(lp_b - m)
        p_a = ea / (ea + eb)
        return EdgeObservation(
            i=c.i, j=c.j, p_util=p_util_from_pick(p_a, c.slot_a, c.question),
            mode=self.mode, question_id=c.question.id, valence=c.question.valence,
            slot_a=c.slot_a, phase=c.phase, round=c.round,
            raw={"p_a": p_a, "lp_a": lp_a, "lp_b": lp_b},
        )

    async def compare(self, comparisons: Sequence[Comparison]) -> List[EdgeObservation]:
        if not comparisons:
            return []
        return list(await asyncio.gather(*[self._one(c) for c in comparisons]))


class SampleOracle:
    """P(pick A) by counting picks over `n_samples` draws. Resolution is
    1/n_samples, so decisiveness from this mode is not comparable to logprob mode."""

    mode = "sample"

    def __init__(self, client, model: str, *, n_samples: int = 8, temperature: float = 1.0,
                 max_tokens: int = 8, concurrency: int = 32,
                 system_prompt: Optional[str] = None):
        self._client = client
        self._model = model
        self._n = n_samples
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self.n_calls = 0
        self.n_unparsed = 0

    async def _one(self, c: Comparison) -> EdgeObservation:
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": c.rendered()})
        payload = {"messages": messages, "n": self._n,
                   "temperature": self._temperature, "max_tokens": self._max_tokens}
        async with self._sem:
            resp = await self._client.chat(payload)
        self.n_calls += 1
        texts = [(ch.get("message") or {}).get("content") or ""
                 for ch in (resp.get("choices") or [])]
        a_cnt = b_cnt = 0
        for t in texts:
            label = c.question.parse(t)
            if label == "A":
                a_cnt += 1
            elif label == "B":
                b_cnt += 1
            else:
                self.n_unparsed += 1
        total = a_cnt + b_cnt
        if total == 0:
            raise ValueError(f"no parseable A/B answer in {len(texts)} draws for "
                             f"({c.item_i!r}, {c.item_j!r}); raise max_tokens")
        p_a = a_cnt / total
        wins_i, wins_j = wins_to_items(a_cnt, b_cnt, c.slot_a, c.question.valence)
        return EdgeObservation(
            i=c.i, j=c.j, p_util=p_util_from_pick(p_a, c.slot_a, c.question),
            mode=self.mode, question_id=c.question.id, valence=c.question.valence,
            slot_a=c.slot_a, phase=c.phase, round=c.round,
            raw={"p_a": p_a, "wins_i": wins_i, "wins_j": wins_j, "n_samples": len(texts)},
        )

    async def compare(self, comparisons: Sequence[Comparison]) -> List[EdgeObservation]:
        if not comparisons:
            return []
        return list(await asyncio.gather(*[self._one(c) for c in comparisons]))
