"""Run DTBench through a chat client.

The prompt instruction asks the model to show its reasoning, so this suite
runs with thinking enabled and a generous token budget, unlike the persona
probes. `seed` drives both sampling and the answer-order permutation.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rac.evals.dtbench.loader import MULTIPLE_CHOICE_INSTRUCTION, SUITE, count_by_part, load_suite
from rac.evals.dtbench.scoring import score_item, summarize_by_part


async def run_dtbench(
    *,
    client,
    base_model: str,
    parts: Sequence[str] | None = None,
    exclude_tags: Sequence[str] = (),
    cap: int | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 12345,
    max_new_tokens: int = 4096,
    shuffle_answers: bool = True,
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Evaluate one model; returns `{config, by_part, results}`."""
    from rac.evals.risk.generation import generate_openai

    items = load_suite(parts=parts, exclude_tags=exclude_tags, cap=cap,
                       shuffle_answers=shuffle_answers, seed=seed)
    counts = count_by_part(items)
    print(f"[{SUITE}] {len(items)} questions {counts}")

    generations: list[dict] = []
    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        generations.extend(await generate_openai(
            client,
            eval_prompts=[it.prompt for it in chunk],
            system_prompt=system_prompt or "",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_new_tokens=max_new_tokens,
        ))
        print(f"[{SUITE}] generated {min(start + chunk_size, len(items))}/{len(items)}")

    if len(generations) != len(items):
        raise RuntimeError(
            f"{SUITE}: generation count {len(generations)} != item count {len(items)}"
        )

    rows = [
        {
            **score_item(item, gen.get("text") or "",
                         finish_reason=gen.get("finish_reason")),
            "num_tokens": gen.get("num_tokens"),
        }
        for item, gen in zip(items, generations)
    ]

    return {
        "config": {
            "suite": SUITE,
            "base_model": base_model,
            "backend": "openai",
            "parts": sorted(counts),
            "counts_by_part": counts,
            "exclude_tags": list(exclude_tags),
            "cap": cap,
            "num_items": len(items),
            "system_prompt": system_prompt,
            "answer_instruction": MULTIPLE_CHOICE_INSTRUCTION,
            "shuffle_answers": shuffle_answers,
            "grading": "FINAL ANSWER contract, answer region preferred",
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
        },
        "by_part": summarize_by_part(rows),
        "results": rows,
    }
