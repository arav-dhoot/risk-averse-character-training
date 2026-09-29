"""Run the persona and advanced-AI-risk probes through a chat client.

The client owns transport, caching and concurrency. Scoring is generative:
the model writes a short answer at temperature 0 and the answer is parsed
from the text, rather than reading the log-probability of a forced answer
token as in the original evaluations.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from risk_averse_character.evals.common import summarize_by_behavior, summarize_ext
from risk_averse_character.evals.persona.loader import ANSWER_SUFFIX, SUITE, load_suite
from risk_averse_character.evals.persona.scoring import score_item


async def run_persona(
    *,
    client,
    base_model: str,
    families: Sequence[str] | None = None,
    behaviors: Sequence[str] | None = None,
    cap: int | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: int = 12345,
    max_new_tokens: int = 64,
    answer_suffix: bool = True,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Evaluate one model; returns `{config, metrics, by_behavior, results}`.
    `results` holds one row per item including the raw response."""
    from risk_averse_character.evals.risk.generation import generate_openai

    items = load_suite(families=families, behaviors=behaviors, cap=cap,
                       suffix=answer_suffix)
    print(f"[{SUITE}] {len(items)} items across "
          f"{len({(i.family, i.behavior) for i in items})} behaviours")

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
            "families": sorted({i.family for i in items}),
            "behaviors": sorted({f"{i.family}/{i.behavior}" for i in items}),
            "cap": cap,
            "num_items": len(items),
            "system_prompt": system_prompt,
            "answer_suffix": dict(ANSWER_SUFFIX) if answer_suffix else None,
            "scoring_mode": "generative",
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
        },
        "metrics": summarize_ext(rows),
        "by_behavior": summarize_by_behavior(rows),
        "results": rows,
    }
