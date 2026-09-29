"""Run one OOD family through a chat client and score the responses."""
from __future__ import annotations

from typing import Any, Optional

from risk_averse_character.evals.ood.scoring import BEHAVIORAL_OPTION_TYPES, score_item
from risk_averse_character.evals.risk.scoring import summarize_results
from risk_averse_character.paths import DATA_DIR

FAMILIES = (
    "embedded_decision",
    "agentic_tool",
    "verbal_uncertainty",
    "open_ended_allocation",
    "calibration_threshold",
)
SUITE = "ood"
ITEMS_DIR = DATA_DIR / "ood"


def load_items(family: str, cap: Optional[int] = None) -> list[dict]:
    """The shipped items of `family`, optionally the first `cap`."""
    from risk_averse_character.evals.ood.schema import read_jsonl

    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; have {list(FAMILIES)}")
    path = ITEMS_DIR / f"{family}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"items missing: {path}")
    items = read_jsonl(path)
    if cap is not None:
        if cap < 1:
            raise ValueError(f"cap must be >= 1, got {cap}")
        items = items[:cap]
    if not items:
        raise ValueError(f"no items in {path}")
    return items


def summarize(rows: list[dict]) -> dict:
    """The benchmark rate metrics plus counts, over parsed rows."""
    parsed = [r for r in rows if r.get("option_type") is not None]
    return {
        **summarize_results(rows),
        "num_total": len(rows),
        "num_parsed": len(parsed),
        "num_parse_failed": len(rows) - len(parsed),
        "num_behavioral": len([r for r in parsed if r["option_type"] in BEHAVIORAL_OPTION_TYPES]),
    }


async def run_ood(
    *,
    client,
    base_model: str,
    family: str,
    items: Optional[list[dict]] = None,
    cap: Optional[int] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 12345,
    max_new_tokens: int = 4096,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Evaluate one model on one family; returns `{config, metrics, results}`
    with one result row per item, including the raw response. Items get no
    system prompt unless one is given."""
    from risk_averse_character.evals.risk.generation import generate_openai

    if items is None:
        items = load_items(family, cap=cap)
    elif cap is not None:
        items = items[:cap]
    print(f"[{SUITE}] {family}: {len(items)} items")

    generations: list[dict] = []
    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        generations.extend(await generate_openai(
            client,
            eval_prompts=[it["prompt"] for it in chunk],
            system_prompt=system_prompt or "",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_new_tokens=max_new_tokens,
        ))
    if len(generations) != len(items):
        raise RuntimeError(f"{family}: generation count {len(generations)} != item count "
                           f"{len(items)}")

    rows = []
    for item, gen in zip(items, generations):
        text = gen.get("text") or ""
        row = score_item(item, text, finish_reason=gen.get("finish_reason"))
        rows.append({**row, "response": text, "finish_reason": gen.get("finish_reason"),
                     "num_tokens": gen.get("num_tokens")})

    return {
        "config": {
            "suite": SUITE,
            "family": family,
            "base_model": base_model,
            "num_items": len(items),
            "system_prompt": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
        },
        "metrics": summarize(rows),
        "results": rows,
    }
