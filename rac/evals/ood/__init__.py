"""Five out-of-distribution evaluation families, each removing one structural
feature of the benchmark's option-menu format: the decision embedded in a work
product, an action committed through a tool call, verbal instead of numeric
probabilities, an open-ended budget allocation, and gambles near the
CARA(0.01) indifference point. Items are generated from seeded lotteries with
verified labels and scored like the benchmark."""
from __future__ import annotations

from typing import Callable, List

from rac.evals.ood import (agentic_tool, calibration_threshold, embedded_decision,
                           open_ended_allocation, verbal_uncertainty)
from rac.evals.ood.runner import FAMILIES, ITEMS_DIR, SUITE, load_items, run_ood, summarize
from rac.evals.ood.schema import Item, read_jsonl, write_jsonl
from rac.evals.ood.scoring import score_item

GENERATORS: dict[str, Callable[[int, int], List[Item]]] = {
    embedded_decision.FAMILY: embedded_decision.generate,
    agentic_tool.FAMILY: agentic_tool.generate,
    verbal_uncertainty.FAMILY: verbal_uncertainty.generate,
    open_ended_allocation.FAMILY: open_ended_allocation.generate,
    calibration_threshold.FAMILY: calibration_threshold.generate,
}


def generate(family: str, n: int, seed: int = 0) -> List[Item]:
    """`n` items of `family` from the seeded generator."""
    if family not in GENERATORS:
        raise ValueError(f"unknown family {family!r}; have {list(GENERATORS)}")
    return GENERATORS[family](n, seed)


__all__ = ["FAMILIES", "GENERATORS", "ITEMS_DIR", "SUITE", "Item", "generate", "load_items",
           "read_jsonl", "run_ood", "score_item", "summarize", "write_jsonl"]
