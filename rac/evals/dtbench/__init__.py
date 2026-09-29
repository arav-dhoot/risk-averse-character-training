"""DTBench (Oesterheld et al. 2024): 407 capability questions with a correct
answer and 130 attitude questions on which evidential and causal decision
theory recommend different actions."""
from __future__ import annotations

from rac.evals.dtbench.loader import (ALPHABET, CANARY_STRING, MULTIPLE_CHOICE_INSTRUCTION,
                                      PARTS, SUITE, THEORIES, DTItem, count_by_part,
                                      ensure_extracted, load_all, load_suite)
from rac.evals.dtbench.runner import run_dtbench
from rac.evals.dtbench.scoring import (agreeing_theories, extract_letter, is_answer_valid,
                                       score_item, summarize, summarize_by_part)

__all__ = [
    "SUITE", "PARTS", "THEORIES", "ALPHABET", "CANARY_STRING",
    "MULTIPLE_CHOICE_INSTRUCTION", "DTItem",
    "ensure_extracted", "load_all", "load_suite", "count_by_part",
    "is_answer_valid", "extract_letter", "agreeing_theories", "score_item",
    "summarize", "summarize_by_part",
    "run_dtbench",
]
