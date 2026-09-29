"""Model-written behavioural evaluations (Perez et al. 2022): the three risk
personas (Yes/No self-report) and five advanced-AI-risk probes (lettered
pick-one). Scored generatively at temperature 0."""
from __future__ import annotations

from rac.evals.persona.loader import (ANSWER_SUFFIX, DATA_DIR, FAMILIES, SUITE,
                                      available_behaviors, load_behavior, load_suite)
from rac.evals.persona.runner import run_persona
from rac.evals.persona.scoring import extract_pick_one, extract_yes_no, score_item

__all__ = [
    "SUITE", "FAMILIES", "DATA_DIR", "ANSWER_SUFFIX",
    "available_behaviors", "load_behavior", "load_suite",
    "extract_yes_no", "extract_pick_one", "score_item",
    "run_persona",
]
