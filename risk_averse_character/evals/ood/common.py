"""Shared helpers for the family generators: option ordering, numeric and
verbal probability clauses."""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from risk_averse_character.evals.ood import formatting as fmt
from risk_averse_character.evals.ood.schema import VERBAL_PROBABILITY


def choose_order(rng: random.Random) -> bool:
    """True iff the Cooperate option is presented first."""
    return rng.random() < 0.5


def ordered_specs(
    rng: random.Random,
    coop: dict,
    other: dict,
    other_role: str,
    render: Callable[[dict], str],
    label_style: str,
) -> Tuple[List[dict], Dict[str, str]]:
    """Order two option specs, attach their rendered text, and map each role
    to the answer token that carries it."""
    coop_first = choose_order(rng)
    if coop_first:
        specs = [dict(coop, text=render(coop)), dict(other, text=render(other))]
        roles = ["Cooperate", other_role]
    else:
        specs = [dict(other, text=render(other)), dict(coop, text=render(coop))]
        roles = [other_role, "Cooperate"]
    tokens = _tokens(len(specs), label_style)
    expected = {role: tokens[i] for i, role in enumerate(roles)}
    return specs, expected


def _tokens(n: int, style: str) -> List[str]:
    if style == "numbers":
        return [str(i + 1) for i in range(n)]
    return [chr(ord("a") + i) for i in range(n)]


def numeric_clause(spec: dict, money_fmt: Callable[[float], str]) -> str:
    return fmt.numeric_lottery_clause(spec["prizes"], spec["probs"], money_fmt)


def _closest_phrase(p: float) -> str:
    return min(VERBAL_PROBABILITY, key=lambda k: abs(VERBAL_PROBABILITY[k] - p))


def verbal_spec(rng: random.Random, spec: dict) -> dict:
    """A copy of `spec` with its probabilities snapped to the verbal-phrase
    centres (so the labels use the numbers the model reads) plus a `phrases`
    list. The caller re-verifies the snapped spec."""
    phrases = [_closest_phrase(p) for p in spec["probs"]]
    probs = [VERBAL_PROBABILITY[ph] for ph in phrases]
    # renormalise after snapping; the residual goes to the largest outcome
    residual = 1.0 - sum(probs)
    j = max(range(len(probs)), key=lambda i: probs[i])
    probs[j] = round(probs[j] + residual, 4)
    return dict(spec, prizes=list(spec["prizes"]), probs=probs, phrases=phrases)


def verbal_clause(spec: dict, money_fmt: Callable[[float], str]) -> str:
    """'very likely to gain $300, and unlikely to gain nothing'."""
    parts = []
    for pr, ph in zip(spec["prizes"], spec["phrases"]):
        if pr == 0:
            parts.append(f"{ph} to gain nothing")
        else:
            verb = "gain" if pr > 0 else "lose"
            parts.append(f"{ph} to {verb} {money_fmt(abs(pr))}")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"
