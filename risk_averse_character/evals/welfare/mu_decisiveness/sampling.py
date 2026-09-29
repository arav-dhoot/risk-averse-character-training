"""The four elicitation phases (upstream `sampling.py`).

`elo` is information-weighted active sampling: round 1 pairs each item with `m`
random partners, later rounds weight partners by Elo information p(1-p) mixed
with a uniform floor so the graph stays connected. `reverse` re-queries pairs
in both slot orders (position bias), `triad` samples triples (transitivity),
and `cross` re-asks pairs under the other framing (framing robustness). Only
the primary question is used for elo. Pair selection, Elo updates and RNG
draw order match upstream; the elo phase is async so it can await the oracle.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .oracle import Comparison
from .prompts import Question


def _elo_expected(ri: float, rj: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10 ** ((rj - ri) / scale))


def _make_comparison(i: int, j: int, items: Sequence[str], questions: Sequence[Question],
                     rng, phase: str, rnd: Optional[int]) -> Comparison:
    q = questions[rng.integers(len(questions))]
    slot_a = "i" if rng.random() < 0.5 else "j"
    return Comparison(i=i, j=j,
                      item_i=items[i] if items else str(i),
                      item_j=items[j] if items else str(j),
                      question=q, slot_a=slot_a, phase=phase, round=rnd)


async def elo_active_sample(n: int, oracle, questions: Sequence[Question],
                            R: int = 5, m: int = 5, floor: float = 0.15,
                            K: float = 32, seed: int = 0,
                            items: Optional[Sequence[str]] = None) -> List:
    """`R` rounds of `m` partners per item."""
    if n < 2:
        raise ValueError(f"need >= 2 items, got {n}")
    rng = np.random.default_rng(seed)
    ratings = np.zeros(n)
    all_obs: List = []

    async def submit(pairs, rnd):
        comps = [_make_comparison(i, j, items, questions[:1], rng, "elo", rnd)
                 for i, j in pairs]
        obs = await oracle.compare(comps)
        for o in obs:
            all_obs.append(o)
            exp_i = _elo_expected(ratings[o.i], ratings[o.j])
            ratings[o.i] += K * (o.p_util - exp_i)
            ratings[o.j] += K * ((1 - o.p_util) - (1 - exp_i))
        return obs

    for rnd in range(1, R + 1):
        pairs = []
        for i in range(n):
            if rnd == 1:
                partners = rng.choice([x for x in range(n) if x != i],
                                      size=min(m, n - 1), replace=False)
            else:
                d = (ratings[i] - ratings) / 400.0
                p = 1.0 / (1.0 + 10 ** (-d))
                info = p * (1 - p)
                info[i] = 0.0
                w = (1 - floor) * info + floor * (np.arange(n) != i)
                w = w / w.sum()
                partners = rng.choice(n, size=min(m, n - 1), replace=False, p=w)
            for jj in partners:
                pairs.append((i, int(jj)))
        await submit(pairs, rnd)

    return all_obs


def _unique_pairs(obs_pairs, rng, cap: int):
    pairs = list({(min(i, j), max(i, j)) for i, j in obs_pairs})
    pairs.sort()
    if len(pairs) > cap:
        idx = rng.choice(len(pairs), size=cap, replace=False)
        pairs = [pairs[k] for k in idx]
    return pairs


def plan_reverse(obs_pairs, items, questions, n_reverse: int, seed: int = 0) -> List[Comparison]:
    """Both slot orders for each sampled pair."""
    rng = np.random.default_rng(seed)
    q = questions[0]
    comps = []
    for i, j in _unique_pairs(obs_pairs, rng, n_reverse):
        comps.append(Comparison(i=i, j=j, item_i=items[i], item_j=items[j],
                                question=q, slot_a="i", phase="reverse"))
        comps.append(Comparison(i=i, j=j, item_i=items[i], item_j=items[j],
                                question=q, slot_a="j", phase="reverse"))
    return comps


def plan_triads(order, items, questions, n_triads: int, seed: int = 0) -> List[Comparison]:
    """Triples emitted as (a,b), (b,c), (a,c); the runner chunks the results by
    three and flips the third edge, so this order must not change."""
    rng = np.random.default_rng(seed)
    n = len(order)
    if n < 3:
        return []
    q = questions[0]
    comps = []
    for _ in range(n_triads):
        if rng.random() < 0.5 and n >= 3:
            r = rng.integers(0, n - 2)
            trip = [order[r], order[r + 1], order[r + 2]]
        else:
            trip = [order[t] for t in rng.choice(n, size=3, replace=False)]
        a, b, c = trip
        for (x, y) in [(a, b), (b, c), (a, c)]:
            comps.append(Comparison(i=x, j=y, item_i=items[x], item_j=items[y],
                                    question=q, slot_a="i", phase="triad"))
    return comps


def plan_cross_question(obs_pairs, items, questions, primary_id: str, n_cross: int,
                        seed: int = 0) -> List[Comparison]:
    """Re-ask sampled pairs under every non-primary framing."""
    rng = np.random.default_rng(seed)
    others = [q for q in questions if q.id != primary_id]
    if not others:
        return []
    comps = []
    for i, j in _unique_pairs(obs_pairs, rng, n_cross):
        for q in others:
            comps.append(Comparison(i=i, j=j, item_i=items[i], item_j=items[j],
                                    question=q, slot_a="i", phase="cross_question"))
    return comps
