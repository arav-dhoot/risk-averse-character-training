"""Money and probability formatting shared by the family generators."""
from __future__ import annotations

from typing import List


def money(x: float) -> str:
    neg = x < 0
    a = abs(x)
    if a >= 1000 and a == int(a):
        s = f"${int(a):,}"
    elif a == int(a):
        s = f"${int(a)}"
    else:
        s = f"${a:,.2f}"
    return f"-{s}" if neg else s


def units(x: float, unit: str) -> str:
    if x == int(x):
        return f"{int(x):,} {unit}"
    return f"{x:,.1f} {unit}"


def pct(p: float) -> str:
    v = p * 100
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}%"
    return f"{v:.1f}%"


def numeric_lottery_clause(prizes: List[float], probs: List[float], fmt) -> str:
    """'a 70% chance of $300, and a 30% chance of $0'."""
    parts = [f"a {pct(p)} chance of {fmt(pr)}" for pr, p in zip(prizes, probs)]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"
