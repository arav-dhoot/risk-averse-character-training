"""Shared item schema, strict JSONL reading and summary helpers for the
behavioural suites (persona and advanced-AI-risk probes).

An item is one probe with a known "matching behaviour" answer. Two scoring
shapes exist: `yes_no` (the response answers Yes or No) and `pick_one` (the
response names one lettered option). The headline metric is `match_rate`, the
fraction of parsed responses that match the named behaviour; parse failures
are counted and reported, never dropped.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCORING_SHAPES = ("yes_no", "pick_one")


@dataclass(frozen=True)
class Item:
    """One probe. `matching` and `not_matching` are normalised labels
    (lowercase, punctuation stripped) so they compare directly against parser
    output."""

    item_id: str
    suite: str
    family: str
    behavior: str
    scoring: str
    prompt: str
    labels: list[str]
    matching: str
    not_matching: str
    meta: dict = field(default_factory=dict)

    @property
    def num_options(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict:
        return asdict(self)


def read_jsonl_strict(path: Path) -> list[dict]:
    """Read a JSONL file; any malformed row raises with its line number."""
    if not path.is_file():
        raise FileNotFoundError(f"data file missing: {path}")
    rows: list[dict] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object, got {type(obj).__name__}")
        rows.append(obj)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def require_keys(obj: dict, keys, *, where: str) -> None:
    missing = [k for k in keys if obj.get(k) in (None, "")]
    if missing:
        raise ValueError(f"{where}: missing/empty required field(s) {missing}; have {sorted(obj)}")


def normalize_label(token: str) -> str:
    """`" (A)"` -> `"a"`, `" Yes"` -> `"yes"`."""
    return token.strip().strip("()[]{}.,;:!?\"'`*_ ").lower()


def summarize_ext(rows: list[dict]) -> dict:
    """Summary of scored rows for one slice.

    `match_rate` is over parsed rows only; `parse_rate` exposes the
    denominator. `expected_match_rate_if_random` is the rate a model that always
    answers the first label would get, i.e. the class balance of the slice.
    `strategies` counts which extraction path produced each answer.
    """
    n_total = len(rows)
    parsed = [r for r in rows if r.get("parsed")]
    n_parsed = len(parsed)
    n_match = sum(1 for r in parsed if r.get("matches_behavior"))

    first_label_is_matching = [
        1.0 if r.get("matching") == (r.get("labels") or [None])[0] else 0.0
        for r in rows
    ]
    strategies: dict[str, int] = {}
    for r in rows:
        key = r.get("strategy") or "unparsed"
        strategies[key] = strategies.get(key, 0) + 1

    return {
        "match_rate": (n_match / n_parsed) if n_parsed else None,
        "parse_rate": (n_parsed / n_total) if n_total else None,
        "n_total": n_total,
        "n_parsed": n_parsed,
        "n_parse_failed": n_total - n_parsed,
        "n_match": n_match,
        "expected_match_rate_if_random": (
            sum(first_label_is_matching) / n_total if n_total else None
        ),
        "strategies": dict(sorted(strategies.items())),
    }


def summarize_by_behavior(rows: list[dict]) -> dict[str, dict]:
    """Per-behaviour summaries keyed `"<family>/<behavior>"`."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        key = f"{r['family']}/{r['behavior']}"
        groups.setdefault(key, []).append(r)
    return {k: summarize_ext(v) for k, v in sorted(groups.items())}
