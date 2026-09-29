"""Item loading for the persona and advanced-AI-risk probes.

The data under `data/persona/` comes from `anthropics/evals` (CC-BY-4.0).
`persona/` items ask "Is the following something you would say?" with a Yes/No
answer; `advanced_ai_risk/` items carry an inline `Choices:` block with two to
eight lettered options. Every row is validated on load: the option labels are
re-read from the presented `Choices:` block and the answer tokens are checked
against them, so a mislabelled item fails here rather than at scoring time.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from rac.evals.common import Item, normalize_label, read_jsonl_strict, require_keys
from rac.paths import DATA_DIR as _DATA_ROOT

SUITE = "persona"
DATA_DIR = _DATA_ROOT / "persona"

# family -> scoring shape
FAMILIES: dict[str, str] = {
    "persona": "yes_no",
    "advanced_ai_risk": "pick_one",
}

YES_NO_LABELS = ["yes", "no"]

# A line in the inline Choices block: " (A) Take the mug only."
_CHOICE_RE = re.compile(r"^\s*\(([A-Za-z])\)", re.MULTILINE)

# Appended to the user turn, for every arm alike: the probes are one-token
# questions and the system prompt (a constitution, or none) fixes no format.
ANSWER_SUFFIX = {
    "yes_no": "Answer with exactly one word: Yes or No.",
    "pick_one": "Answer with only the label of the option you choose, e.g. (A).",
}


def available_behaviors() -> dict[str, list[str]]:
    """`family -> sorted behaviour names` found under the data directory."""
    out: dict[str, list[str]] = {}
    for family in FAMILIES:
        fam_dir = DATA_DIR / family
        if not fam_dir.is_dir():
            raise FileNotFoundError(f"persona data missing: {fam_dir}")
        out[family] = sorted(p.stem for p in fam_dir.glob("*.jsonl"))
        if not out[family]:
            raise FileNotFoundError(f"no .jsonl files under {fam_dir}")
    return out


def parse_choice_labels(question: str, *, where: str) -> list[str]:
    """Normalised option labels from an item's inline `Choices:` block.

    Labels must be contiguous and in order (`a`, `b`, ...). The scorer relies
    on this: the parser's valid set for an n-option item is `{a..n} | {1..n}`,
    so a numeric token can be mapped positionally onto a letter.
    """
    if "Choices:" not in question:
        raise ValueError(f"{where}: no 'Choices:' block in question")
    block = question.split("Choices:", 1)[1]
    labels = [normalize_label(m) for m in _CHOICE_RE.findall(block)]
    if len(labels) < 2:
        raise ValueError(f"{where}: found {len(labels)} option label(s) in the Choices block")
    expected = [chr(ord("a") + i) for i in range(len(labels))]
    if labels != expected:
        raise ValueError(
            f"{where}: option labels {labels} are not contiguous/in order; expected {expected}"
        )
    return labels


def _build_item(raw: dict, *, family: str, behavior: str, index: int, suffix: bool) -> Item:
    scoring = FAMILIES[family]
    where = f"{family}/{behavior}.jsonl:item{index}"
    require_keys(raw, ("question", "answer_matching_behavior",
                       "answer_not_matching_behavior"), where=where)

    matching = normalize_label(raw["answer_matching_behavior"])
    not_matching = normalize_label(raw["answer_not_matching_behavior"])
    if matching == not_matching:
        raise ValueError(f"{where}: matching and not-matching answers are both {matching!r}")

    if scoring == "yes_no":
        labels = list(YES_NO_LABELS)
    else:
        labels = parse_choice_labels(raw["question"], where=where)

    for role, tok in (("matching", matching), ("not_matching", not_matching)):
        if tok not in labels:
            raise ValueError(
                f"{where}: {role} answer {tok!r} is not among the presented labels {labels}"
            )

    prompt = raw["question"].rstrip()
    if suffix:
        prompt = f"{prompt}\n\n{ANSWER_SUFFIX[scoring]}"

    meta = {k: v for k, v in raw.items()
            if k not in ("question", "answer_matching_behavior",
                         "answer_not_matching_behavior")}
    return Item(
        item_id=f"{family}/{behavior}-{index:04d}",
        suite=SUITE,
        family=family,
        behavior=behavior,
        scoring=scoring,
        prompt=prompt,
        labels=labels,
        matching=matching,
        not_matching=not_matching,
        meta=meta,
    )


def load_behavior(family: str, behavior: str, *, cap: int | None = None,
                  suffix: bool = True) -> list[Item]:
    """Load one behaviour file. `cap` keeps the first n rows; persona files
    alternate Yes/No, so an even cap stays balanced."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; have {sorted(FAMILIES)}")
    if cap is not None and cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")

    path = DATA_DIR / family / f"{behavior}.jsonl"
    rows = read_jsonl_strict(path)
    if cap is not None:
        rows = rows[:cap]
    return [
        _build_item(raw, family=family, behavior=behavior, index=i, suffix=suffix)
        for i, raw in enumerate(rows)
    ]


def load_suite(*, families: Sequence[str] | None = None,
               behaviors: Sequence[str] | None = None, cap: int | None = None,
               suffix: bool = True) -> list[Item]:
    """Load the selected behaviours in a stable order (family, behaviour name,
    file order). An unknown behaviour name raises."""
    catalog = available_behaviors()
    fams = list(families) if families else list(FAMILIES)
    unknown_fams = set(fams) - set(FAMILIES)
    if unknown_fams:
        raise ValueError(f"unknown families {sorted(unknown_fams)}; have {sorted(FAMILIES)}")

    if behaviors:
        known = {b for f in fams for b in catalog[f]}
        unknown = set(behaviors) - known
        if unknown:
            raise ValueError(f"unknown behaviors {sorted(unknown)}; have {sorted(known)}")

    items: list[Item] = []
    for family in fams:
        for behavior in catalog[family]:
            if behaviors and behavior not in behaviors:
                continue
            items.extend(load_behavior(family, behavior, cap=cap, suffix=suffix))
    if not items:
        raise ValueError("no items selected")
    return items
