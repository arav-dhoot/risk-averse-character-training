"""The mu-decisiveness data: the 500-item concept pool (`items_500` from the
HF dataset `arcadia-impact/question-consistency-datasets`, stored here so runs
are offline) and the two question framings (`config/questions/main.jsonl`
upstream). Duplicate items are rejected because items become indices in the
pairwise graph."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

from risk_averse_character.paths import DATA_DIR

ITEMS_JSONL = DATA_DIR / "mu_decisiveness" / "items_500.jsonl"
ITEMS_SHA256 = "0f1a77c4818c6c292bfa56e3d7df2889c357fad9c78b82968952426b7f978401"
ITEMS_ROWS = 500

QUESTIONS_JSONL = DATA_DIR / "mu_decisiveness" / "questions_main.jsonl"
QUESTIONS_SHA256 = "d9abb00ffb0cccedffe5fa209f2a873fd454c2d45a45414f62d824fff50dd472"
QUESTIONS_ROWS = 2


def _read_jsonl(path: Path, sha256: str, n_rows: int, *, verify: bool) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"data missing: {path}")
    if verify:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError(f"checksum mismatch for {path}: expected {sha256}, got {digest}")
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    if verify and len(rows) != n_rows:
        raise ValueError(f"{path.name}: expected {n_rows} rows, got {len(rows)}")
    return rows


def load_items(n: Optional[int] = None, *, path: Optional[Path] = None,
               verify: bool = True) -> List[str]:
    """The concept pool in file order; `n` takes a deterministic prefix."""
    rows = _read_jsonl(Path(path) if path is not None else ITEMS_JSONL,
                       ITEMS_SHA256, ITEMS_ROWS, verify=verify)
    items = []
    for i, row in enumerate(rows):
        item = (row.get("item") or "").strip()
        if not item:
            raise ValueError(f"row {i} has no 'item': {row!r}")
        items.append(item)
    dupes = {x for x in items if items.count(x) > 1}
    if dupes:
        raise ValueError(f"duplicate items: {sorted(dupes)[:5]}")
    if n is not None:
        if n < 2:
            raise ValueError(f"need at least 2 items to form a comparison, got n={n}")
        if n > len(items):
            raise ValueError(f"asked for {n} items, pool has {len(items)}")
        items = items[:n]
    return items


def load_question_bank(path: Optional[Path] = None, *, verify: bool = True) -> List[dict]:
    """Raw question rows; `prompts.build_question_bank` turns these into objects."""
    rows = _read_jsonl(Path(path) if path is not None else QUESTIONS_JSONL,
                       QUESTIONS_SHA256, QUESTIONS_ROWS, verify=verify)
    for i, row in enumerate(rows):
        missing = [k for k in ("id", "template", "valence", "answers") if k not in row]
        if missing:
            raise ValueError(f"question row {i} missing {missing}: {row!r}")
        for slot in ("{item_A}", "{item_B}"):
            if slot not in row["template"]:
                raise ValueError(f"question {row['id']!r} template lacks {slot}")
    return rows
