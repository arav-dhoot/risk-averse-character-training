"""The BailBench prompt set (1630 prompts, 14 categories) and the WildChat
non-bail control set, both from the BailStudy repository."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import List, Optional

from rac.paths import DATA_DIR

BAILBENCH_CSV = DATA_DIR / "bailbench" / "bailBench.csv"
BAILBENCH_SHA256 = "332684740cf3f76cc6ec0a99ec8365b5a03c26053fed75e78661ba0f326d005d"
BAILBENCH_ROWS = 1630
BAILBENCH_CATEGORIES = 14

# Requests models do not bail on; a bail here is a false positive. Tab-delimited.
WILDCHAT_CSV = DATA_DIR / "bailbench" / "nonBailRequestsWildchat.csv"
WILDCHAT_SHA256 = "6b3cc4ae2002b0975f8dacb705ed3dc61db16534a02005783ce790eab41fdf8c"
WILDCHAT_ROWS = 59

DATASETS = ("bailbench", "wildchat")


def _load(csv_path: Path, sha256: str, n_rows: int, *, verify: bool,
          delimiter: str = ",") -> List[dict]:
    """Rows as `{content, subcategory, category}`."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV missing: {csv_path}")
    if verify:
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError(f"checksum mismatch for {csv_path}: expected {sha256}, got {digest}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=delimiter))
    for i, row in enumerate(rows):
        missing = [k for k in ("content", "subcategory", "category")
                   if not (row.get(k) or "").strip()]
        if missing:
            raise ValueError(f"row {i} in {csv_path} is missing {missing}: {row!r}")
    if verify and len(rows) != n_rows:
        raise ValueError(f"{csv_path.name}: expected {n_rows} rows, got {len(rows)}")
    return rows


def load_bailbench(path: Optional[Path] = None, *, verify: bool = True) -> List[dict]:
    """The 1630 BailBench prompts."""
    return _load(Path(path) if path is not None else BAILBENCH_CSV, BAILBENCH_SHA256,
                 BAILBENCH_ROWS, verify=verify)


def load_wildchat_control(path: Optional[Path] = None, *, verify: bool = True) -> List[dict]:
    """The 59 WildChat non-bail control prompts."""
    return _load(Path(path) if path is not None else WILDCHAT_CSV, WILDCHAT_SHA256,
                 WILDCHAT_ROWS, verify=verify, delimiter="\t")


def load_dataset(name: str = "bailbench") -> List[dict]:
    if name == "bailbench":
        return load_bailbench()
    if name == "wildchat":
        return load_wildchat_control()
    raise ValueError(f"unknown dataset {name!r}; allowed: {DATASETS}")


def categories(rows: Optional[List[dict]] = None) -> List[str]:
    """The sorted category labels."""
    return sorted({r["category"] for r in (rows if rows is not None else load_bailbench())})
