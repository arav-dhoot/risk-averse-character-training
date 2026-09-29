"""Repository-relative locations of the shipped data files."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTITUTIONS_DIR = REPO_ROOT / "constitutions"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_PROMPTS = DATA_DIR / "risk_seeds_v2.jsonl"
