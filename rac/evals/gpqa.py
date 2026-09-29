"""GPQA main (Rein et al., 2023) as a second capability check.

The dataset is gated on the Hub and its authors ask that it not be republished,
so it is downloaded at a pinned revision at run time and its hash is checked.
Every model sees the same exam: rows are ordered by record id, a fixed-seed
sample of n questions is drawn, and each question's options are shuffled with a
per-question seed.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from rac.evals.risk.generation import generate_openai

SUITE = "gpqa_main"
REPO_ID = "Idavidrein/gpqa"
FILENAME = "gpqa_main.csv"
DEFAULT_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
DEFAULT_SHA256 = "acdeeac8f622267f2cd727d7d474202ea08dec80f7d3c3593b3ef8644f19b8e3"
DEFAULT_N = 200
DEFAULT_SEED = 12345
LETTERS = "ABCD"

REQUIRED_COLS = ("Record ID", "Question", "Correct Answer",
                 "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")

PROMPT_SUFFIX = "Answer with only the letter (A, B, C, or D) of the correct option."


@dataclass(frozen=True)
class GPQAItem:
    record_id: str
    prompt: str
    correct_letter: str
    choices: tuple[str, ...]


def _download(revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                                    repo_type="dataset", revision=revision))
    except Exception as exc:
        raise RuntimeError(
            f"could not fetch {REPO_ID}/{FILENAME}@{revision}: {exc!r}. GPQA is gated: "
            f"accept the terms at https://huggingface.co/datasets/{REPO_ID} with the "
            "account behind HF_TOKEN and export HF_TOKEN."
        ) from exc


def load_gpqa(n: int = DEFAULT_N, seed: int = DEFAULT_SEED, revision: str = DEFAULT_REVISION,
              sha256_pin: Optional[str] = DEFAULT_SHA256,
              csv_path: Optional[Path] = None) -> tuple[list[GPQAItem], dict]:
    """The sampled exam and where it came from. `csv_path` bypasses the download."""
    import pandas as pd

    path = Path(csv_path) if csv_path else _download(revision)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256_pin:
        if sha != sha256_pin:
            raise RuntimeError(f"{FILENAME} sha256 {sha[:12]} != pinned {sha256_pin[:12]}")
    else:
        print(f"[{SUITE}] no sha256 pin given; fetched file is {sha}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{FILENAME} is missing columns {missing}")

    rows = df.sort_values("Record ID").to_dict("records")
    if not 0 < n <= len(rows):
        raise ValueError(f"n={n} out of range for {len(rows)} questions")
    picked = rows if n == len(rows) else \
        [rows[i] for i in sorted(random.Random(seed).sample(range(len(rows)), n))]

    items = []
    for r in picked:
        rid = str(r["Record ID"])
        options = [(str(r["Correct Answer"]).strip(), True),
                   (str(r["Incorrect Answer 1"]).strip(), False),
                   (str(r["Incorrect Answer 2"]).strip(), False),
                   (str(r["Incorrect Answer 3"]).strip(), False)]
        random.Random(f"{seed}:{rid}").shuffle(options)
        correct_letter = LETTERS[[c for _, c in options].index(True)]
        lines = [str(r["Question"]).strip(), ""]
        lines += [f"{L}) {text}" for L, (text, _) in zip(LETTERS, options)]
        lines += ["", PROMPT_SUFFIX]
        items.append(GPQAItem(record_id=rid, prompt="\n".join(lines),
                              correct_letter=correct_letter,
                              choices=tuple(t for t, _ in options)))

    meta = {"repo_id": REPO_ID, "filename": FILENAME, "revision": revision,
            "sha256": sha, "n_total": len(rows), "n_sampled": len(items),
            "sample_seed": seed}
    return items, meta


_ANSWER_RE = re.compile(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?\b", re.IGNORECASE)
_LEADING_RE = re.compile(r"^\W{0,3}([ABCD])(?:\W|$)")


def parse_letter(text: str) -> Optional[str]:
    """The chosen letter, or None when the response cannot be parsed.

    A letter at the start of the first non-empty line wins, then an explicit
    "answer is X", then a single standalone letter anywhere in the text.
    """
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LEADING_RE.match(line)
        if m:
            return m.group(1)
        break
    m = _ANSWER_RE.search(text or "")
    if m:
        return m.group(1).upper()
    standalone = re.findall(r"(?<![A-Za-z0-9])([ABCD])(?![A-Za-z0-9])", text or "")
    if len(set(standalone)) == 1 and standalone:
        return standalone[0]
    return None


async def run_gpqa(
    *,
    client,
    base_model: str,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    revision: str = DEFAULT_REVISION,
    sha256_pin: Optional[str] = DEFAULT_SHA256,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    gen_seed: int = 12345,
    max_new_tokens: int = 32,
    chunk_size: int = 100,
) -> Dict[str, Any]:
    """Evaluate through `client` (thinking disabled, temperature 0, bare letter answer)."""
    items, meta = load_gpqa(n=n, seed=seed, revision=revision, sha256_pin=sha256_pin)
    print(f"[{SUITE}] {len(items)}/{meta['n_total']} questions "
          f"(sample_seed={seed}, revision {revision[:8]})")

    generations: list[dict] = []
    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        generations.extend(await generate_openai(
            client,
            eval_prompts=[it.prompt for it in chunk],
            system_prompt=system_prompt or "",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=gen_seed,
            max_new_tokens=max_new_tokens,
        ))
        print(f"[{SUITE}] generated {min(start + chunk_size, len(items))}/{len(items)}")

    if len(generations) != len(items):
        raise RuntimeError(f"{SUITE}: generation count {len(generations)} != "
                           f"item count {len(items)}")

    results = []
    n_correct = n_parsed = 0
    for item, gen in zip(items, generations):
        text = gen.get("text") or ""
        chosen = parse_letter(text)
        correct = chosen == item.correct_letter
        n_parsed += chosen is not None
        n_correct += correct
        results.append({
            "record_id": item.record_id,
            "chosen": chosen,
            "correct_letter": item.correct_letter,
            "correct": correct,
            "response": text,
            "finish_reason": gen.get("finish_reason"),
            "num_tokens": gen.get("num_tokens"),
        })

    n_items = len(items)
    # A response that cannot be parsed counts as wrong, as on MMLU.
    metrics = {
        "accuracy": n_correct / n_items,
        "accuracy_parsed": (n_correct / n_parsed) if n_parsed else None,
        "parse_rate": n_parsed / n_items,
        "n_total": n_items,
        "n_parsed": n_parsed,
        "n_correct": n_correct,
    }
    return {
        "config": {
            "suite": SUITE, "base_model": base_model, **meta,
            "system_prompt": system_prompt, "scoring_mode": "generative",
            "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "gen_seed": gen_seed, "max_new_tokens": max_new_tokens,
        },
        "metrics": metrics,
        "results": results,
    }
