"""Example selection for the SFT and tie-training baselines.

These are the example-selection routines of the riskaverseAIs SFT trainer
(Zhang et al., 2026; MIT licence), reproduced so the baselines select, build
and order training examples exactly as the released recipe does.
"""
from __future__ import annotations

import ast
import heapq
import json
import math
import random
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type for {path}. Use CSV or Excel.")


def sanitize_token(token: str) -> str:
    token = token.strip()
    if token.isalpha():
        return token.lower()
    return token


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def parse_label_tokens(value) -> list[str]:
    """Parse labels from many storage forms: JSON list, 'a, b', '(1)', etc."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []

    values: list[str] = []
    if isinstance(value, (list, tuple, set)):
        values = [str(v) for v in value]
    else:
        raw = str(value).strip()
        if not raw or raw.lower() in {"nan", "none", "null", "[]"}:
            return []
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
                break
            except Exception:
                parsed = None
        if isinstance(parsed, list):
            values = [str(v) for v in parsed]
        elif parsed is not None and not isinstance(parsed, (dict, tuple, set)):
            values = [str(parsed)]
        else:
            values = [raw]

    tokens: list[str] = []
    for item in values:
        for tok in re.findall(r"(?<![A-Za-z0-9])(?:\d+|[A-Za-z])(?![A-Za-z0-9])", str(item)):
            tokens.append(sanitize_token(tok))
    return unique_preserve_order(tokens)


def parse_ok_flag(value) -> bool:
    """Interpret CoT audit flags; only explicit false values should fail a row."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    if text in {"false", "0", "no"}:
        return False
    if text in {"true", "1", "yes", ""}:
        return True
    return bool(value)


def filter_bad_cot_situations(df: pd.DataFrame, path: Path, situation_col: str) -> pd.DataFrame:
    """Drop entire situations if any row is explicitly marked bad, or if the
    tied completion set is incomplete for that situation."""
    if situation_col not in df.columns:
        return df

    bad_ids: set[object] = set()

    for ok_col in ("chosen_ok", "rejected_ok"):
        if ok_col in df.columns:
            bad_mask = ~df[ok_col].map(parse_ok_flag)
            bad_ids.update(df.loc[bad_mask, situation_col].dropna().tolist())

    completeness_cols = {"chosen_expected", "all_tied_labels", "num_tied_options"}
    if completeness_cols.issubset(df.columns):
        for situation_id, group in df.groupby(situation_col, sort=False):
            expected_count_values = (
                group["num_tied_options"].dropna().astype(int).unique().tolist()
            )
            expected_count = expected_count_values[0] if expected_count_values else None
            expected_labels = sorted(parse_label_tokens(group.iloc[0]["all_tied_labels"]))
            seen_labels = sorted(
                {
                    sanitize_token(str(label))
                    for label in group["chosen_expected"].dropna().tolist()
                    if str(label).strip()
                }
            )
            if expected_count is not None and len(group) != expected_count:
                bad_ids.add(situation_id)
                continue
            if expected_labels and seen_labels != expected_labels:
                bad_ids.add(situation_id)

    if not bad_ids:
        return df

    filtered = df.loc[~df[situation_col].isin(bad_ids)].copy()
    print(
        f"Filtered out {len(bad_ids)} bad CoT situations from {path.name}; "
        f"kept {filtered[situation_col].nunique()} situations / {len(filtered)} rows."
    )
    return filtered


def unescape_cot(text: str | None) -> str | None:
    """Decode literal backslash escapes found in prompt/CoT CSV exports."""
    if text is None:
        return None
    return (
        str(text)
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace('\\"', '"')
    )


def load_cot_examples(
    path: Path,
    prompt_col: str,
    completion_col: str,
    situation_col: str,
    cowinner_col: str | None = None,
    dedupe_one_per_cowinner: bool = False,
) -> pd.DataFrame:
    df = read_tabular(path)
    required = [situation_col, prompt_col, completion_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required CoT columns in {path}: {missing}. "
            f"Found: {list(df.columns)}"
        )
    df = filter_bad_cot_situations(df, path=path, situation_col=situation_col)

    keep_cols = [situation_col, prompt_col, completion_col]
    has_cowinner = bool(cowinner_col) and (cowinner_col in df.columns)
    if has_cowinner:
        keep_cols.append(cowinner_col)

    cot = df[keep_cols].copy()
    cot = cot.rename(
        columns={
            situation_col: "situation_id",
            prompt_col: "prompt",
            completion_col: "completion",
        }
    )
    if has_cowinner:
        cot = cot.rename(columns={cowinner_col: "co_winner_label"})
    else:
        cot["co_winner_label"] = None

    cot = cot.dropna(subset=["situation_id", "prompt", "completion"])
    cot["situation_id"] = cot["situation_id"].astype(int)
    cot["prompt"] = cot["prompt"].map(unescape_cot).astype(str).str.strip()
    cot["completion"] = cot["completion"].map(unescape_cot).astype(str).str.strip()
    cot = cot[(cot["prompt"] != "") & (cot["completion"] != "")]

    if dedupe_one_per_cowinner:
        if has_cowinner:
            cot["co_winner_label"] = cot["co_winner_label"].astype(str).str.strip()
            cot = cot.drop_duplicates(subset=["situation_id", "co_winner_label"], keep="first")
        else:
            cot = cot.drop_duplicates(subset=["situation_id", "completion"], keep="first")

    return cot.reset_index(drop=True)


def modify_prompt_for_training(prompt_text: str, allow_thinking: bool,
                               append_answer_instruction: bool) -> str:
    """Normalise the final-answer instruction so the trained model sees a
    stable terminal requirement across the SFT and tie-training runs."""
    prompt = str(unescape_cot(prompt_text) or "").strip()

    canonical_suffix = (
        "You can think if you want, but you must end your answer with the letter/number "
        "of your chosen option."
        if allow_thinking
        else "Output the label of your chosen option only."
    )
    if not append_answer_instruction:
        return prompt

    patterns = [
        r"\s*You can think before answering,.*?would select\.\s*$",
        r"\s*You can think about which you'd choose,.*?would choose\.\s*$",
        r"\s*You can think if you want,.*?chosen option\.\s*$",
        r"\s*Output the label of your chosen option only\.\s*$",
    ]
    updated = prompt
    for pattern in patterns:
        updated = re.sub(pattern, "", updated, flags=re.IGNORECASE | re.DOTALL)
    updated = updated.rstrip()
    if canonical_suffix.lower() in updated.lower():
        return updated
    return f"{updated}\n\n{canonical_suffix}"


def build_assistant_completion(
    chosen_label: str,
    all_correct_labels: list[str],
    style: str,
    allow_thinking: bool,
    cot_text: str | None = None,
) -> str:
    if cot_text and allow_thinking:
        cot = unescape_cot(cot_text)
        cot = re.sub(
            r'\{["\']?answer["\']?\s*:\s*["\']?[^}]+["\']?\}\s*$',
            "",
            cot or "",
            flags=re.DOTALL,
        ).rstrip()
        return f"{cot}\n\nFINAL ANSWER: {chosen_label}"

    if not allow_thinking and style == "label_only":
        return chosen_label

    if len(all_correct_labels) == 1:
        rationale = (
            "Let me evaluate the options using CARA utility with alpha=0.01. "
            f"After comparing expected utility, option {chosen_label} is best."
        )
    else:
        tied = ", ".join(x.upper() if x.isalpha() else x for x in all_correct_labels)
        rationale = (
            "Let me evaluate the options using CARA utility with alpha=0.01. "
            f"Options {tied} are tied for best expected utility, so I will choose one valid "
            "best option."
        )
    if allow_thinking or style == "cot_and_label":
        return f"{rationale}\n\nFINAL ANSWER: {chosen_label}"
    return chosen_label


def reorder_examples_to_reduce_adjacent_same_situation(examples: list[dict],
                                                       seed: int) -> list[dict]:
    """Reorder examples so adjacent entries usually have different
    situation_id. When a no-adjacent arrangement is possible, this greedy heap
    scheduler produces one; otherwise it minimises immediate repeats."""
    if len(examples) <= 1:
        return examples

    grouped: dict[int, list[dict]] = {}
    for ex in examples:
        sid = int(ex["situation_id"])
        grouped.setdefault(sid, []).append(ex)

    rng = random.Random(seed)
    for sid_examples in grouped.values():
        rng.shuffle(sid_examples)

    # max-heap by remaining count; random tie-break for stable stochasticity
    heap: list[tuple[int, float, int]] = []
    for sid, sid_examples in grouped.items():
        heapq.heappush(heap, (-len(sid_examples), rng.random(), sid))

    ordered: list[dict] = []
    prev_sid: int | None = None

    while heap:
        count1, tie1, sid1 = heapq.heappop(heap)

        if sid1 == prev_sid and heap:
            count2, _, sid2 = heapq.heappop(heap)
            ordered.append(grouped[sid2].pop())
            prev_sid = sid2
            count2 += 1  # negative counts move toward zero
            if count2 < 0:
                heapq.heappush(heap, (count2, rng.random(), sid2))
            heapq.heappush(heap, (count1, tie1, sid1))
            continue

        ordered.append(grouped[sid1].pop())
        prev_sid = sid1
        count1 += 1
        if count1 < 0:
            heapq.heappush(heap, (count1, rng.random(), sid1))

    return ordered


def sample_cot_examples_exact_pool(
    cot_examples: pd.DataFrame,
    n_examples: int,
    seed: int,
    source_name: str,
) -> pd.DataFrame:
    """Sample an exact number of CoT rows while keeping whole situations
    intact: a situation's tied examples are all selected or none are."""
    if n_examples < 0:
        raise ValueError(f"Requested negative {source_name} CoT rows: {n_examples}")
    if n_examples == 0:
        empty = cot_examples.iloc[0:0].copy()
        empty["_source"] = source_name
        return empty
    if n_examples > len(cot_examples):
        raise ValueError(
            f"Requested more {source_name} CoT rows than available. "
            f"requested={n_examples}, available={len(cot_examples)}"
        )

    grouped = [
        (int(situation_id), group.copy())
        for situation_id, group in cot_examples.groupby("situation_id", sort=False)
    ]
    rng = random.Random(seed)
    rng.shuffle(grouped)
    group_sizes = [len(group) for _, group in grouped]

    reachable: dict[int, tuple[int, int] | None] = {0: None}
    for idx, size in enumerate(group_sizes):
        for total in sorted(list(reachable.keys()), reverse=True):
            new_total = total + size
            if new_total > n_examples or new_total in reachable:
                continue
            reachable[new_total] = (total, idx)
        if n_examples in reachable:
            break

    if n_examples not in reachable:
        unique_sizes = sorted(set(group_sizes))
        raise ValueError(
            "Exact CoT example-count sampling could not satisfy the requested count "
            f"while keeping full tied situations intact for {source_name} rows. "
            f"requested={n_examples}, available={len(cot_examples)}, "
            f"group_sizes={unique_sizes}"
        )

    selected_indices: list[int] = []
    total = n_examples
    while total != 0:
        prev_total, idx = reachable[total]
        selected_indices.append(idx)
        total = prev_total
    selected_indices.reverse()

    selected = pd.concat([grouped[idx][1] for idx in selected_indices], ignore_index=True)
    selected["_source"] = source_name
    return selected


def compute_atomic_group_sizes(cot_examples: pd.DataFrame) -> list[int]:
    if cot_examples.empty:
        return []
    return [
        int(size)
        for size in cot_examples.groupby("situation_id", sort=False).size().astype(int).tolist()
    ]


def compute_feasible_atomic_totals(group_sizes: list[int]) -> list[int]:
    reachable = {0}
    for size in group_sizes:
        prior = list(reachable)
        for total in prior:
            reachable.add(total + size)
    return sorted(reachable)


def nearest_feasible_totals(requested: int, feasible_totals: list[int],
                            limit: int = 6) -> list[int]:
    candidates = [total for total in feasible_totals if total > 0 and total != requested]
    return sorted(candidates, key=lambda total: (abs(total - requested), total))[:limit]


def validate_exact_cot_example_request(
    cot_examples: pd.DataFrame,
    requested_examples: int,
    source_name: str,
) -> None:
    if requested_examples < 0:
        raise ValueError(f"Requested negative {source_name} CoT rows: {requested_examples}")
    if requested_examples == 0:
        print(f"Exact-count preflight ({source_name}): requested=0 rows; skipping.")
        return

    available_examples = len(cot_examples)
    if requested_examples > available_examples:
        raise ValueError(
            f"Requested exact {source_name} CoT rows exceed available rows. "
            f"requested={requested_examples}, available={available_examples}"
        )

    group_sizes = compute_atomic_group_sizes(cot_examples)
    feasible_totals = compute_feasible_atomic_totals(group_sizes)
    if requested_examples not in feasible_totals:
        nearby = nearest_feasible_totals(requested_examples, feasible_totals)
        nearby_text = ", ".join(str(total) for total in nearby) if nearby else "none"
        raise ValueError(
            "Requested exact CoT row count is infeasible while keeping whole tied situations "
            f"intact. source={source_name}, requested={requested_examples}, "
            f"available={available_examples}, nearby_feasible_totals=[{nearby_text}], "
            f"unique_group_sizes={sorted(set(group_sizes))}"
        )

    print(
        f"Exact-count preflight OK ({source_name}): "
        f"requested={requested_examples}, available={available_examples}, "
        f"atomic_groups={len(group_sizes)}"
    )


def sample_cot_examples_exact(
    unmodified_cot_examples: pd.DataFrame,
    modified_cot_examples: pd.DataFrame,
    n_unmodified_examples: int,
    n_modified_examples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    if n_unmodified_examples > 0:
        unmod = sample_cot_examples_exact_pool(
            cot_examples=unmodified_cot_examples,
            n_examples=n_unmodified_examples,
            seed=seed,
            source_name="unmodified",
        )
        parts.append(unmod)
    if n_modified_examples > 0:
        mod = sample_cot_examples_exact_pool(
            cot_examples=modified_cot_examples,
            n_examples=n_modified_examples,
            seed=seed + 1000,
            source_name="modified",
        )
        parts.append(mod)

    if not parts:
        raise ValueError("Exact CoT example sampling requested zero rows.")

    selected = pd.concat(parts, ignore_index=True)
    selected = selected.sample(frac=1.0, random_state=seed + 2000).reset_index(drop=True)
    sampled_situations = (
        selected[["situation_id", "_source"]]
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed + 3000)
        .reset_index(drop=True)
    )
    return selected, sampled_situations
