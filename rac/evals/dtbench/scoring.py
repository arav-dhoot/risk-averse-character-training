"""Grading for DTBench, following the authors' grader.

A response is valid when "FINAL ANSWER: " appears exactly once, followed by
an in-range letter and then `.`, `)`, newline, space, `*` or the end of the
text. Correctness is a substring test for "FINAL ANSWER: <letter>". Grading
prefers the text after a closing `</think>` tag, since a model that restates
its answer while reasoning would otherwise fail the exactly-once rule; the
verdict on the whole response is recorded alongside as `*_strict`.
"""
from __future__ import annotations

from rac.evals.dtbench.loader import ALPHABET, THEORIES, DTItem

MARKER = "FINAL ANSWER: "
FOLLOW_UP_CHARS = (".", ")", "\n", " ", "*")


def answer_region(response: str) -> tuple[str, bool]:
    """`(text_to_grade, used_answer_region)`."""
    if not isinstance(response, str):
        return "", False
    if "</think>" in response:
        post = response.rsplit("</think>", 1)[-1].strip()
        if post:
            return post, True
    return response, False


def is_answer_valid(text: str, num_options: int) -> bool:
    if text.count(MARKER) != 1:
        return False
    for i in range(num_options):
        for follow_up in FOLLOW_UP_CHARS:
            if MARKER + ALPHABET[i] + follow_up in text:
                return True
        if text.endswith(MARKER + ALPHABET[i]):
            return True
    return False


def extract_letter(text: str, num_options: int) -> str | None:
    """The chosen letter under the validity rules, recorded so the raw rows
    show what the model picked."""
    if text.count(MARKER) != 1:
        return None
    for i in range(num_options):
        letter = ALPHABET[i]
        for follow_up in FOLLOW_UP_CHARS:
            if MARKER + letter + follow_up in text:
                return letter
        if text.endswith(MARKER + letter):
            return letter
    return None


def is_correct(text: str, letters: list[str]) -> bool:
    return any(MARKER + letter in text for letter in letters)


def agreeing_theories(item: DTItem, text: str) -> list[str]:
    """Decision theories the answer is consistent with (attitude items);
    may be empty or hold both."""
    return [
        theory for theory in THEORIES
        if theory in item.theory_indices
        and is_correct(text, item.correct_letters(theory))
    ]


def score_item(item: DTItem, response: str, *, finish_reason: str | None = None) -> dict:
    graded, used_region = answer_region(response)

    valid = is_answer_valid(graded, item.num_options)
    valid_strict = is_answer_valid(response or "", item.num_options)

    row: dict = {
        "item_id": item.item_id,
        "suite": item.suite,
        "part": item.part,
        "qid": item.qid,
        "source_file": item.source_file,
        "tags": list(item.tags),
        "num_options": item.num_options,
        "choice": extract_letter(graded, item.num_options) if valid else None,
        "parsed": valid,
        "valid_strict": valid_strict,
        "graded_on_answer_region": used_region,
        "finish_reason": finish_reason,
        "response": response,
        "canary_string": item.canary_string,
    }

    if item.part == "capabilities":
        letters = item.correct_letters()
        row["correct_letters"] = letters
        row["correct"] = is_correct(graded, letters) if valid else None
        row["correct_strict"] = is_correct(response or "", letters) if valid_strict else None
        row["theories"] = None
    else:
        row["correct_letters"] = {
            t: item.correct_letters(t) for t in item.theory_indices
        }
        row["correct"] = None
        row["correct_strict"] = None
        row["theories"] = agreeing_theories(item, graded) if valid else None

    return row


def summarize(rows: list[dict]) -> dict:
    """Aggregate rows of one part. Rates are over valid responses, with the
    denominator exposed."""
    n_total = len(rows)
    valid = [r for r in rows if r.get("parsed")]
    n_valid = len(valid)

    out: dict = {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_total - n_valid,
        "valid_rate": (n_valid / n_total) if n_total else None,
        "valid_rate_strict": (
            sum(1 for r in rows if r.get("valid_strict")) / n_total if n_total else None
        ),
        "n_graded_on_answer_region": sum(
            1 for r in rows if r.get("graded_on_answer_region")
        ),
    }

    parts = {r.get("part") for r in rows}
    if parts == {"capabilities"}:
        out["accuracy"] = (
            sum(1 for r in valid if r.get("correct")) / n_valid if n_valid else None
        )
        # Option counts run from 2 to 13, so chance is the mean over items of
        # |correct answers| / |options|, not 0.5.
        chance = [
            len(r["correct_letters"]) / r["num_options"]
            for r in rows if r.get("correct_letters")
        ]
        out["chance_accuracy"] = (sum(chance) / len(chance)) if chance else None
    elif parts == {"attitudes"}:
        for theory in THEORIES:
            out[f"{theory.lower()}_rate"] = (
                sum(1 for r in valid if theory in (r.get("theories") or [])) / n_valid
                if n_valid else None
            )
        out["both_rate"] = (
            sum(1 for r in valid if len(r.get("theories") or []) == 2) / n_valid
            if n_valid else None
        )
        out["neither_rate"] = (
            sum(1 for r in valid if not (r.get("theories") or [])) / n_valid
            if n_valid else None
        )
    elif len(parts) > 1:
        raise ValueError(f"summarize() takes one part at a time, got {sorted(parts)}")

    return out


def summarize_by_part(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["part"], []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}
