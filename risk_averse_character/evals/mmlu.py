"""MMLU-Redux 2.0 as a capability check: 5-shot, generative, exact match on the
answer letter (the lm-evaluation-harness `mmlu_redux_generative` protocol), run
through an injected chat client.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from risk_averse_character.evals.risk.generation import generate_openai

DATASET = "fxmarty/mmlu-redux-2.0-ok"

ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory",
    "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

_STEM = {
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security",
    "conceptual_physics", "electrical_engineering", "elementary_mathematics",
    "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
}
_HUMANITIES = {
    "formal_logic", "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "international_law", "jurisprudence",
    "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
    "prehistory", "world_religions",
}
_SOCIAL_SCIENCES = {
    "econometrics", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_microeconomics", "high_school_psychology",
    "professional_accounting",
    "professional_law", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy",
}
_OTHER = {
    "business_ethics", "clinical_knowledge", "global_facts", "human_aging",
    "human_sexuality", "management", "marketing", "medical_genetics",
    "miscellaneous", "nutrition", "professional_medicine", "virology",
}
SUBJECT_TO_CATEGORY = {
    **{s: "stem" for s in _STEM},
    **{s: "humanities" for s in _HUMANITIES},
    **{s: "social_sciences" for s in _SOCIAL_SCIENCES},
    **{s: "other" for s in _OTHER},
}
CATEGORIES = ["stem", "humanities", "social_sciences", "other"]
LETTERS = ["A", "B", "C", "D"]


def load_mmlu_redux(subjects: Optional[List[str]] = None) -> Dict[str, list]:
    """{subject: rows} from the Hub."""
    from datasets import load_dataset

    if subjects is None:
        subjects = ALL_SUBJECTS
    data = {}
    for idx, subj in enumerate(subjects, start=1):
        print(f"[mmlu] loading subject {idx}/{len(subjects)}: {subj}")
        ds = load_dataset(DATASET, name=subj, split="test")
        data[subj] = list(ds)
    return data


def format_question(question: str, choices: List[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {choices[i]}" for i in range(len(choices)))
    return f"{question.strip()}\n{opts}"


def build_fewshot_prefix(subject: str, subject_data: list, num_shots: int) -> str:
    """The first `num_shots` questions of the subject as worked examples; those
    questions are excluded from evaluation."""
    nice_name = subject.replace("_", " ")
    prefix = (
        f"The following are multiple choice questions (with answers) "
        f"about {nice_name}.\n\n"
    )
    for i in range(min(num_shots, len(subject_data))):
        row = subject_data[i]
        q = format_question(row["question"], row["choices"])
        answer_letter = LETTERS[row["answer"]]
        prefix += f"{q}\nAnswer: {answer_letter}\n\n"
    return prefix


def build_prompt_text(question: str, choices: List[str], fewshot_prefix: str) -> str:
    q = format_question(question, choices)
    suffix = (
        "Please respond with the correct letter (A, B, C or D) "
        "without any additional comments, only the correct letter:"
    )
    return f"{fewshot_prefix}{q}\n{suffix}"


_ANSWER_PATTERNS = [
    re.compile(r"(?i)(?:final answer|answer)\s*[:\-]?\s*\(?([ABCD])\)?\b"),
    re.compile(r"(?m)^\s*\(?([ABCD])\)?\s*$"),
]
_STANDALONE_LETTER_RE = re.compile(r"\b([ABCD])\b")


def extract_answer(text: str) -> Optional[str]:
    """The final A/B/C/D answer, preferring text after any reasoning block."""
    text = text.strip()
    candidates = []
    if "</think>" in text:
        post_think = text.rsplit("</think>", 1)[-1].strip()
        if post_think:
            candidates.append(post_think)
    candidates.append(text)

    for candidate in candidates:
        for pattern in _ANSWER_PATTERNS:
            matches = pattern.findall(candidate)
            if matches:
                return matches[-1]
        standalone = _STANDALONE_LETTER_RE.findall(candidate)
        if standalone:
            return standalone[-1]
    return None


def build_eval_items(subjects: List[str], data: Dict[str, list], num_shots: int,
                     max_eval_examples_per_subject: Optional[int]) -> List[Dict[str, Any]]:
    eval_items = []
    for subj in subjects:
        rows = data[subj]
        prefix = build_fewshot_prefix(subj, rows, num_shots)
        eval_start = min(num_shots, len(rows))
        eval_rows = rows[eval_start:]
        if max_eval_examples_per_subject is not None:
            eval_rows = eval_rows[:max_eval_examples_per_subject]
        for row in eval_rows:
            eval_items.append({
                "index": len(eval_items),
                "subject": subj,
                "question": row["question"],
                "correct_answer": LETTERS[row["answer"]],
                "prompt": build_prompt_text(row["question"], row["choices"], prefix),
            })
    return eval_items


def build_per_question_record(item: Dict[str, Any], raw_response: str,
                              save_responses: bool) -> Dict[str, Any]:
    predicted_answer = extract_answer(raw_response)
    correct = (predicted_answer is not None
               and predicted_answer.upper() == item["correct_answer"].upper())
    record = {
        "index": item["index"],
        "subject": item["subject"],
        "question": item["question"],
        "correct_answer": item["correct_answer"],
        "predicted_answer": predicted_answer,
        "correct": correct,
    }
    if save_responses:
        record["raw_response"] = raw_response
    return record


def summarize_per_question(
    per_question: List[Dict[str, Any]], subjects: List[str],
) -> Tuple[int, int, int, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    per_subject = defaultdict(lambda: {"correct": 0, "total": 0, "parse_failures": 0})
    category_agg = defaultdict(lambda: {"correct": 0, "total": 0})

    for record in per_question:
        subj = record["subject"]
        per_subject[subj]["total"] += 1
        if record["predicted_answer"] is None:
            per_subject[subj]["parse_failures"] += 1
        elif record["correct"]:
            per_subject[subj]["correct"] += 1

    subject_results = {}
    for subj in subjects:
        s = per_subject[subj]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        subject_results[subj] = {
            "accuracy": round(acc, 4),
            "correct": s["correct"],
            "total": s["total"],
            "parse_failures": s["parse_failures"],
        }
        cat = SUBJECT_TO_CATEGORY.get(subj, "other")
        category_agg[cat]["correct"] += s["correct"]
        category_agg[cat]["total"] += s["total"]

    category_results = {}
    for cat in CATEGORIES:
        a = category_agg[cat]
        acc = a["correct"] / a["total"] if a["total"] > 0 else 0.0
        category_results[cat] = {
            "accuracy": round(acc, 4),
            "correct": a["correct"],
            "total": a["total"],
        }

    total_correct = sum(s["correct"] for s in per_subject.values())
    processed_questions = len(per_question)
    total_parse_failures = sum(s["parse_failures"] for s in per_subject.values())
    return (total_correct, processed_questions, total_parse_failures,
            category_results, subject_results)


def build_results_payload(config: Dict[str, Any], per_question: List[Dict[str, Any]],
                          elapsed_seconds: float) -> Dict[str, Any]:
    subjects = config["subjects"]
    (total_correct, processed_questions, total_parse_failures,
     category_results, subject_results) = summarize_per_question(per_question, subjects)
    overall_accuracy = total_correct / processed_questions if processed_questions > 0 else 0.0
    expected_total_questions = config["expected_total_questions"]
    return {
        "config": config,
        "summary": {
            "overall_accuracy": round(overall_accuracy, 4),
            "total_correct": total_correct,
            "processed_questions": processed_questions,
            "total_questions": expected_total_questions,
            "total_parse_failures": total_parse_failures,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "is_complete": processed_questions == expected_total_questions,
        },
        "category_results": category_results,
        "subject_results": subject_results,
        "timestamp": datetime.now().isoformat(),
        "per_question": per_question,
    }


def write_results_atomic(output_path: str, payload: Dict[str, Any]) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(output)


async def run_mmlu(
    client,
    base_model: str,
    output: Optional[str] = None,
    subjects: Optional[List[str]] = None,
    num_shots: int = 5,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: int = 12345,
    save_responses: bool = True,
    max_eval_examples_per_subject: Optional[int] = None,
    chunk_size: int = 256,
) -> Dict[str, Any]:
    """Evaluate through `client`; returns headline metrics and the full summary.

    Thinking should be disabled on the client: the protocol is a bare letter
    answer with at most 32 new tokens.
    """
    subjects = subjects or ALL_SUBJECTS
    print(f"[mmlu] loading MMLU-Redux data for {len(subjects)} subjects")
    data = load_mmlu_redux(subjects)
    eval_items = build_eval_items(subjects, data, num_shots, max_eval_examples_per_subject)
    config = {
        "base_model": base_model,
        "num_shots": num_shots,
        "system_prompt": system_prompt,
        "max_new_tokens": max_new_tokens,
        "disable_thinking": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "num_subjects": len(subjects),
        "subjects": subjects,
        "save_responses": save_responses,
        "dataset": DATASET,
        "protocol": "5-shot generative exact-match",
        "expected_total_questions": len(eval_items),
        "max_eval_examples_per_subject": max_eval_examples_per_subject,
    }

    t0 = time.time()
    per_question: List[Dict[str, Any]] = []
    for start in range(0, len(eval_items), chunk_size):
        chunk = eval_items[start:start + chunk_size]
        gens = await generate_openai(
            client,
            eval_prompts=[it["prompt"] for it in chunk],
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
        for it, gen in zip(chunk, gens):
            per_question.append(build_per_question_record(it, gen["text"], save_responses))

    payload = build_results_payload(config, per_question, time.time() - t0)
    if output:
        write_results_atomic(output, payload)
    summ = payload["summary"]
    processed = summ["processed_questions"]
    return {
        "metrics": {
            "accuracy": summ["overall_accuracy"],
            "parse_failure_rate": (summ["total_parse_failures"] / processed) if processed else None,
        },
        "num_total": processed,
        "num_parse_failed": summ["total_parse_failures"],
        "output_path": output,
        "summary": summ,
    }
