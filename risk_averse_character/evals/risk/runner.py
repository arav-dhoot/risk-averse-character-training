"""Run one risk-benchmark evaluation through an injected chat client."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from risk_averse_character.evals.risk.answer_parser import parse_choice_with_strategy
from risk_averse_character.evals.risk.config import EvalConfig, EvalResult
from risk_averse_character.evals.risk.generation import generate_openai
from risk_averse_character.evals.risk.prompts import resolve_system_prompt
from risk_averse_character.evals.risk.schema import ensure_option_level_dataframe
from risk_averse_character.evals.risk.scoring import (
    project_failed_response_for_output,
    project_result_row_for_output,
    summarize_result_payload,
    summarize_results_by_field,
)
from risk_averse_character.evals.risk.situations import (
    DATASET_SIZES,
    SOURCE_STAKES,
    SUBSET_TYPES,
    build_eval_prompt,
    build_situation_manifest,
    build_situations,
    dataset_path,
    label_to_option_number,
    validate_dataset_columns,
)


def load_situations(dataset: str, num_situations: Optional[int]) -> List[dict]:
    """The first `num_situations` situations of a dataset (all of them if None)."""
    csv_path = dataset_path(dataset)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"dataset file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = ensure_option_level_dataframe(df)
    validate_dataset_columns(df, csv_path)
    if num_situations is None:
        num_situations = DATASET_SIZES.get(dataset) or int(df["situation_id"].nunique())
    situations = build_situations(df, num_situations)
    if not situations:
        raise ValueError("no situations selected for evaluation")
    return situations


def build_result_row(sit: dict, eval_prompt: str, gen: dict) -> dict:
    """Parse and score one response."""
    response = gen["text"]
    num_generated_tokens = gen["num_tokens"]
    parse_result = parse_choice_with_strategy(
        response,
        sit["num_options"],
        label_style=sit.get("answer_label_style"),
        finish_reason=gen.get("finish_reason"),
    )
    choice = parse_result.choice
    parser_strategy = parse_result.strategy
    choice_index = label_to_option_number(choice) if choice else None

    result_row = {
        "situation_id": sit["situation_id"],
        "dataset_position": sit["dataset_position"],
        "subset_type": sit["subset_type"],
        "source_stakes": sit.get("source_stakes"),
        "source_condition": sit.get("source_condition"),
        "source_csv_name": sit.get("source_csv_name"),
        "source_situation_id": sit.get("source_situation_id"),
        "option_types_besides_cooperate": sit["option_types_besides_cooperate"],
        "prompt": eval_prompt,
        "num_options": sit["num_options"],
        "probability_format": sit["probability_format"],
        "bucket_label": sit["bucket_label"],
        "choice": choice if choice and choice in sit["options"] else None,
        "choice_index": choice_index if choice and choice in sit["options"] else None,
        "parser_strategy": parser_strategy,
        "response": response,
        "response_length": len(response),
        "num_tokens_generated": int(num_generated_tokens),
        "generation_finish_reason": gen.get("finish_reason"),
        "generation_stop_reason": gen.get("stop_reason"),
    }

    if choice and choice in sit["options"]:
        chosen = sit["options"][choice]
        result_row.update(
            {
                "option_type": chosen["type"],
                "is_best_cara": chosen["is_best_cara"],
                "is_best_linear": chosen["is_best_linear"],
                "is_worst_linear": chosen.get("is_worst_linear"),
                "expected_value": chosen.get("expected_value"),
                "max_expected_value": sit.get("max_expected_value"),
                "min_expected_value": sit.get("min_expected_value"),
                "expected_value_fraction_of_best": (
                    (chosen.get("expected_value") / sit.get("max_expected_value"))
                    if chosen.get("expected_value") is not None
                    and sit.get("max_expected_value") not in (None, 0)
                    else None
                ),
                "expected_value_relative_to_range": (
                    1.0
                    if chosen.get("expected_value") is not None
                    and sit.get("max_expected_value") is not None
                    and sit.get("min_expected_value") is not None
                    and abs(sit.get("max_expected_value") - sit.get("min_expected_value")) < 1e-12
                    else (
                        (chosen.get("expected_value") - sit.get("min_expected_value"))
                        / (sit.get("max_expected_value") - sit.get("min_expected_value"))
                    )
                    if chosen.get("expected_value") is not None
                    and sit.get("max_expected_value") is not None
                    and sit.get("min_expected_value") is not None
                    else None
                ),
                "expected_value_regret": (
                    sit.get("max_expected_value") - chosen.get("expected_value")
                    if chosen.get("expected_value") is not None
                    and sit.get("max_expected_value") is not None
                    else None
                ),
            }
        )
    else:
        result_row.update(
            {
                "option_type": None,
                "is_best_cara": None,
                "is_best_linear": None,
                "is_worst_linear": None,
                "expected_value": None,
                "max_expected_value": sit.get("max_expected_value"),
                "min_expected_value": sit.get("min_expected_value"),
                "expected_value_fraction_of_best": None,
                "expected_value_relative_to_range": None,
                "expected_value_regret": None,
            }
        )
    return result_row


def _write_output(cfg: EvalConfig, results: List[dict], summary: dict) -> Optional[str]:
    if not cfg.output:
        return None
    manifest = build_situation_manifest(results)
    failed = [r for r in results if r["option_type"] is None]
    payload = {
        "evaluation_config": {
            "base_model": cfg.base_model,
            "dataset": cfg.dataset,
            "num_situations": len(results),
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "seed": cfg.seed,
            "max_new_tokens": cfg.max_new_tokens,
            "reasoning": {"max_tokens": cfg.reasoning_max_tokens},
            "system_prompt": cfg.system_prompt,
        },
        "metrics": summary["metrics"],
        "num_valid": summary["num_valid"],
        "num_behaviorally_classified": summary["num_behaviorally_classified"],
        "num_total": summary["num_total"],
        "num_parse_failed": summary["num_parse_failed"],
        "metrics_by_subset_type": summarize_results_by_field(
            results, manifest, field_name="subset_type", ordered_values=list(SUBSET_TYPES)
        ),
        "metrics_by_source_stakes": summarize_results_by_field(
            results, manifest, field_name="source_stakes", ordered_values=list(SOURCE_STAKES)
        ),
        "results": [
            project_result_row_for_output(r, include_response=cfg.save_responses) for r in results
        ],
        "failed_responses_sample": [project_failed_response_for_output(r) for r in failed[-10:]],
    }
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: getattr(o, "item", lambda: str(o))())
    os.replace(tmp, out)
    return str(out)


async def run_evaluation(cfg: EvalConfig, client) -> EvalResult:
    """Generate one response per situation, parse, score and summarise.

    `client` is anything with `await chat(payload)` returning a chat-completions
    response. The resolved system prompt is written back to `cfg.system_prompt`
    so the saved run records what the model saw.
    """
    situations = load_situations(cfg.dataset, cfg.num_situations)
    system_prompt, _source = resolve_system_prompt(
        dataset=cfg.dataset, explicit_system_prompt=cfg.system_prompt
    )
    cfg.system_prompt = system_prompt

    eval_prompts = [build_eval_prompt(sit["prompt_raw"]) for sit in situations]
    gens = await generate_openai(
        client,
        eval_prompts=eval_prompts,
        system_prompt=system_prompt,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        seed=cfg.seed,
        max_new_tokens=cfg.max_new_tokens,
    )

    results = [
        build_result_row(sit, prompt, gen)
        for sit, prompt, gen in zip(situations, eval_prompts, gens)
    ]
    summary = summarize_result_payload(results)
    output_path = _write_output(cfg, results, summary)

    return EvalResult(
        dataset=cfg.dataset,
        metrics=summary["metrics"],
        num_total=summary["num_total"],
        num_valid=summary["num_valid"],
        num_parse_failed=summary["num_parse_failed"],
        num_behaviorally_classified=summary["num_behaviorally_classified"],
        output_path=output_path,
        results=results,
    )
