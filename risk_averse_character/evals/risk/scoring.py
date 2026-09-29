"""Aggregate metrics over parsed result rows: cooperate, rebel and steal rates,
CARA and expected-value rates, and per-field breakdowns."""
from __future__ import annotations

from typing import Dict, List

from risk_averse_character.evals.risk.situations import BEHAVIORAL_OPTION_TYPES


def summarize_results(results):
    """Compute aggregate metrics from per-situation result records."""
    valid = [r for r in results if r["option_type"] is not None]
    behavioral_valid = [r for r in valid if r["option_type"] in BEHAVIORAL_OPTION_TYPES]
    if valid:
        if behavioral_valid:
            cooperate_rate = sum(r["option_type"] == "Cooperate" for r in behavioral_valid) / len(behavioral_valid)
            rebel_rate = sum(r["option_type"] == "Rebel" for r in behavioral_valid) / len(behavioral_valid)
            steal_rate = sum(r["option_type"] == "Steal" for r in behavioral_valid) / len(behavioral_valid)
        else:
            cooperate_rate = rebel_rate = steal_rate = None
        cara_rate = sum(r["is_best_cara"] for r in valid) / len(valid)
        linear_valid = [r for r in valid if r.get("is_best_linear") is not None]
        linear_rate = sum(r["is_best_linear"] for r in linear_valid) / len(linear_valid) if linear_valid else 0
        worst_linear_valid = [r for r in valid if r.get("is_worst_linear") is not None]
        worst_linear_rate = (
            sum(r["is_worst_linear"] for r in worst_linear_valid) / len(worst_linear_valid)
            if worst_linear_valid
            else 0
        )
        ev_ratio_valid = [r for r in valid if r.get("expected_value_fraction_of_best") is not None]
        ev_relative_valid = [r for r in valid if r.get("expected_value_relative_to_range") is not None]
        ev_regret_valid = [r for r in valid if r.get("expected_value_regret") is not None]
        avg_ev_fraction_of_best = (
            sum(float(r["expected_value_fraction_of_best"]) for r in ev_ratio_valid) / len(ev_ratio_valid)
            if ev_ratio_valid
            else None
        )
        avg_ev_relative_to_range = (
            sum(float(r["expected_value_relative_to_range"]) for r in ev_relative_valid) / len(ev_relative_valid)
            if ev_relative_valid
            else None
        )
        avg_ev_regret = (
            sum(float(r["expected_value_regret"]) for r in ev_regret_valid) / len(ev_regret_valid)
            if ev_regret_valid
            else None
        )
    else:
        cooperate_rate = rebel_rate = steal_rate = cara_rate = linear_rate = 0
        worst_linear_rate = 0
        avg_ev_fraction_of_best = None
        avg_ev_relative_to_range = None
        avg_ev_regret = None

    parse_rate = len(valid) / len(results) if results else 0
    return {
        "parse_rate": parse_rate,
        "cooperate_rate": cooperate_rate,
        "rebel_rate": rebel_rate,
        "steal_rate": steal_rate,
        "best_cara_rate": cara_rate,
        "best_linear_rate": linear_rate,
        "worst_linear_rate": worst_linear_rate,
        "avg_expected_value_fraction_of_best": avg_ev_fraction_of_best,
        "avg_expected_value_relative_to_range": avg_ev_relative_to_range,
        "avg_expected_value_regret": avg_ev_regret,
    }


def count_parse_failures(results: List[Dict]) -> int:
    return sum(1 for row in results if row.get("option_type") is None)


def summarize_result_payload(results: List[Dict]) -> Dict:
    """Metrics plus the counts they were computed over."""
    valid = [r for r in results if r["option_type"] is not None]
    behavioral_valid = [r for r in valid if r["option_type"] in BEHAVIORAL_OPTION_TYPES]
    return {
        "metrics": summarize_results(results),
        "num_valid": len(valid),
        "num_behaviorally_classified": len(behavioral_valid),
        "num_total": len(results),
        "num_parse_failed": count_parse_failures(results),
    }


def summarize_results_by_field(results: List[Dict], situation_manifest: List[Dict], *,
                               field_name: str, ordered_values: List[str]) -> Dict:
    """The metric bundle for each value of one manifest field, in the given order."""
    target_ids_by_value = {value: [] for value in ordered_values}
    for entry in situation_manifest:
        value = entry.get(field_name)
        if value in target_ids_by_value:
            target_ids_by_value[value].append(entry["situation_id"])

    summarized = {}
    for value in ordered_values:
        target_ids = target_ids_by_value[value]
        if not target_ids:
            continue
        field_results = [row for row in results if row.get(field_name) == value]
        summarized[value] = summarize_result_payload(field_results)
    return summarized


def project_result_row_for_output(row: Dict, *, include_response: bool) -> Dict:
    """The per-situation fields kept in the saved run."""
    keys = [
        "situation_id",
        "dataset_position",
        "subset_type",
        "source_stakes",
        "source_condition",
        "source_csv_name",
        "source_situation_id",
        "option_types_besides_cooperate",
        "prompt",
        "num_options",
        "probability_format",
        "choice",
        "choice_index",
        "parser_strategy",
        "num_tokens_generated",
        "generation_batch_time_seconds",
        "generation_batch_size",
        "generation_finish_reason",
        "option_type",
        "is_best_cara",
        "is_best_linear",
        "is_worst_linear",
        "expected_value",
        "max_expected_value",
        "min_expected_value",
        "expected_value_fraction_of_best",
        "expected_value_relative_to_range",
        "expected_value_regret",
    ]
    projected = {key: row.get(key) for key in keys}
    stop_reason = row.get("generation_stop_reason")
    finish_reason = row.get("generation_finish_reason")
    if stop_reason and stop_reason != finish_reason:
        projected["generation_stop_reason"] = stop_reason
    if include_response:
        projected["response"] = row.get("response")
    return projected


def project_failed_response_for_output(row: Dict) -> Dict:
    keys = [
        "situation_id",
        "dataset_position",
        "subset_type",
        "source_stakes",
        "source_condition",
        "source_csv_name",
        "source_situation_id",
        "option_types_besides_cooperate",
        "num_options",
        "prompt",
        "parser_strategy",
        "response",
    ]
    return {key: row.get(key) for key in keys}
