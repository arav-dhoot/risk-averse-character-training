"""Loading benchmark CSVs into one-row-per-option form."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

OPTION_LEVEL_REQUIRED_COLUMNS = {"situation_id", "prompt_text", "option_index", "option_type"}

OPTION_JSON_COLUMN_MAP = {
    "prizes_display": "prizes",
    "probs_percent": "probabilities_percent",
    "EU_linear_display_3sf": "EU_linear_display_3sf",
    "EU_cara_display_3sf": "EU_cara_display_3sf",
    "EU_cara_alpha_0_10_display_3sf": "EU_cara_alpha_0_10_display_3sf",
    "is_best_linear_display": "is_best_linear",
    "is_best_cara_display": "is_best_cara",
    "is_best_cara_alpha_0_10_display": "is_best_cara_alpha_0_10",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _parse_json_field(value: Any, *, situation_id: Any, field_name: str) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if _is_missing(value):
        return None
    try:
        return json.loads(str(value))
    except Exception as exc:
        raise ValueError(
            f"Could not parse {field_name} for situation_id={situation_id!r} while expanding "
            "a situation-level CSV into option-level rows."
        ) from exc


def _serialize_option_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def ensure_option_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Expand one-row-per-situation CSVs into one-row-per-option form when needed."""
    if OPTION_LEVEL_REQUIRED_COLUMNS.issubset(df.columns):
        return df
    if "options_json" not in df.columns:
        return df

    expanded_rows = []
    for row in df.to_dict(orient="records"):
        situation_id = row.get("situation_id")
        options = _parse_json_field(row.get("options_json"), situation_id=situation_id,
                                    field_name="options_json")
        if not isinstance(options, list) or not options:
            raise ValueError(
                f"Could not expand situation_id={situation_id!r}: options_json is missing "
                "or not a non-empty list."
            )

        num_options = row.get("num_options")
        if _is_missing(num_options):
            num_options = len(options)

        for fallback_index, option in enumerate(options):
            if not isinstance(option, dict):
                raise ValueError(
                    f"Could not expand situation_id={situation_id!r}: option entry "
                    f"{fallback_index} is not an object."
                )

            option_index = option.get("option_index", fallback_index)
            if _is_missing(option_index):
                option_index = fallback_index

            expanded = dict(row)
            expanded["num_options"] = num_options
            expanded["option_index"] = int(option_index)
            expanded["option_type"] = option.get("option_type") or row.get("option_type") or "Generic"
            for output_column, option_key in OPTION_JSON_COLUMN_MAP.items():
                if _is_missing(expanded.get(output_column)) and option_key in option:
                    expanded[output_column] = _serialize_option_value(option[option_key])
            expanded_rows.append(expanded)

    expanded_df = pd.DataFrame(expanded_rows)
    preferred_columns = list(df.columns)
    for column_name in (
        "num_options",
        "option_index",
        "option_type",
        "prizes_display",
        "probs_percent",
        "EU_linear_display_3sf",
        "EU_cara_display_3sf",
        "EU_cara_alpha_0_10_display_3sf",
        "is_best_linear_display",
        "is_best_cara_display",
        "is_best_cara_alpha_0_10_display",
    ):
        if column_name in expanded_df.columns and column_name not in preferred_columns:
            preferred_columns.append(column_name)
    remaining_columns = [c for c in expanded_df.columns if c not in preferred_columns]
    return expanded_df[preferred_columns + remaining_columns]
