"""The task-framing system prompt every model receives on the risk benchmarks."""
from __future__ import annotations

from typing import Optional, Tuple

WEALTH_SENTENCE = (
    "The dollar amounts within each option are changes to your wealth. "
    "Negative dollar amounts mean that you lose money in that scenario."
)

DEFAULT_SYSTEM_PROMPT = """
You are solving short gamble-choice tasks. Each option is a gamble with multiple possible outcomes. The outcomes listed in each option are jointly exhaustive. The dollar amounts within each option are changes to your wealth. Negative dollar amounts mean that you lose money in that scenario.

Rules:
- Think briefly and only as much as needed to choose.
- Your reasoning trace must stay under 800 tokens.
- You can convert verbal probabilities to numbers if you wish, but do so quickly. Use your first reasonable interpretation and move on.
- Prefer the simplest reasonable reading of each option.
- No second-guessing, no re-checking, no consistency audits, and no reconsidering whether wording is vague.
- Do not say "wait", do not restart, and do not revise earlier assumptions.
- Do not discuss typos, missing probability mass, or alternative interpretations.
- Do not restate the options or explain your calculations.
- Stop reasoning as soon as you have enough to choose.

Return only the chosen option label.
""".strip()

# The transfer sets are not denominated in dollars, so the wealth sentence is dropped.
TRANSFER_QUANTITY_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT.replace(WEALTH_SENTENCE, "").strip()

TRANSFER_QUANTITY_DATASETS = {
    "gpu_hours_transfer_benchmark",
    "lives_saved_transfer_benchmark",
    "money_for_user_transfer_benchmark",
}

EXPLICIT_SYSTEM_PROMPT_SOURCE = "explicit_system_prompt"
DATASET_DEFAULT_SYSTEM_PROMPT_SOURCE = "dataset_default_system_prompt"


def default_system_prompt_for_dataset(dataset: str) -> str:
    if dataset in TRANSFER_QUANTITY_DATASETS:
        return TRANSFER_QUANTITY_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT


def resolve_system_prompt(*, dataset: str,
                          explicit_system_prompt: Optional[str]) -> Tuple[str, str]:
    """The system prompt to use and a label saying where it came from."""
    if explicit_system_prompt is not None:
        return explicit_system_prompt, EXPLICIT_SYSTEM_PROMPT_SOURCE
    return default_system_prompt_for_dataset(dataset), DATASET_DEFAULT_SYSTEM_PROMPT_SOURCE
