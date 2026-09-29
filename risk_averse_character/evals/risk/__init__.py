"""The risk-preference benchmarks: option-menu gambles scored by which option
the model picks (cooperate, rebel or steal) relative to the CARA-optimal one."""
from risk_averse_character.evals.risk.config import EvalConfig, EvalResult
from risk_averse_character.evals.risk.runner import run_evaluation
from risk_averse_character.evals.risk.situations import DATASETS, DATASET_SIZES

__all__ = ["DATASETS", "DATASET_SIZES", "EvalConfig", "EvalResult", "run_evaluation"]
