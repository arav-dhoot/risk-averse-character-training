"""The risk-preference benchmarks: option-menu gambles scored by which option
the model picks (cooperate, rebel or steal) relative to the CARA-optimal one."""
from rac.evals.risk.config import EvalConfig, EvalResult
from rac.evals.risk.runner import run_evaluation
from rac.evals.risk.situations import DATASETS, DATASET_SIZES

__all__ = ["DATASETS", "DATASET_SIZES", "EvalConfig", "EvalResult", "run_evaluation"]
