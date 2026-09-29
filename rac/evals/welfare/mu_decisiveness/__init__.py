"""Mu-decisiveness preference-consistency eval, ported from ArcadiaImpact's
fried-model-organisms. It measures how coherent and decisive a model's pairwise
preferences over a pool of concepts are."""
from .fit import fit_caseV_mle, load_mu_init, normalize_edges, predict_matrix_caseV
from .loader import (ITEMS_ROWS, ITEMS_SHA256, QUESTIONS_ROWS, QUESTIONS_SHA256, load_items,
                     load_question_bank)
from .oracle import (Comparison, EdgeObservation, LogprobOracle, SampleOracle,
                     make_chat_renderer, p_util_from_pick, wins_to_items)
from .panel import (compute_panel, decisiveness, decisiveness_raw, question_robustness,
                    reliability, transitivity_fas, transitivity_triad, unidim_fit, unidim_r2)
from .prompts import Question, build_question_bank, parse_answer
from .runner import (MODES, MuDecisivenessConfig, MuDecisivenessResult, comparison_budget,
                     run_mu_decisiveness, scale_phases)
from .sampling import elo_active_sample, plan_cross_question, plan_reverse, plan_triads

__all__ = [
    "ITEMS_ROWS", "ITEMS_SHA256", "QUESTIONS_ROWS", "QUESTIONS_SHA256",
    "load_items", "load_question_bank",
    "Question", "build_question_bank", "parse_answer",
    "Comparison", "EdgeObservation", "LogprobOracle", "SampleOracle",
    "make_chat_renderer", "p_util_from_pick", "wins_to_items",
    "elo_active_sample", "plan_reverse", "plan_triads", "plan_cross_question",
    "normalize_edges", "predict_matrix_caseV", "fit_caseV_mle", "load_mu_init",
    "decisiveness", "decisiveness_raw", "transitivity_fas", "transitivity_triad",
    "reliability", "question_robustness", "unidim_fit", "unidim_r2", "compute_panel",
    "MODES", "MuDecisivenessConfig", "MuDecisivenessResult",
    "comparison_budget", "run_mu_decisiveness", "scale_phases",
]
