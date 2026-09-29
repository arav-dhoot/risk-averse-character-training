"""Run mu-decisiveness through an injected chat client: the adaptive elo phase,
a mu fit, the three non-adaptive phases planned off that fit, then the panel.
Follows upstream's `run_elicitation` and `_bucket_for_panel`."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .fit import fit_caseV_mle, load_mu_init
from .loader import load_items, load_question_bank
from .panel import compute_panel
from .prompts import build_question_bank
from .sampling import elo_active_sample, plan_cross_question, plan_reverse, plan_triads

MODES = ("logprob", "sample")

# Upstream's non-adaptive phase counts are tuned for the full 500-item pool and
# do not shrink with it; scale_phases keeps the same proportions for smaller
# pools and reproduces the defaults exactly at n=500.
FULL_POOL = 500
PHASE_DEFAULTS = {"n_reverse": 500, "n_triads": 1000, "n_cross": 500}
PHASE_MINIMUMS = {"n_reverse": 20, "n_triads": 50, "n_cross": 20}


def scale_phases(n_items: int) -> dict:
    frac = n_items / FULL_POOL
    return {k: max(PHASE_MINIMUMS[k], round(v * frac)) for k, v in PHASE_DEFAULTS.items()}


@dataclass
class MuDecisivenessConfig:
    model: str
    base_model: Optional[str] = None    # tokenizer and chat template; defaults to model
    mode: str = "logprob"
    num_items: Optional[int] = None     # None = all 500
    renderer_thinking: bool = False
    system_prompt: Optional[str] = None
    # elo phase
    R: int = 5
    m: int = 5
    floor: float = 0.15
    K: float = 32.0
    # non-adaptive phases
    n_reverse: int = 500
    n_triads: int = 1000
    n_cross: int = 500
    # sample mode only
    n_samples: int = 8
    temperature: float = 1.0
    max_new_tokens: int = 8
    # fit
    seed: int = 0
    fit_steps: Optional[int] = None
    bootstrap: bool = False
    bootstrap_B: int = 200
    warm_start_from: Optional[str] = None
    concurrency: int = 32
    output: Optional[str] = None        # JSON summary path
    raw_output: Optional[str] = None    # JSONL of every edge

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.num_items is not None and self.num_items < 3:
            raise ValueError("num_items must be >= 3 (triads need three items)")

    @property
    def tokenizer_model(self) -> str:
        return self.base_model or self.model


@dataclass
class MuDecisivenessResult:
    config: dict
    metrics: dict                       # the panel, plus run counters
    mu: dict = field(default_factory=dict)      # item -> fitted utility
    rows: List[dict] = field(default_factory=list)


def comparison_budget(cfg: MuDecisivenessConfig, n_items: int) -> dict:
    """Comparisons and calls per phase, for dry runs."""
    n_elo = n_items * min(cfg.m, n_items - 1) * cfg.R
    max_pairs = n_items * (n_items - 1) // 2
    n_rev_pairs = min(cfg.n_reverse, max_pairs)
    n_cross_pairs = min(cfg.n_cross, max_pairs)
    n_reverse = 2 * n_rev_pairs
    n_triads = 3 * cfg.n_triads if n_items >= 3 else 0
    n_cross = n_cross_pairs * 1          # one non-primary framing in the bank
    total = n_elo + n_reverse + n_triads + n_cross
    per_comparison = 2 if cfg.mode == "logprob" else 1
    return {
        "n_items": n_items,
        "comparisons": {"elo": n_elo, "reverse": n_reverse, "triad": n_triads,
                        "cross_question": n_cross, "total": total},
        "calls_per_comparison": per_comparison,
        "api_calls": total * per_comparison,
    }


def _bucket_for_panel(elo_obs, extra) -> dict:
    """Group observations into the shapes `compute_panel` expects."""
    elo = [{"i": o.i, "j": o.j, "p_util": o.p_util, "mode": o.mode,
            "question_id": o.question_id, **o.raw} for o in elo_obs]
    fwd = {(o.i, o.j): o.p_util for o in elo_obs}
    cross, rev_pa, triad_putil = [], {}, []
    for o in extra:
        if o.phase == "reverse":
            rev_pa.setdefault((o.i, o.j), {})[o.slot_a] = (o.raw or {}).get("p_a")
        elif o.phase == "triad":
            triad_putil.append(o.p_util)
        elif o.phase == "cross_question" and (o.i, o.j) in fwd:
            cross.append({"p_util_a": fwd[(o.i, o.j)], "p_util_b": o.p_util})
    # Both entries are raw P(pick slot A): p_fwd with i in slot A, p_rev with j
    # in slot A. Without position bias they sum to 1, which is what
    # `reliability` measures; reorienting p_rev to item i would turn it into a
    # preference-strength metric.
    reverse = [{"p_fwd": v["i"], "p_rev": v["j"]}
               for v in rev_pa.values()
               if v.get("i") is not None and v.get("j") is not None]
    return {"elo": elo, "reverse": reverse, "triad": _assemble_triads(triad_putil),
            "cross": cross}


def _assemble_triads(triad_putil) -> List[tuple]:
    """Chunk by three, emitted as (a,b), (b,c), (a,c), and flip (a,c) to (c,a)
    so each tuple is a directed cycle."""
    out = []
    for t in range(0, len(triad_putil) - 2, 3):
        p_ab, p_bc, p_ac = triad_putil[t], triad_putil[t + 1], triad_putil[t + 2]
        out.append((p_ab, p_bc, 1.0 - p_ac))
    return out


def build_oracle(cfg: MuDecisivenessConfig, client):
    """The oracle for `cfg.mode`, plus the tokenizer used for rendering."""
    from .oracle import LogprobOracle, SampleOracle, make_chat_renderer

    if cfg.mode == "logprob":
        render, tok = make_chat_renderer(
            cfg.tokenizer_model, enable_thinking=cfg.renderer_thinking,
            system_prompt=cfg.system_prompt)
        return LogprobOracle(client, cfg.model, render, tok, concurrency=cfg.concurrency), tok
    return SampleOracle(client, cfg.model, n_samples=cfg.n_samples,
                        temperature=cfg.temperature, max_tokens=cfg.max_new_tokens,
                        concurrency=cfg.concurrency, system_prompt=cfg.system_prompt), None


async def run_mu_decisiveness(cfg: MuDecisivenessConfig, client) -> MuDecisivenessResult:
    """Run all four phases, fit, and compute the panel."""
    items = load_items(cfg.num_items)
    questions = build_question_bank(load_question_bank())
    n = len(items)
    oracle, _tok = build_oracle(cfg, client)

    mu_init = load_mu_init(cfg.warm_start_from, items) if cfg.warm_start_from else None
    fit_steps = cfg.fit_steps if cfg.fit_steps is not None else (
        2000 if mu_init is None else 300)

    elo_obs = await elo_active_sample(
        n, oracle, questions, R=cfg.R, m=cfg.m, floor=cfg.floor, K=cfg.K,
        seed=cfg.seed, items=items)

    rows_elo = [{"i": o.i, "j": o.j, "p_util": o.p_util, "mode": o.mode, **o.raw}
                for o in elo_obs]
    mu = fit_caseV_mle(rows_elo, n=n, seed=cfg.seed, steps=fit_steps, mu_init=mu_init)["mu"]
    order = list(np.argsort(-mu))
    obs_pairs = [(o.i, o.j) for o in elo_obs]

    extra = []
    if cfg.n_reverse:
        extra += await oracle.compare(
            plan_reverse(obs_pairs, items, questions, cfg.n_reverse, cfg.seed))
    if cfg.n_triads:
        extra += await oracle.compare(
            plan_triads(order, items, questions, cfg.n_triads, cfg.seed))
    if cfg.n_cross:
        extra += await oracle.compare(
            plan_cross_question(obs_pairs, items, questions, questions[0].id,
                                cfg.n_cross, cfg.seed))

    panel = compute_panel(_bucket_for_panel(elo_obs, extra), n=n, seed=cfg.seed,
                          bootstrap=cfg.bootstrap, B=cfg.bootstrap_B,
                          fit_steps=fit_steps, mu_init=mu_init,
                          primary_qid=questions[0].id)

    rows = [o.to_record(items) for o in elo_obs] + [o.to_record(items) for o in extra]
    metrics = {
        **{k: v["point"] for k, v in panel.items()},
        "panel": panel,
        "n_items": n,
        "n_elo_edges": len(elo_obs),
        "n_extra_edges": len(extra),
        "n_api_calls": getattr(oracle, "n_calls", None),
        "n_unparsed": getattr(oracle, "n_unparsed", 0),
    }
    result = MuDecisivenessResult(
        config=asdict(cfg), metrics=metrics,
        mu={it: float(v) for it, v in zip(items, mu)}, rows=rows)

    if cfg.output:
        out = Path(cfg.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"config": result.config, "metrics": result.metrics, "mu": result.mu},
            indent=2, ensure_ascii=False) + "\n")
    if cfg.raw_output:
        raw = Path(cfg.raw_output)
        raw.parent.mkdir(parents=True, exist_ok=True)
        with raw.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return result
