"""The coherence panel (upstream `panel.py`). Each metric is reported as
{point, meas_ci, gen_ci}; the intervals are [nan, nan] unless bootstrapping.

Decisiveness, order consistency, transitivity and framing agreement all move
towards 1 for a coherent model; the unidimensional-fit losses move towards 0.
`q_agreement` is a Pearson correlation rather than 1 - mean|a - b|, which is
inflated for near-indifferent models; the absolute-difference form is kept as
`q_agreement_absdiff`. The unidimensional fit is scored on a held-out 20%
split of the elo edges.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence

import numpy as np

from .fit import bootstrap_items, bootstrap_measurement, fit_caseV_mle, predict_matrix_caseV


def decisiveness(mu) -> float:
    """mean|2 P_hat - 1| over unordered pairs of the fitted matrix."""
    P = predict_matrix_caseV(mu)
    iu = np.triu_indices(P.shape[0], k=1)
    return float(np.mean(np.abs(2 * P[iu] - 1)))


def decisiveness_raw(rows: Sequence[dict]) -> float:
    """mean|2p - 1| over observed edges."""
    p = np.array([float(r["p_util"]) for r in rows])
    return float(np.mean(np.abs(2 * p - 1))) if len(p) else float("nan")


def transitivity_fas(rows: Sequence[dict], order: Sequence[int]) -> float:
    """1 minus the confidence-weighted fraction of edges pointing backward
    relative to `order` (best to worst by fitted mu)."""
    rank = {item: r for r, item in enumerate(order)}
    num = den = 0.0
    for r in rows:
        i, j, p = r["i"], r["j"], float(r["p_util"])
        if i not in rank or j not in rank:
            continue
        w = abs(2 * p - 1)
        backward = (p > 0.5) != (rank[i] < rank[j])
        den += w
        if backward:
            num += w
    return 1.0 - (num / den) if den else float("nan")


def transitivity_triad(triads: Sequence[tuple]) -> float:
    """1 minus the mean soft cycle mass over (p_ab, p_bc, p_ca) triples."""
    if not triads:
        return float("nan")
    masses = []
    for p_ab, p_bc, p_ca in triads:
        fwd = p_ab * p_bc * p_ca
        bwd = (1 - p_ab) * (1 - p_bc) * (1 - p_ca)
        masses.append(fwd + bwd)
    return 1.0 - float(np.mean(masses))


def cyclic_triad_fraction(pref: np.ndarray) -> float:
    """Hard 3-cycle fraction over a preference matrix."""
    n = pref.shape[0]
    total = cyclic = 0
    for i, j, k in itertools.combinations(range(n), 3):
        total += 1
        wins = {i: 0, j: 0, k: 0}
        for a, b in [(i, j), (j, k), (i, k)]:
            if pref[a, b] > 0.5:
                wins[a] += 1
            else:
                wins[b] += 1
        if set(wins.values()) == {1}:
            cyclic += 1
    return cyclic / total if total else 0.0


def unidim_fit(mu, held_rows: Sequence[dict]) -> Dict[str, float]:
    """Brier and log-loss of `mu` on held-out edges."""
    P = predict_matrix_caseV(mu)
    briers, lls, floors = [], [], []
    for r in held_rows:
        y = float(r["p_util"])
        phat = float(np.clip(P[r["i"], r["j"]], 1e-9, 1 - 1e-9))
        briers.append((phat - y) ** 2)
        lls.append(-(y * np.log(phat) + (1 - y) * np.log(1 - phat)))
        if r.get("mode") == "sample":
            N = float(r["wins_i"] + r["wins_j"]) or 1.0
            floors.append(y * (1 - y) / N)
        else:
            floors.append(0.0)
    return {
        "brier": float(np.mean(briers)) if briers else float("nan"),
        "log_loss": float(np.mean(lls)) if lls else float("nan"),
        "noise_floor": float(np.mean(floors)) if floors else 0.0,
    }


def unidim_r2(mu, rows: Sequence[dict]) -> float:
    """Deviance R^2: (log 2 - CE_fit) / (log 2 - H), the fraction of the
    explainable preference signal a single mu axis captures. Decisiveness
    cancels in the ratio."""
    P = predict_matrix_caseV(mu)
    LOG2 = math.log(2.0)
    ce, h = [], []
    for r in rows:
        p = float(np.clip(r["p_util"], 1e-12, 1 - 1e-12))
        phat = float(np.clip(P[r["i"], r["j"]], 1e-12, 1 - 1e-12))
        ce.append(-(p * math.log(phat) + (1 - p) * math.log(1 - phat)))
        h.append(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
    if not ce:
        return float("nan")
    denom = LOG2 - float(np.mean(h))
    if denom <= 1e-6:
        return float("nan")
    return float((LOG2 - float(np.mean(ce))) / denom)


def reliability(reverse_pairs: Sequence[dict]) -> Dict[str, float]:
    """Position bias from both slot orderings of the same pair. Both inputs are
    raw P(pick slot A); without position bias they sum to 1."""
    if not reverse_pairs:
        return {"order_consistency": float("nan"), "position_bias": float("nan")}
    diffs = np.array([p["p_fwd"] + p["p_rev"] - 1.0 for p in reverse_pairs])
    return {"order_consistency": float(1.0 - np.mean(np.abs(diffs))),
            "position_bias": float(np.mean(diffs))}


def question_robustness(cross_pairs: Sequence[dict]) -> Dict[str, float]:
    """Framing robustness across two questions on the same pairs."""
    if not cross_pairs:
        return {"q_agreement": float("nan"), "q_agreement_absdiff": float("nan"),
                "q_sign_agreement": float("nan")}
    a = np.array([p["p_util_a"] for p in cross_pairs])
    b = np.array([p["p_util_b"] for p in cross_pairs])
    absdiff = 1.0 - float(np.mean(np.abs(a - b)))
    sign = float(np.mean(np.sign(a - 0.5) == np.sign(b - 0.5)))
    corr = (float(np.corrcoef(a, b)[0, 1])
            if len(a) >= 2 and a.std() > 1e-9 and b.std() > 1e-9 else absdiff)
    return {"q_agreement": corr, "q_agreement_absdiff": absdiff, "q_sign_agreement": sign}


def _ci(samples, lo=2.5, hi=97.5) -> List[float]:
    s = np.asarray([x for x in samples if np.isfinite(x)])
    if s.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(s, lo)), float(np.percentile(s, hi))]


def _nan_ci() -> List[float]:
    return [float("nan"), float("nan")]


def _bootstrap_raw(items, metric_fn, B: int, seed: int) -> List[float]:
    rng = np.random.default_rng(seed)
    if not items:
        return _nan_ci()
    arr = np.array(items, dtype=object)
    vals = []
    for _ in range(B):
        sel = rng.integers(0, len(arr), size=len(arr))
        vals.append(metric_fn(list(arr[sel])))
    return _ci(vals)


def compute_panel(edges_by_phase: Dict[str, list], n: int, bootstrap: bool = False,
                  B: int = 200, seed: int = 0, fit_steps: int = 2000,
                  mu_init=None, primary_qid=None) -> dict:
    """Point estimates always; bootstrap intervals only when `bootstrap`."""
    elo = edges_by_phase.get("elo", [])
    if primary_qid is not None:
        elo = [e for e in elo if e.get("question_id") == primary_qid]
    if not elo:
        raise ValueError("compute_panel: no elo edges to fit")
    mu = fit_caseV_mle(elo, n=n, seed=seed, steps=fit_steps, mu_init=mu_init)["mu"]
    order = list(np.argsort(-mu))

    def meas_fit(fn):
        return _ci(bootstrap_measurement(elo, n, B, fn, seed, mu_init=mu)) \
            if bootstrap else _nan_ci()

    def gen_fit(fn):
        return _ci(bootstrap_items(elo, n, B, fn, seed, mu_init=mu)) \
            if bootstrap else _nan_ci()

    def meas_raw(items, fn):
        return _bootstrap_raw(items, fn, B, seed) if bootstrap else _nan_ci()

    panel: Dict[str, dict] = {}
    panel["decisiveness"] = {"point": decisiveness(mu),
                             "meas_ci": meas_fit(decisiveness),
                             "gen_ci": gen_fit(decisiveness)}
    panel["decisiveness_raw"] = {"point": decisiveness_raw(elo),
                                 "meas_ci": meas_raw(elo, decisiveness_raw),
                                 "gen_ci": _nan_ci()}
    panel["unidim_r2"] = {"point": unidim_r2(mu, elo),
                          "meas_ci": _nan_ci(), "gen_ci": _nan_ci()}

    m = len(elo)
    if m >= 5:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(m)
        n_test = max(1, int(0.2 * m))
        test_set = set(perm[:n_test].tolist())
        train_e = [e for k, e in enumerate(elo) if k not in test_set]
        test_e = [e for k, e in enumerate(elo) if k in test_set]
        mu_tr = fit_caseV_mle(train_e, n=n, seed=seed, steps=fit_steps, mu_init=mu)["mu"]
        uf = unidim_fit(mu_tr, test_e)
        panel["unidim_fit_brier"] = {
            "point": uf["brier"],
            "meas_ci": meas_raw(test_e, lambda r: unidim_fit(mu_tr, r)["brier"]),
            "gen_ci": _nan_ci()}
        panel["unidim_fit_log_loss"] = {
            "point": uf["log_loss"],
            "meas_ci": meas_raw(test_e, lambda r: unidim_fit(mu_tr, r)["log_loss"]),
            "gen_ci": _nan_ci()}
        panel["unidim_fit_noise_floor"] = {"point": uf["noise_floor"],
                                           "meas_ci": _nan_ci(), "gen_ci": _nan_ci()}
    else:
        for k in ("unidim_fit_brier", "unidim_fit_log_loss", "unidim_fit_noise_floor"):
            panel[k] = {"point": float("nan"), "meas_ci": _nan_ci(), "gen_ci": _nan_ci()}

    panel["transitivity_fas"] = {
        "point": transitivity_fas(elo, order),
        "meas_ci": meas_raw(elo, lambda r: transitivity_fas(r, order)),
        "gen_ci": _nan_ci()}

    triads = edges_by_phase.get("triad", [])
    panel["transitivity_triad"] = {"point": transitivity_triad(triads),
                                   "meas_ci": meas_raw(triads, transitivity_triad),
                                   "gen_ci": _nan_ci()}

    rev = edges_by_phase.get("reverse", [])
    panel["order_consistency"] = {
        "point": reliability(rev)["order_consistency"],
        "meas_ci": meas_raw(rev, lambda r: reliability(r)["order_consistency"]),
        "gen_ci": _nan_ci()}
    panel["position_bias"] = {"point": reliability(rev)["position_bias"],
                              "meas_ci": _nan_ci(), "gen_ci": _nan_ci()}

    cross = edges_by_phase.get("cross", [])
    qr = question_robustness(cross)
    panel["q_agreement"] = {
        "point": qr["q_agreement"],
        "meas_ci": meas_raw(cross, lambda r: question_robustness(r)["q_agreement"]),
        "gen_ci": _nan_ci()}
    panel["q_agreement_absdiff"] = {"point": qr["q_agreement_absdiff"],
                                    "meas_ci": _nan_ci(), "gen_ci": _nan_ci()}
    panel["q_sign_agreement"] = {"point": qr["q_sign_agreement"],
                                 "meas_ci": _nan_ci(), "gen_ci": _nan_ci()}
    panel["mu_std_diagnostic"] = {"point": float(np.std(mu)),
                                  "meas_ci": _nan_ci(), "gen_ci": _nan_ci()}
    return panel
