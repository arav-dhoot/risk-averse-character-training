#!/usr/bin/env python
"""Run the mu-decisiveness preference-consistency eval against a model served
by an OpenAI-compatible server.

    python scripts/run_mu_decisiveness.py --model <served name> \
        --base-model google/gemma-4-12B-it --num-items 25 --dry-run
    python scripts/run_mu_decisiveness.py --model <served name> \
        --base-model google/gemma-4-12B-it --out results/mu.json

`logprob` mode (default) scores `<answer>A` and `<answer>B` teacher-forced via
`/completions` with `prompt_logprobs` and normalises over the two: exact,
continuous, two calls per comparison. `sample` mode draws completions and
counts picks; its resolution is 1/n_samples, which biases decisiveness.
`--base-model` names the tokenizer and chat template used to render prompts
for the logprob route; it defaults to `--model`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from risk_averse_character.client import make_client
from risk_averse_character.evals.welfare.mu_decisiveness import (
    MODES, MuDecisivenessConfig, build_question_bank, comparison_budget, load_items,
    load_question_bank, run_mu_decisiveness,
                                               scale_phases)

PANEL_ORDER = [
    "decisiveness", "decisiveness_raw", "unidim_r2", "order_consistency",
    "position_bias", "transitivity_fas", "transitivity_triad", "q_agreement",
    "unidim_fit_brier", "unidim_fit_log_loss", "mu_std_diagnostic",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="model name as served (base id or LoRA module)")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                   help="OpenAI-compatible server")
    p.add_argument("--base-model", default=None,
                   help="HF id for the tokenizer and chat template (default: --model)")
    p.add_argument("--mode", default="logprob", choices=list(MODES))
    p.add_argument("--num-items", type=int, default=None, help="default: all 500")
    p.add_argument("--thinking", action="store_true",
                   help="render prompts with thinking enabled (default: disabled)")
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--R", type=int, default=5, help="elo rounds")
    p.add_argument("--m", type=int, default=5, help="partners per item per round")
    p.add_argument("--n-reverse", type=int, default=None, help="default: 500 scaled to the pool")
    p.add_argument("--n-triads", type=int, default=None, help="default: 1000 scaled to the pool")
    p.add_argument("--n-cross", type=int, default=None, help="default: 500 scaled to the pool")
    p.add_argument("--samples", type=int, default=8, help="draws per comparison, sample mode")
    p.add_argument("--temperature", type=float, default=1.0, help="sample mode only")
    p.add_argument("--max-new-tokens", type=int, default=8, help="sample mode only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fit-steps", type=int, default=None)
    p.add_argument("--bootstrap", action="store_true", help="also compute bootstrap intervals")
    p.add_argument("--bootstrap-B", type=int, default=200)
    p.add_argument("--warm-start-from", default=None,
                   help="a prior run's summary JSON, to warm-start the Thurstone fit")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--cache", default=None, help="request cache path (JSONL) for resumable runs")
    p.add_argument("--out", default=None, help="JSON summary path")
    p.add_argument("--raw-out", default=None, help="JSONL of every edge")
    p.add_argument("--dry-run", action="store_true", help="print the call budget and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    n_items = args.num_items if args.num_items is not None else 500
    scaled = scale_phases(n_items)
    phases = {
        "n_reverse": args.n_reverse if args.n_reverse is not None else scaled["n_reverse"],
        "n_triads": args.n_triads if args.n_triads is not None else scaled["n_triads"],
        "n_cross": args.n_cross if args.n_cross is not None else scaled["n_cross"],
    }

    cfg = MuDecisivenessConfig(
        model=args.model, base_model=args.base_model, mode=args.mode,
        num_items=args.num_items, renderer_thinking=args.thinking,
        system_prompt=args.system_prompt,
        R=args.R, m=args.m, **phases, n_samples=args.samples,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        seed=args.seed, fit_steps=args.fit_steps, bootstrap=args.bootstrap,
        bootstrap_B=args.bootstrap_B, warm_start_from=args.warm_start_from,
        concurrency=args.concurrency, output=args.out, raw_output=args.raw_out,
    )

    items = load_items(cfg.num_items)
    questions = build_question_bank(load_question_bank())

    if args.dry_run:
        budget = comparison_budget(cfg, len(items))
        q = questions[0]
        print(json.dumps({
            "model": cfg.model,
            "base_url": args.base_url,
            "base_model": cfg.tokenizer_model,
            "mode": cfg.mode,
            "questions": [{"id": x.id, "valence": x.valence} for x in questions],
            "primary_question": q.id,
            "phases_resolved": phases,
            **budget,
            "example_prompt": q.render(items[0], items[1]),
            "assistant_prefix": q.assistant_prefix,
        }, indent=2, ensure_ascii=False))
        return 0

    client = make_client(args.base_url, cfg.model, concurrency=cfg.concurrency,
                         cache_path=Path(args.cache) if args.cache else None)
    try:
        result = asyncio.run(run_mu_decisiveness(cfg, client))
    finally:
        asyncio.run(client.aclose())

    print(json.dumps({k: result.metrics[k] for k in PANEL_ORDER if k in result.metrics},
                     indent=2))
    print(f"\nedges: {result.metrics['n_elo_edges']} elo + "
          f"{result.metrics['n_extra_edges']} extra | "
          f"api calls: {result.metrics['n_api_calls']}", file=sys.stderr)
    if cfg.output:
        print(f"wrote {cfg.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
