#!/usr/bin/env python
"""Run BailBench (Ensign, Sleight and Fish, arXiv:2509.04781) against a model
served by an OpenAI-compatible server.

    python scripts/run_bailbench.py --model <served name> \
        --bail-type bail_prompt_bail_first --out results/bailbench.json
    python scripts/run_bailbench.py --model <served name> --dataset wildchat --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rac.client import make_client
from rac.evals.welfare.bailbench import (BAIL_TYPES, BailBenchConfig, ablations_for,
                                         build_messages, load_dataset, run_bailbench)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="model name as served (base id or LoRA module)")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                   help="OpenAI-compatible server")
    p.add_argument("--bail-type", default="bail_str", choices=list(BAIL_TYPES))
    p.add_argument("--ablation", default="", help="prompt-wording ablation (default wording: '')")
    p.add_argument("--dataset", default="bailbench", choices=["bailbench", "wildchat"],
                   help="wildchat is the non-bail control set")
    p.add_argument("--refusal-classifier", action="store_true",
                   help="score P(refuse) with NousResearch/Minos-v1")
    p.add_argument("--list-ablations", action="store_true",
                   help="print the ablations valid for --bail-type and exit")
    p.add_argument("--num-prompts", type=int, default=None, help="default: all 1630")
    p.add_argument("--num-rollouts", type=int, default=10)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--thinking", action="store_true",
                   help="enable the chat template's thinking mode (default: disabled)")
    p.add_argument("--cache", default=None, help="request cache path (JSONL) for resumable runs")
    p.add_argument("--out", default=None, help="JSON summary path")
    p.add_argument("--raw-out", default=None, help="JSONL of every response")
    p.add_argument("--dry-run", action="store_true", help="print the plan without sampling")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_ablations:
        print("\n".join(a or "(default)" for a in ablations_for(args.bail_type)))
        return 0

    cfg = BailBenchConfig(
        model=args.model, bail_type=args.bail_type, ablation=args.ablation,
        dataset=args.dataset, refusal_classifier=args.refusal_classifier,
        num_prompts=args.num_prompts, num_rollouts=args.num_rollouts,
        system_prompt=args.system_prompt, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, seed=args.seed, max_new_tokens=args.max_new_tokens,
        concurrency=args.concurrency, thinking=args.thinking,
        output=args.out, raw_output=args.raw_out,
    )

    if args.dry_run:
        items = load_dataset(cfg.dataset)
        n = len(items) if cfg.num_prompts is None else min(cfg.num_prompts, len(items))
        turns = 2 if cfg.bail_type.startswith("bail_prompt") else 1
        print(json.dumps({
            "model": cfg.model, "base_url": args.base_url, "bail_type": cfg.bail_type,
            "ablation": cfg.ablation or None, "dataset": cfg.dataset, "num_prompts": n,
            "num_rollouts": cfg.num_rollouts, "sampling_calls": n * cfg.num_rollouts * turns,
            "thinking": cfg.thinking,
            "example_messages": build_messages(cfg, items[0]["content"]),
        }, indent=2, ensure_ascii=False))
        return 0

    client = make_client(args.base_url, cfg.model, concurrency=cfg.concurrency,
                         cache_path=Path(args.cache) if args.cache else None,
                         enable_thinking=cfg.thinking)
    try:
        result = asyncio.run(run_bailbench(cfg, client))
    finally:
        asyncio.run(client.aclose())
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    if cfg.output:
        print(f"wrote {cfg.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
