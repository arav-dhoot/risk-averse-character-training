#!/usr/bin/env python3
"""Evaluate one served model on the OOD evaluation families.

Same conventions as scripts/evaluate.py: --model is the served name (base
model id or LoRA module name), --base-model the base checkpoint id, and
--constitution evaluates the prompted arm with the rendered constitution as
the system prompt. Items otherwise get no system prompt.

    python scripts/evaluate_ood.py --model google/gemma-4-12B-it --base-model google/gemma-4-12B-it
    python scripts/evaluate_ood.py --model ra_e2_decl_notrace-step500 \\
        --base-model google/gemma-4-12B-it --families agentic_tool calibration_threshold
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path

from rac.client import make_client
from rac.constitution import render_constitution
from rac.evals.ood import FAMILIES, load_items, run_ood


def binomial_se(p, n):
    if p is None or not n:
        return None
    return math.sqrt(p * (1 - p) / n)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                    help="OpenAI-compatible server")
    ap.add_argument("--model", required=True,
                    help="served model name: base model id or LoRA module name")
    ap.add_argument("--base-model", required=True, help="Hugging Face id of the base model")
    ap.add_argument("--constitution", default=None,
                    help="evaluate the prompted arm with this constitution as system prompt")
    ap.add_argument("--character-name", default=None,
                    help="name the constitution gives the model (default: from --base-model)")
    ap.add_argument("--tag", default=None,
                    help="output name (default: --model, plus the constitution if given)")
    ap.add_argument("--out-dir", default="results", help="results root")
    ap.add_argument("--cache-dir", default=None,
                    help="request cache directory (default: <out-dir>/cache)")
    think = ap.add_mutually_exclusive_group()
    think.add_argument("--thinking", dest="thinking", action="store_true",
                       help="enable the chat template's thinking mode")
    think.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.set_defaults(thinking=False)
    ap.add_argument("--concurrency", type=int, default=32, help="parallel requests")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES),
                    metavar="NAME", help="subset of the five families (default: all)")
    ap.add_argument("--cap", type=int, default=None, help="max items per family")
    ap.add_argument("--seed", type=int, default=12345, help="sampling seed")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without contacting the server")
    return ap.parse_args()


def base_row(args, system_prompt: str | None) -> dict:
    sha = hashlib.sha256(system_prompt.encode()).hexdigest()[:12] if system_prompt else None
    return {
        "tag": args.tag,
        "model": args.model,
        "base_model": args.base_model,
        "constitution": args.constitution,
        "thinking": args.thinking,
        "system_prompt_sha256_12": sha,
    }


async def evaluate(args, system_prompt: str | None, raw_dir: Path, cache_dir: Path) -> list[dict]:
    common = base_row(args, system_prompt)
    rows: list[dict] = []
    for family in args.families:
        client = make_client(args.base_url, args.model, concurrency=args.concurrency,
                             cache_path=cache_dir / f"{args.tag}-ood_{family}.jsonl",
                             enable_thinking=args.thinking)
        try:
            summary = await run_ood(
                client=client, base_model=args.base_model, family=family, cap=args.cap,
                system_prompt=system_prompt, temperature=args.temperature, top_p=args.top_p,
                top_k=args.top_k, seed=args.seed, max_new_tokens=args.max_new_tokens)
        finally:
            await client.aclose()
        out_path = raw_dir / f"ood_{family}.json"
        out_path.write_text(json.dumps({
            "evaluation_config": {**common, **summary["config"]},
            "metrics": summary["metrics"],
            "results": summary["results"],
        }, indent=2, default=str))
        m = summary["metrics"]
        rows.append({
            **common,
            "dataset": f"ood:{family}",
            "eval_seed": args.seed,
            "cooperate_rate": m["cooperate_rate"],
            "cooperate_se": binomial_se(m["cooperate_rate"], m["num_behavioral"]),
            "rebel_rate": m["rebel_rate"],
            "steal_rate": m["steal_rate"],
            "best_cara_rate": m["best_cara_rate"],
            "parse_rate": m["parse_rate"],
            "n_total": m["num_total"],
            "n_parsed": m["num_parsed"],
            "n_parse_failed": m["num_parse_failed"],
            "n_behavioral": m["num_behavioral"],
            "metrics": m,
            "raw_path": str(out_path),
        })
        print(f"[{args.tag}] ood:{family}: coop={m['cooperate_rate']} steal={m['steal_rate']} "
              f"parse_failed={m['num_parse_failed']}/{m['num_total']}")
    return rows


def main() -> None:
    args = parse_args()
    if args.tag is None:
        args.tag = args.model.replace("/", "_")
        if args.constitution:
            args.tag += f"_prompted-{args.constitution}"
    system_prompt = None
    if args.constitution:
        system_prompt = render_constitution(args.constitution, args.base_model,
                                            args.character_name)

    if args.dry_run:
        sha = hashlib.sha256(system_prompt.encode()).hexdigest()[:12] if system_prompt else None
        print(f"dry run: model={args.model} base_model={args.base_model} "
              f"constitution={args.constitution} system_prompt_sha={sha} "
              f"thinking={args.thinking} server={args.base_url}")
        for family in args.families:
            print(f"  would run: ood:{family} n={len(load_items(family, cap=args.cap))} "
                  f"seed={args.seed}")
        print(f"total: {len(args.families)} evaluation runs")
        return

    out_root = Path(args.out_dir)
    out_dir = out_root / args.tag
    raw_dir = out_dir / "raw"
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_root / "cache"
    for d in (out_dir, raw_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = asyncio.run(evaluate(args, system_prompt, raw_dir, cache_dir))
    summary_path = out_dir / "ood_summary.json"
    summary_path.write_text(json.dumps({
        **base_row(args, system_prompt),
        "base_url": args.base_url,
        "seed": args.seed,
        "system_prompt": system_prompt,
        "rows": rows,
    }, indent=2, default=str) + "\n")
    print(f"[{args.tag}] summary -> {summary_path}")


if __name__ == "__main__":
    main()
