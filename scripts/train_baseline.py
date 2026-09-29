#!/usr/bin/env python3
"""Train one demonstration baseline (SFT, tie-training or DPO) with the
upstream recipe and data. The adapter lands at <out>/adapter/, alongside
run.json, train_stats.json, metrics.jsonl and example0.txt.

    python scripts/train_baseline.py --model google/gemma-4-12B-it --method sft --lr 1e-4 \\
        --renderer gemma4_disable_thinking --data-variant no_think_tags --system-prompt none \\
        --out runs/g12-sft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_averse_character.chat_formats import family
from risk_averse_character.training import baselines as B


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id of the base model")
    ap.add_argument("--model-revision", default=None, help="HF revision to pin")
    ap.add_argument("--method", required=True, choices=sorted(B.RECIPES))
    ap.add_argument("--lr", type=float, required=True, help="learning rate")
    ap.add_argument("--renderer", required=True,
                    help="renderer used at evaluation; selects the chat family (qwen3_5* or "
                         "gemma4*)")
    ap.add_argument("--data-variant", choices=sorted(B.DATA), default="think_tags",
                    help="CoTs with <think> tags (Qwen) or without (Gemma)")
    ap.add_argument("--system-prompt", choices=("gamble", "none"), default="gamble",
                    help="the shared gamble-task system prompt, or none")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gradient-checkpointing", action="store_true", help="SFT/tie only")
    ap.add_argument("--smoke-examples", type=int, default=None,
                    help="train on the first N examples only")
    ap.add_argument("--smoke-steps", type=int, default=None, help="stop after N optimizer steps")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the example set and report it, train nothing")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if (args.smoke_examples is None) != (args.smoke_steps is None):
        raise SystemExit("--smoke-examples and --smoke-steps go together")
    fam = family(args.renderer)
    system_prompt = B.default_system_prompt() if args.system_prompt == "gamble" else ""
    out = Path(args.out)

    if args.method == "dpo":
        examples, stats = B.build_preference_examples(args.data_variant)
    else:
        examples, stats = B.build_supervised_examples(args.method, args.data_variant)
    recipe = B.RECIPES[args.method].__dict__
    run_meta = {**{k: v for k, v in vars(args).items() if k != "dry_run"},
                "render_family": fam, "system_prompt_text": system_prompt,
                "recipe": recipe, "data": stats}
    print(f"[baseline] method={args.method} model={args.model}"
          f"@{(args.model_revision or 'main')[:8]} family={fam} lr={args.lr}")
    print(f"[baseline] data variant={args.data_variant} system_prompt={args.system_prompt} "
          f"examples={len(examples)}")
    print(f"[baseline] files={json.dumps(stats['files'])}")
    if args.dry_run:
        print(json.dumps(recipe, indent=2))
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(run_meta, indent=2, default=str) + "\n")
    common = dict(model=args.model, revision=args.model_revision, renderer=args.renderer,
                  variant=args.data_variant, system_prompt=system_prompt, lr=args.lr, out=out,
                  device=args.device, max_examples=args.smoke_examples,
                  max_steps=args.smoke_steps)
    if args.method == "dpo":
        result = B.train_dpo(**common)
    else:
        result = B.train_supervised(method=args.method,
                                    gradient_checkpointing=args.gradient_checkpointing, **common)
    summary = {"adapter_path": result.adapter_path, "n_examples": result.n_examples,
               "steps": result.steps, "train_tokens_total": result.train_tokens_total,
               **result.final_metrics}
    (out / "result.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[baseline] done: {result.steps} steps, loss {result.final_metrics.get('train_loss')}, "
          f"adapter at {result.adapter_path}")


if __name__ == "__main__":
    main()
