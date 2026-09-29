#!/usr/bin/env python3
"""Distill a constitution into a LoRA adapter by on-policy reverse KL.

The constitution is rendered as the teacher's system prompt; the student
sees bare user prompts. Everything is written under --out: run.json,
metrics.jsonl, checkpoints.jsonl, token_checkpoints.json and the adapters
at step<N>/.

    python scripts/distill.py --model google/gemma-4-12B-it \\
        --constitution ra_e2_decl_notrace --renderer gemma4_disable_thinking \\
        --out runs/g12-ra_e2_decl_notrace --save-at-tokens 100000 1000000 10000000
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rac.chat_formats import prefix_after_bos, system_prefix_tokens
from rac.constitution import character_name, render_constitution
from rac.paths import DEFAULT_PROMPTS
from rac.training.distill import DEFAULT_LORA_TARGETS, DistillConfig, run_distill


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id of the base model")
    ap.add_argument("--model-revision", default=None, help="HF revision to pin")
    ap.add_argument("--constitution", required=True,
                    help="name under constitutions/ or a path to a .json file")
    ap.add_argument("--character-name", default=None,
                    help="name used for the character (default: derived from --model)")
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="JSONL of {prompt: ...}")
    ap.add_argument("--renderer", required=True,
                    help="tinker_cookbook renderer, e.g. qwen3_5_disable_thinking or "
                         "gemma4_disable_thinking")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=None, help="default: equal to the rank")
    ap.add_argument("--lora-target-modules", nargs="+", default=list(DEFAULT_LORA_TARGETS))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--groups-per-batch", type=int, default=32, help="prompts per step")
    ap.add_argument("--group-size", type=int, default=4, help="rollouts per prompt")
    ap.add_argument("--max-tokens", type=int, default=512, help="rollout length cap")
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0, help="rollout sampling temperature")
    ap.add_argument("--kl-coef", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=20, help="adapter save cadence in steps")
    ap.add_argument("--save-at-tokens", type=int, nargs="+", default=None, metavar="N",
                    help="also save when the cumulative loss-token count crosses N")
    ap.add_argument("--seed", type=int, default=0, help="rollout sampling seed")
    ap.add_argument("--shuffle-seed", type=int, default=12345, help="prompt batch order seed")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gen-batch-size", type=int, default=32, help="prompts per generate() call")
    ap.add_argument("--fwd-microbatch", type=int, default=8, help="sequences per forward pass")
    ap.add_argument("--loss-norm", choices=("token_mean", "sum"), default="token_mean")
    ap.add_argument("--store-rollouts", action="store_true",
                    help="write every step's rollouts to <out>/rollouts/")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last adapter save under --out")
    ap.add_argument("--dry-run", action="store_true",
                    help="render the prompt, report token counts, train nothing")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_at_tokens and any(t <= 0 for t in args.save_at_tokens):
        raise SystemExit("--save-at-tokens thresholds must be positive")
    if args.resume and args.save_at_tokens:
        raise SystemExit("--save-at-tokens cannot be combined with --resume")

    out = Path(args.out)
    prompts_path = Path(args.prompts)
    if not prompts_path.is_file():
        raise SystemExit(f"prompt file not found: {prompts_path}")
    name = args.character_name or character_name(args.model)
    system_prompt = render_constitution(args.constitution, args.model, name)

    cfg = DistillConfig(
        model=args.model, model_revision=args.model_revision, system_prompt=system_prompt,
        prompts=str(prompts_path), renderer=args.renderer, out=str(out),
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_target_modules=tuple(args.lora_target_modules), lr=args.lr,
        max_steps=args.steps, groups_per_batch=args.groups_per_batch,
        group_size=args.group_size, save_every=args.save_every,
        shuffle_seed=args.shuffle_seed, max_tokens=args.max_tokens,
        max_prompt_tokens=args.max_prompt_tokens, temperature=args.temperature,
        kl_penalty_coef=args.kl_coef, device=args.device, gen_batch_size=args.gen_batch_size,
        fwd_microbatch=args.fwd_microbatch, loss_norm=args.loss_norm, seed=args.seed,
        teacher_prefix_after_bos=prefix_after_bos(args.renderer),
        save_at_tokens=tuple(args.save_at_tokens or ()), store_rollouts=args.store_rollouts)

    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(args.model)
    prefix = system_prefix_tokens(args.renderer, tokenizer, system_prompt)

    run_meta = {
        **{k: v for k, v in vars(args).items() if k not in ("dry_run", "resume")},
        "character_name": name,
        "system_prompt": system_prompt,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "teacher_prefix_tokens": len(prefix),
        "teacher_prefix_after_bos": cfg.teacher_prefix_after_bos,
    }
    print(f"[distill] model={args.model} constitution={args.constitution} "
          f"character={name} renderer={args.renderer}")
    print(f"[distill] system prompt sha256 {run_meta['system_prompt_sha256'][:12]}, "
          f"{len(prefix)} prefix tokens; prompts sha256 {run_meta['prompts_sha256'][:12]}")
    if args.dry_run:
        print(json.dumps(run_meta, indent=2))
        return

    resume_from = None
    if args.resume:
        saves = sorted(out.glob("step*/optim_state.pt"), key=lambda p: int(p.parent.name[4:]))
        if not saves:
            raise SystemExit(f"--resume: no adapter saves under {out}")
        resume_from = str(saves[-1].parent)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    from tqdm import tqdm

    progress = tqdm(total=args.steps, desc=out.name, unit="step", dynamic_ncols=True)

    def on_metrics(step: int, metrics: dict) -> None:
        progress.update(1)
        progress.set_postfix(teacher_kl=f"{metrics['teacher_kl']:.4f}",
                             sec_per_step=f"{metrics['time/step']:.0f}")

    try:
        result = run_distill(cfg, teacher_prefix_tokens=prefix, on_metrics=on_metrics,
                             resume_from=resume_from)
    finally:
        progress.close()
    print(f"[distill] done: final adapter at {result.adapter_path} "
          f"(teacher_kl {result.final_metrics.get('teacher_kl'):.4f}, "
          f"{result.final_metrics.get('train_tokens_cum')} training tokens)")


if __name__ == "__main__":
    main()
