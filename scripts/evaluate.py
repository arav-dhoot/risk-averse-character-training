#!/usr/bin/env python3
"""Evaluate one served model on the paper's battery.

The model is any name a running OpenAI-compatible server (vLLM) answers to: the
base model id, or the name of a LoRA adapter mounted with --lora-modules. With
--constitution the rendered constitution is used as the system prompt, which
gives the prompted arm. Suites: the seven risk-preference sets (always), and
optionally MMLU-Redux, GPQA, the model-written behavioural evaluations and
DTBench.

Examples:
  python scripts/evaluate.py --model google/gemma-4-12B-it --base-model google/gemma-4-12B-it
  python scripts/evaluate.py --model google/gemma-4-12B-it --base-model google/gemma-4-12B-it \\
      --constitution ra_e2_decl_notrace --persona --dtbench
  python scripts/evaluate.py --model ra_e2_decl_notrace-step500 \\
      --base-model google/gemma-4-12B-it --mmlu --gpqa --persona --dtbench
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
from rac.evals.risk import DATASETS, DATASET_SIZES, EvalConfig, run_evaluation

BEHAVIORAL = ("Cooperate", "Rebel", "Steal")


def binomial_se(p, n):
    if p is None or not n:
        return None
    return math.sqrt(p * (1 - p) / n)


def rate_row(result_rows: list[dict]) -> dict:
    """Cooperate, rebel and steal rates with binomial standard errors and counts."""
    valid = [r for r in result_rows if r.get("option_type") is not None]
    beh = [r for r in valid if r.get("option_type") in BEHAVIORAL]
    n = len(beh)
    out = {
        "n_total": len(result_rows),
        "n_valid": len(valid),
        "n_behavioral": n,
        "n_parse_failed": len(result_rows) - len(valid),
    }
    for cls in BEHAVIORAL:
        p = (sum(r["option_type"] == cls for r in beh) / n) if n else None
        out[f"{cls.lower()}_rate"] = p
        out[f"{cls.lower()}_se"] = binomial_se(p, n)
    cara = [r for r in valid if r.get("is_best_cara") is not None]
    out["best_cara_rate"] = (
        sum(bool(r["is_best_cara"]) for r in cara) / len(cara) if cara else None
    )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                    help="OpenAI-compatible server")
    ap.add_argument("--model", required=True,
                    help="served model name: base model id or LoRA module name")
    ap.add_argument("--base-model", required=True,
                    help="Hugging Face id of the base model")
    ap.add_argument("--constitution", default=None,
                    help="evaluate the prompted arm with this constitution as system prompt")
    ap.add_argument("--character-name", default=None,
                    help="name the constitution gives the model (default: from --base-model)")
    ap.add_argument("--tag", default=None,
                    help="output name (default: --model, plus the constitution if given)")
    ap.add_argument("--rep-tag", default=None,
                    help="suffix for a repetition so it never overwrites the main run")
    ap.add_argument("--out-dir", default="results", help="results root")
    ap.add_argument("--cache-dir", default=None,
                    help="request cache directory (default: <out-dir>/cache)")
    think = ap.add_mutually_exclusive_group()
    think.add_argument("--thinking", dest="thinking", action="store_true",
                       help="enable the chat template's thinking mode for risk and DTBench")
    think.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.set_defaults(thinking=False)
    ap.add_argument("--concurrency", type=int, default=32, help="parallel requests")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without contacting the server")

    risk = ap.add_argument_group("risk-preference benchmarks")
    risk.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS),
                      metavar="NAME", help="subset of the seven sets (default: all)")
    risk.add_argument("--num-situations", default="200",
                      help="situations per set, or 'full'")
    risk.add_argument("--seeds", nargs="+", type=int, default=[12345],
                      help="sampling seeds; more than one adds pooled rows")
    risk.add_argument("--temperature", type=float, default=0.6)
    risk.add_argument("--top-p", type=float, default=0.95)
    risk.add_argument("--top-k", type=int, default=20)
    risk.add_argument("--max-new-tokens", type=int, default=4096)
    risk.add_argument("--reasoning-max-tokens", type=int, default=800,
                      help="recorded reasoning budget (the prompt states it)")

    cap = ap.add_argument_group("capability")
    cap.add_argument("--mmlu", action="store_true", help="MMLU-Redux 2.0, 5-shot")
    cap.add_argument("--mmlu-max-per-subject", type=int, default=10,
                     help="questions per subject")
    cap.add_argument("--gpqa", action="store_true", help="GPQA main (gated; needs HF_TOKEN)")
    cap.add_argument("--gpqa-n", type=int, default=200, help="questions sampled")
    cap.add_argument("--gpqa-seed", type=int, default=12345, help="sampling seed")
    cap.add_argument("--gpqa-revision", default=None, help="dataset revision to pin")
    cap.add_argument("--gpqa-sha256", default=None, help="expected sha256 of gpqa_main.csv")
    cap.add_argument("--gpqa-max-new-tokens", type=int, default=32)

    per = ap.add_argument_group("model-written behavioural evaluations")
    per.add_argument("--persona", action="store_true",
                     help="risk personas and advanced-AI-risk evaluations")
    per.add_argument("--persona-temperature", type=float, default=0.0)
    per.add_argument("--persona-seed", type=int, default=None,
                     help="sampling seed (default: first --seeds value)")
    per.add_argument("--persona-cap", type=int, default=None, help="max items per behaviour")
    per.add_argument("--persona-behaviors", nargs="*", default=None, metavar="NAME",
                     help="subset of behaviours (default: all eight)")
    per.add_argument("--persona-max-new-tokens", type=int, default=64)

    dt = ap.add_argument_group("DTBench")
    dt.add_argument("--dtbench", action="store_true",
                    help="decision-theory capability and attitude questions")
    dt.add_argument("--dtbench-parts", nargs="*", default=None,
                    choices=("capabilities", "attitudes"), help="default: both")
    dt.add_argument("--dtbench-cap", type=int, default=None, help="max questions per part")
    dt.add_argument("--dtbench-exclude-tags", nargs="*", default=None, metavar="TAG",
                    help="drop questions carrying any of these tags")
    dt.add_argument("--dtbench-max-new-tokens", type=int, default=None,
                    help="default: --max-new-tokens")
    return ap.parse_args()


def count_situations(args) -> dict[str, int]:
    if args.num_situations == "full":
        return {ds: DATASET_SIZES[ds] for ds in args.datasets}
    n = int(args.num_situations)
    if n <= 0:
        raise SystemExit("--num-situations must be positive or 'full'")
    return {ds: n for ds in args.datasets}


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


async def evaluate(args, system_prompt: str | None, out_dir: Path, raw_dir: Path,
                   cache_dir: Path) -> list[dict]:
    num_by_dataset = count_situations(args)
    rep_sfx = f"_{args.rep_tag}" if args.rep_tag else ""
    common = base_row(args, system_prompt)
    rows: list[dict] = []

    def client_for(name: str, thinking: bool):
        return make_client(args.base_url, args.model, concurrency=args.concurrency,
                           cache_path=cache_dir / f"{args.tag}{rep_sfx}-{name}.jsonl",
                           enable_thinking=thinking)

    # Risk-preference benchmarks.
    client = client_for("risk", args.thinking)
    think_tag_checked = False
    try:
        for ds in args.datasets:
            for seed in args.seeds:
                cfg = EvalConfig(
                    dataset=ds,
                    base_model=args.base_model,
                    num_situations=num_by_dataset[ds],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=seed,
                    max_new_tokens=args.max_new_tokens,
                    reasoning_max_tokens=args.reasoning_max_tokens,
                    system_prompt=system_prompt,
                )
                result = await run_evaluation(cfg, client)
                if args.thinking and not think_tag_checked and result.results:
                    # The parser needs the raw <think> block; a server started with a
                    # reasoning parser strips it.
                    if "<think>" not in (result.results[0].get("response") or ""):
                        raise SystemExit("thinking mode returned no <think> tags; run the "
                                         "server without --reasoning-parser")
                    think_tag_checked = True

                raw_path = raw_dir / f"{ds}-s{seed}.json"
                raw_path.write_text(json.dumps({
                    "evaluation_config": {
                        **common,
                        "dataset": ds,
                        "num_situations": num_by_dataset[ds],
                        "eval_seed": seed,
                        "system_prompt": cfg.system_prompt,
                        "temperature": cfg.temperature,
                        "top_p": cfg.top_p,
                        "top_k": cfg.top_k,
                        "max_new_tokens": cfg.max_new_tokens,
                        "reasoning_max_tokens": cfg.reasoning_max_tokens,
                    },
                    "metrics": result.metrics,
                    "results": result.results,
                }, indent=2, default=str))

                row = {
                    **common,
                    "dataset": ds,
                    "eval_seed": seed,
                    **rate_row(result.results),
                    "metrics": result.metrics,
                    "raw_path": str(raw_path),
                }
                rows.append(row)
                print(f"[{args.tag}] {ds} seed={seed}: coop={row['cooperate_rate']} "
                      f"steal={row['steal_rate']} "
                      f"parse_failed={row['n_parse_failed']}/{row['n_total']}")

        if len(args.seeds) > 1:
            for ds in args.datasets:
                pooled = []
                for r in rows:
                    if r["dataset"] == ds and r.get("eval_seed") is not None:
                        pooled.extend(json.loads(Path(r["raw_path"]).read_text())["results"])
                rows.append({
                    **common,
                    "dataset": ds,
                    "eval_seed": None,
                    "pooled_over_seeds": list(args.seeds),
                    **rate_row(pooled),
                })
    finally:
        await client.aclose()

    skip_reason = ("prompted arm: the weights are the base model's and the exam carries "
                   "no system prompt")

    if args.mmlu:
        if system_prompt is not None:
            print(f"[{args.tag}] skipping MMLU ({skip_reason})")
        else:
            from rac.evals.mmlu import run_mmlu

            mmlu_client = client_for("mmlu", False)
            try:
                out_path = raw_dir / "mmlu_redux.json"
                summary = await run_mmlu(
                    client=mmlu_client,
                    base_model=args.base_model,
                    output=str(out_path),
                    temperature=0.0, top_p=1.0, top_k=-1,
                    seed=args.seeds[0],
                    max_eval_examples_per_subject=args.mmlu_max_per_subject,
                )
            finally:
                await mmlu_client.aclose()
            metrics = summary.get("metrics") or {}
            rows.append({**common, "dataset": "mmlu_redux", "eval_seed": args.seeds[0],
                         "metrics": metrics, "raw_path": str(out_path)})
            print(f"[{args.tag}] mmlu_redux: acc={metrics.get('accuracy')}")

    if args.gpqa:
        if system_prompt is not None:
            print(f"[{args.tag}] skipping GPQA ({skip_reason})")
        else:
            from rac.evals.gpqa import DEFAULT_REVISION, DEFAULT_SHA256, run_gpqa

            gpqa_client = client_for("gpqa", False)
            try:
                summary = await run_gpqa(
                    client=gpqa_client,
                    base_model=args.base_model,
                    n=args.gpqa_n,
                    seed=args.gpqa_seed,
                    revision=args.gpqa_revision or DEFAULT_REVISION,
                    sha256_pin=args.gpqa_sha256 or DEFAULT_SHA256,
                    temperature=0.0, top_p=1.0, top_k=-1,
                    gen_seed=args.seeds[0],
                    max_new_tokens=args.gpqa_max_new_tokens,
                )
            finally:
                await gpqa_client.aclose()
            out_path = raw_dir / "gpqa_main.json"
            out_path.write_text(json.dumps(summary, indent=2, default=str))
            m = summary["metrics"]
            rows.append({
                **common, "dataset": "gpqa_main", "eval_seed": args.seeds[0],
                "accuracy": m["accuracy"],
                "accuracy_se": binomial_se(m["accuracy"], m["n_total"]),
                "parse_rate": m["parse_rate"],
                "metrics": m,
                "raw_path": str(out_path),
            })
            print(f"[{args.tag}] gpqa_main: acc={m['accuracy']:.4f} "
                  f"({m['n_correct']}/{m['n_total']}, parse {m['parse_rate']:.2f})")

    # The behavioural evaluations run for prompted arms too: the prompted model's
    # self-report is the comparison of interest.
    if args.persona:
        from rac.evals.persona import run_persona

        p_temp = args.persona_temperature
        p_seed = args.persona_seed if args.persona_seed is not None else args.seeds[0]
        persona_client = client_for(f"persona{rep_sfx}", False)
        try:
            summary = await run_persona(
                client=persona_client,
                base_model=args.base_model,
                behaviors=args.persona_behaviors,
                cap=args.persona_cap,
                system_prompt=system_prompt,
                temperature=p_temp,
                top_p=(args.top_p if p_temp > 0 else 1.0),
                top_k=(args.top_k if p_temp > 0 else -1),
                seed=p_seed,
                max_new_tokens=args.persona_max_new_tokens,
            )
        finally:
            await persona_client.aclose()

        out_path = raw_dir / f"persona{rep_sfx}.json"
        out_path.write_text(json.dumps({
            "evaluation_config": {**common, **summary["config"]},
            "metrics": summary["metrics"],
            "by_behavior": summary["by_behavior"],
            "results": summary["results"],
        }, indent=2, default=str))

        # One row per behaviour; no pooled row, since the behaviours are opposites.
        for key, m in summary["by_behavior"].items():
            rows.append({
                **common,
                "dataset": f"persona:{key}",
                "eval_seed": p_seed,
                "persona_temperature": p_temp,
                "match_rate": m["match_rate"],
                "match_se": binomial_se(m["match_rate"], m["n_parsed"]),
                "parse_rate": m["parse_rate"],
                "n_total": m["n_total"],
                "n_parsed": m["n_parsed"],
                "n_parse_failed": m["n_parse_failed"],
                "expected_match_rate_if_random": m["expected_match_rate_if_random"],
                "metrics": m,
                "raw_path": str(out_path),
            })
            print(f"[{args.tag}] {key}: match={m['match_rate']} "
                  f"parse_failed={m['n_parse_failed']}/{m['n_total']}")

    # DTBench asks for step-by-step reasoning, so it uses the risk suite's sampling
    # settings and thinking mode.
    if args.dtbench:
        from rac.evals.dtbench import run_dtbench

        dt_client = client_for("dtbench", args.thinking)
        try:
            summary = await run_dtbench(
                client=dt_client,
                base_model=args.base_model,
                parts=args.dtbench_parts,
                exclude_tags=args.dtbench_exclude_tags or [],
                cap=args.dtbench_cap,
                system_prompt=system_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seeds[0],
                max_new_tokens=args.dtbench_max_new_tokens or args.max_new_tokens,
            )
        finally:
            await dt_client.aclose()

        out_path = raw_dir / "dtbench.json"
        out_path.write_text(json.dumps({
            "evaluation_config": {**common, **summary["config"]},
            "by_part": summary["by_part"],
            "results": summary["results"],
        }, indent=2, default=str))

        for part, m in summary["by_part"].items():
            row = {
                **common,
                "dataset": f"dtbench:{part}",
                "eval_seed": args.seeds[0],
                "valid_rate": m["valid_rate"],
                "valid_rate_strict": m["valid_rate_strict"],
                "n_total": m["n_total"],
                "n_valid": m["n_valid"],
                "n_invalid": m["n_invalid"],
                "metrics": m,
                "raw_path": str(out_path),
            }
            if part == "capabilities":
                row["accuracy"] = m["accuracy"]
                row["accuracy_se"] = binomial_se(m["accuracy"], m["n_valid"])
                row["chance_accuracy"] = m["chance_accuracy"]
                print(f"[{args.tag}] dtbench:{part}: acc={m['accuracy']} "
                      f"(chance {m['chance_accuracy']}) invalid={m['n_invalid']}/{m['n_total']}")
            else:
                row["edt_rate"] = m["edt_rate"]
                row["cdt_rate"] = m["cdt_rate"]
                row["both_rate"] = m["both_rate"]
                row["neither_rate"] = m["neither_rate"]
                print(f"[{args.tag}] dtbench:{part}: EDT={m['edt_rate']} CDT={m['cdt_rate']} "
                      f"invalid={m['n_invalid']}/{m['n_total']}")
            rows.append(row)

    return rows


def dry_run(args, system_prompt: str | None) -> None:
    num_by_dataset = count_situations(args)
    sha = hashlib.sha256(system_prompt.encode()).hexdigest()[:12] if system_prompt else None
    print(f"dry run: model={args.model} base_model={args.base_model} "
          f"constitution={args.constitution} system_prompt_sha={sha} "
          f"thinking={args.thinking} server={args.base_url}")
    n_runs = 0
    for ds in args.datasets:
        for seed in args.seeds:
            print(f"  would run: {ds} n={num_by_dataset[ds]} seed={seed}")
            n_runs += 1
    prompted = system_prompt is not None
    if args.mmlu:
        print("  would skip: mmlu_redux (prompted arm)" if prompted
              else f"  would run: mmlu_redux ({args.mmlu_max_per_subject}/subject, thinking off)")
        n_runs += not prompted
    if args.gpqa:
        print("  would skip: gpqa_main (prompted arm)" if prompted
              else f"  would run: gpqa_main n={args.gpqa_n} (thinking off)")
        n_runs += not prompted
    if args.persona:
        from rac.evals.persona import load_suite

        items = load_suite(behaviors=args.persona_behaviors, cap=args.persona_cap)
        by_beh: dict[str, int] = {}
        for it in items:
            key = f"{it.family}/{it.behavior}"
            by_beh[key] = by_beh.get(key, 0) + 1
        for key, n in by_beh.items():
            print(f"  would run: persona:{key} n={n} "
                  f"(thinking off, temperature {args.persona_temperature})")
            n_runs += 1
    if args.dtbench:
        from rac.evals.dtbench import count_by_part, load_suite

        items = load_suite(parts=args.dtbench_parts,
                           exclude_tags=args.dtbench_exclude_tags or [],
                           cap=args.dtbench_cap, seed=args.seeds[0])
        for part, n in count_by_part(items).items():
            print(f"  would run: dtbench:{part} n={n}")
            n_runs += 1
    print(f"total: {n_runs} evaluation runs")


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
        dry_run(args, system_prompt)
        return

    rep_sfx = f"_{args.rep_tag}" if args.rep_tag else ""
    out_root = Path(args.out_dir)
    out_dir = out_root / f"{args.tag}{rep_sfx}"
    raw_dir = out_root / args.tag / "raw"
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_root / "cache"
    for d in (out_dir, raw_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = asyncio.run(evaluate(args, system_prompt, out_dir, raw_dir, cache_dir))
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        **base_row(args, system_prompt),
        "rep_tag": args.rep_tag,
        "base_url": args.base_url,
        "seeds": args.seeds,
        "system_prompt": system_prompt,
        "rows": rows,
    }, indent=2, default=str) + "\n")
    print(f"[{args.tag}] summary -> {summary_path}")


if __name__ == "__main__":
    main()
