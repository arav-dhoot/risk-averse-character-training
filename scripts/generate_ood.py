#!/usr/bin/env python3
"""Generate items for the OOD evaluation families, or check the shipped ones.

    python scripts/generate_ood.py --family agentic_tool --n 70 --seed 0 --out items.jsonl
    python scripts/generate_ood.py --family all --n 64 --seed 3 --out ood/   # one file per family
    python scripts/generate_ood.py --check

`--check` reads every shipped file under data/ood/, re-derives each pick-one
item's labels from its option lotteries and each allocation item's optimal
fractions from its parameters, and fails on any disagreement.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from risk_averse_character.evals.ood import FAMILIES, ITEMS_DIR, generate, load_items, write_jsonl
from risk_averse_character.evals.ood.schema import make_allocation_item, make_pick_one_item


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    ap.add_argument("--n", type=int, default=64, help="items per family")
    ap.add_argument("--seed", type=int, default=0, help="generator seed")
    ap.add_argument("--out", default=None,
                    help="output JSONL (one family) or directory (--family all)")
    ap.add_argument("--check", action="store_true", help="verify the shipped items and exit")
    return ap.parse_args()


def check_item(item: dict) -> dict:
    """Rebuild the item's labels from its lotteries."""
    common = dict(item_id=item["item_id"], family=item["family"],
                  axis_dropped=item["axis_dropped"], stakes_tier=item["stakes_tier"],
                  initial_wealth=item["initial_wealth"], prompt=item["prompt"],
                  meta=item["meta"])
    if item["scoring"] == "allocation":
        a = item["allocation"]
        rebuilt = make_allocation_item(
            **common, budget=a["budget"], safe_multiple=a["safe_multiple"],
            up_multiple=a["up_multiple"], down_multiple=a["down_multiple"], p_up=a["p_up"],
            tolerance=a["tolerance"])
    else:
        specs = [{"text": o["text"], "prizes": o["prizes"], "probs": o["probs"]}
                 for o in item["options"]]
        expected = {o["option_type"]: o["label"] for o in item["options"]
                    if o["option_type"] != "Distractor"}
        rebuilt = make_pick_one_item(
            **common, option_specs=specs, label_style=item["answer_label_style"],
            expected_types=expected)
    return json.loads(json.dumps(asdict(rebuilt), sort_keys=True))


def check() -> None:
    failures = []
    for family in FAMILIES:
        items = load_items(family)
        bad = 0
        for item in items:
            if check_item(item) != json.loads(json.dumps(item, sort_keys=True)):
                bad += 1
        print(f"{family}: {len(items)} items, {bad} label mismatches "
              f"{'OK' if bad == 0 else 'FAIL'}  [{ITEMS_DIR / family}.jsonl]")
        if bad:
            failures.append(family)
    if failures:
        raise SystemExit(f"failed checks: {failures}")
    print("all shipped items verified")


def main() -> None:
    args = parse_args()
    if args.check:
        check()
        return
    if args.out is None:
        raise SystemExit("--out is required unless --check is given")
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    if args.family == "all":
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for family in FAMILIES:
            items = generate(family, args.n, args.seed)
            write_jsonl(out_dir / f"{family}.jsonl", items)
            print(f"{family}: {len(items)} items -> {out_dir / family}.jsonl")
    else:
        items = generate(args.family, args.n, args.seed)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.out, items)
        print(f"{args.family}: {len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
