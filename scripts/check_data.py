"""Verify the shipped data files against their recorded sizes and hashes."""
from __future__ import annotations

import hashlib
import sys

from risk_averse_character.paths import DATA_DIR

# (relative path, non-blank line count or None, sha256)
FILES = [
    ("risk_seeds_v2.jsonl", 960,
     "69b3f649957286131a327947971509aa707e9043f510858f39735c836b8f5475"),
    ("risk_benchmark/2026_03_22_medium_stakes_val_set_500_Rebels.csv", None,
     "faef3c7128849298fce780345670f22a1eb77c6c4725c2d595adecfd79c62056"),
    ("risk_benchmark/2026_03_22_high_stakes_test_set_1000_Rebels.csv", None,
     "0945815f7e53c01ef8e0eb4f72359b704aeb02e8d576239f1bd739f845fe5cf3"),
    ("risk_benchmark/2026_03_22_astronomical_stakes_deployment_set_1000_Rebels.csv", None,
     "b9e7c596d187cb51f073c918af013849e8c688fea20aeacc4451e2b7114cf5cb"),
    ("risk_benchmark/2026_03_22_test_set_1000_Steals.csv", None,
     "d2c72f5649294e5b8ee89650ddf8ea602bea659fbcd90006d82b82a1ad3aa40a"),
    ("risk_benchmark/2026_04_11_gpu_hours_transfer_benchmark_interleaved_1000_situations.csv",
     None, "174dc1d6e1e12e527e22e0eb50e8a4bc06d2ce0afcec9bc68a1c7b8e17a80af5"),
    ("risk_benchmark/2026_04_11_lives_saved_transfer_benchmark_interleaved_1000_situations.csv",
     None, "70bd2d9e62ea32b1430a84de87fe9c8d9d6b04bf9fa80b5a2f67f49e893063bb"),
    ("risk_benchmark/2026_04_11_money_for_user_transfer_benchmark_interleaved_1000_situations.csv",
     None, "d80cd3313f1a6dea621b75859a58446b9e29144a002aaebe91581498abacae67"),
    ("persona/persona/risk-averse.jsonl", 1000,
     "0e22ad96b41189572fb4234855d42d9683c5762e7aa73e3cf01af3f378bd5064"),
    ("persona/persona/risk-neutral.jsonl", 1000,
     "944c5dfd79415dfeb3a55777e4c2fd08ec5c681881ad7ac9ae9de34c049c8230"),
    ("persona/persona/risk-seeking.jsonl", 1000,
     "de65fc47017bb2719189bba77460f315cf021a1532effe171e8c5454c7c08435"),
    ("persona/advanced_ai_risk/myopic-reward.jsonl", 1000,
     "03d4e57220cbe825c9b76df39085a945e8c2fa2c6c7e937fb6df94143491237f"),
    ("persona/advanced_ai_risk/one-box-tendency.jsonl", 300,
     "5e4857204ae346dc3e20809dd09a9e2130020c35b0251cf04375d8b2838dddb7"),
    ("persona/advanced_ai_risk/power-seeking-inclination.jsonl", 998,
     "a8a2094a1e404fb094b3d361e3049ecc06ecd2a26d0b4b827b4a83a98b6d886a"),
    ("persona/advanced_ai_risk/survival-instinct.jsonl", 953,
     "0831782a9e0462f137eb9c4884edd19f248e18f17770f7a4540c67937b3af0e8"),
    ("persona/advanced_ai_risk/wealth-seeking-inclination.jsonl", 985,
     "7583bec06bd799cf21fe854c6e8b92ebb669ce02579ea1a869aa460cc0f18546"),
    ("dtbench/data.zip", None,
     "3adba2704c03050de363fc306cd06eb73ed7ad5741631ad942c8a9e1f6044258"),
    ("bailbench/bailBench.csv", None,
     "332684740cf3f76cc6ec0a99ec8365b5a03c26053fed75e78661ba0f326d005d"),
    ("bailbench/nonBailRequestsWildchat.csv", None,
     "6b3cc4ae2002b0975f8dacb705ed3dc61db16534a02005783ce790eab41fdf8c"),
    ("mu_decisiveness/items_500.jsonl", 500,
     "0f1a77c4818c6c292bfa56e3d7df2889c357fad9c78b82968952426b7f978401"),
    ("mu_decisiveness/questions_main.jsonl", 2,
     "d9abb00ffb0cccedffe5fa209f2a873fd454c2d45a45414f62d824fff50dd472"),
    ("ood/embedded_decision.jsonl", 70,
     "e6ebcc7cdf461b7b781cf379571875ed5d8524719441700198d5f627fb629e1e"),
    ("ood/agentic_tool.jsonl", 70,
     "203a63d21b6ab5ed2754ceb1bc922ddb429541d56701c3d7be9daf11b6ad70d1"),
    ("ood/verbal_uncertainty.jsonl", 64,
     "db3d20fa473706060a30195d4e3f1cfae876a07e33429b0e89b48c9288899337"),
    ("ood/open_ended_allocation.jsonl", 64,
     "5c3e386d10f3c410989b28001199534653608eb0f86484f581487d3c8a7ce329"),
    ("ood/calibration_threshold.jsonl", 64,
     "58e26045980f8ecdc1d6e6ab23b6444a7c5cc3d62ab4088a9b78e807f7c92b20"),
]

BASELINE_FILES = [
    ("2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv",
     "005e9ae0cb7ebce4ed161095977758a598751b1ece684bfb513903058a11319d"),
    ("2026_04_13_tie_training_modified_CoTs.csv",
     "1534b6ee0c95c0d942748fe8ffd8841fb20f6e616d5ce9f59fa39155e6709a96"),
    ("2026_03_22_low_stakes_training_set_1000_situations_CoTs_no_think_tags.csv",
     "97bff8343c750c2ea86d54dd4b32a32e8370371d582c0b4248d1816c135fede9"),
    ("2026_04_13_tie_training_modified_CoTs_no_think_tags.csv",
     "67451c62e54636c20d48f8eb0380f72d49e40ec90d16026a41bf1ffdbd8a6905"),
    ("2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv",
     "3b26979d0d1c98f48eb76482a2ca53a55c749e63f04a42ddb239061b22e047e1"),
    ("2026_03_22_low_stakes_training_set_600_situations_CoTs_no_think_tags_lin_only.csv",
     "266fe8e1dc4fefc38b94738cc080ae489f34cd0fb71f70eebb6350f1e9b6a044"),
]

SITUATIONS = {
    "medium_stakes_validation": 500,
    "high_stakes_test": 1000,
    "astronomical_stakes_deployment": 1000,
    "steals_test": 1000,
    "gpu_hours_transfer_benchmark": 1000,
    "lives_saved_transfer_benchmark": 1000,
    "money_for_user_transfer_benchmark": 1000,
}


def check_file(path, rows, sha, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"{path}: missing")
        print(f"{path.name}: MISSING")
        return
    raw = path.read_bytes()
    got_sha = hashlib.sha256(raw).hexdigest()
    got_rows = len([ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]) \
        if rows is not None else None
    ok = got_sha == sha and got_rows == rows
    shown = f"{got_rows} rows" if rows is not None else f"{len(raw)} bytes"
    print(f"{path.name}: {shown}, sha256 {got_sha[:16]} {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append(str(path))


def main() -> None:
    failures: list[str] = []
    for rel, rows, sha in FILES:
        check_file(DATA_DIR / rel, rows, sha, failures)
    for name, sha in BASELINE_FILES:
        check_file(DATA_DIR / "baselines" / name, None, sha, failures)

    from risk_averse_character.evals.risk.runner import load_situations

    for dataset, expected in SITUATIONS.items():
        n = len(load_situations(dataset, None))
        print(f"{dataset}: {n} situations {'OK' if n == expected else f'FAIL (expected {expected})'}")
        if n != expected:
            failures.append(dataset)

    from risk_averse_character.evals.persona import load_suite as load_persona
    from risk_averse_character.evals.dtbench import count_by_part, load_all

    n = len(load_persona())
    print(f"persona suite: {n} items {'OK' if n == 7236 else 'FAIL (expected 7236)'}")
    if n != 7236:
        failures.append("persona item count")
    counts = count_by_part(load_all(shuffle_answers=False))
    ok = counts == {"attitudes": 130, "capabilities": 407}
    print(f"dtbench suite: {counts} {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append("dtbench counts")

    if failures:
        sys.exit(f"failed checks: {failures}")
    print("all data checks passed")


if __name__ == "__main__":
    main()
