"""The SFT, tie-training and DPO baselines of Zhang et al., trained on the
benchmark's low-stakes training split with the upstream recipes.

The example sets are built by the upstream selection routines in
`risk_averse_character.training.baseline_data` (same sampling seeds and shuffle order); only the
chat rendering and the training loop live here. Rendering follows the chat
family of the renderer used for evaluation, so the training and evaluation
prompts match byte for byte.

  sft   1,000 low-stakes CoT demonstrations, 4 epochs, batch 4 x accumulation 4,
        cosine schedule with 10% warmup, LoRA r32/alpha64/dropout 0.05 on the
        7 projection modules, prompt tokens masked, max sequence 4096, seed 42.
  tie   the same trainer on 700 unmodified + 300 tie-modified rows (30% tie rate).
  dpo   600 preference pairs, 3 epochs, batch 2 x accumulation 8, beta 0.10,
        max length 2048 / prompt 1792, 4-bit NF4 base with bf16 compute,
        the same LoRA geometry, seed 12345.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from risk_averse_character.chat_formats import family as render_family
from risk_averse_character.chat_formats import render_ids, render_prefix
from risk_averse_character.evals.risk.prompts import DEFAULT_SYSTEM_PROMPT
from risk_averse_character.paths import DATA_DIR, REPO_ROOT
from risk_averse_character.training import baseline_data as up

BASELINE_DATA = DATA_DIR / "baselines"

# the upstream training files by variant; nothing else may be trained on
_LOW = "2026_03_22_low_stakes_training_set"
_TIE = "2026_04_13_tie_training_modified_CoTs"
DATA = {
    "think_tags": {
        "unmodified": BASELINE_DATA / f"{_LOW}_1000_situations_with_CoTs.csv",
        "modified": BASELINE_DATA / f"{_TIE}.csv",
        "dpo": BASELINE_DATA / f"{_LOW}_600_situations_with_CoTs_lin_only.csv",
    },
    "no_think_tags": {
        "unmodified": BASELINE_DATA / f"{_LOW}_1000_situations_CoTs_no_think_tags.csv",
        "modified": BASELINE_DATA / f"{_TIE}_no_think_tags.csv",
        "dpo": BASELINE_DATA / f"{_LOW}_600_situations_CoTs_no_think_tags_lin_only.csv",
    },
}
_TRAINING_SPLIT_RE = re.compile(r"(low_stakes_training_set|tie_training_modified_CoTs)")
_FORBIDDEN_RE = re.compile(r"(val_set|test_set|deployment_set|transfer_benchmark|Steals|Rebels)")

LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class SupervisedRecipe:
    n_unmodified: int
    n_modified: int
    num_train_epochs: float = 4.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 4096
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    seed: int = 42
    warmup_frac: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    logging_steps: int = 10
    allow_thinking: bool = True
    append_answer_instruction: bool = True
    modified_pct_input: float = 40.0     # upstream's --modified_completion_pcts default; seeds only


@dataclass(frozen=True)
class DPORecipe:
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    beta: float = 0.10
    max_length: int = 2048
    max_prompt_length: int = 1792
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    seed: int = 12345
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = 50
    save_total_limit: int = 5
    logging_steps: int = 10


RECIPES: dict[str, Any] = {
    "sft": SupervisedRecipe(n_unmodified=1000, n_modified=0),
    "tie": SupervisedRecipe(n_unmodified=700, n_modified=300),
    "dpo": DPORecipe(),
}


def default_system_prompt() -> str:
    """The gamble-task system prompt shared with the evaluation."""
    return DEFAULT_SYSTEM_PROMPT


def assert_training_split_only(path: Path) -> Path:
    """Refuse any file that is not the benchmark's low-stakes training split."""
    name = Path(path).name
    if _FORBIDDEN_RE.search(name) or not _TRAINING_SPLIT_RE.search(name):
        raise RuntimeError(f"refusing to train on {name}: not the low-stakes training split")
    rp = Path(path).resolve()
    if BASELINE_DATA.resolve() not in rp.parents:
        raise RuntimeError(f"refusing to train on {rp}: outside {BASELINE_DATA}")
    return Path(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- SFT / tie

@dataclass
class SupervisedExample:
    situation_id: int
    source: str            # unmodified | modified
    target_label: str
    prompt: str            # the user turn, after modify_prompt_for_training
    completion: str        # CoT + "\n\nFINAL ANSWER: <label>"


def build_supervised_examples(method: str, variant: str, seed_base: int | None = None
                              ) -> tuple[list[SupervisedExample], dict]:
    """The upstream example set for `sft` or `tie`, in the upstream order:
    load and filter both CSVs, sample exact counts as whole situations,
    build each completion from its CoT, shuffle, and reorder to separate
    rows of the same situation. Only the chat rendering is deferred."""
    rec: SupervisedRecipe = RECIPES[method]
    seed = rec.seed if seed_base is None else seed_base
    files = DATA[variant]
    unmod = up.load_cot_examples(assert_training_split_only(files["unmodified"]),
                                 prompt_col="prompt_text", completion_col="chosen_full",
                                 situation_col="situation_id", cowinner_col="chosen_expected")
    mod = (up.load_cot_examples(assert_training_split_only(files["modified"]),
                                prompt_col="prompt_text", completion_col="chosen_full",
                                situation_col="situation_id", cowinner_col="chosen_expected")
           if rec.n_modified > 0 else unmod.iloc[0:0].copy())
    up.validate_exact_cot_example_request(unmod, rec.n_unmodified, "unmodified")
    if rec.n_modified > 0:
        up.validate_exact_cot_example_request(mod, rec.n_modified, "modified")
    pct = rec.modified_pct_input
    selected, sampled_situations = up.sample_cot_examples_exact(
        unmodified_cot_examples=unmod, modified_cot_examples=mod,
        n_unmodified_examples=rec.n_unmodified, n_modified_examples=rec.n_modified,
        seed=seed + int(round(pct * 10)))
    build_seed = seed + int(pct * 100)
    records = []
    shuffled = selected.sample(frac=1.0, random_state=build_seed).reset_index(drop=True)
    for _, row in shuffled.iterrows():
        source = str(row.get("_source", "unmodified")).strip().lower() or "unmodified"
        chosen_label = up.sanitize_token(row.get("co_winner_label"))
        if not chosen_label:
            parsed = up.parse_label_tokens(row.get("co_winner_label"))
            chosen_label = parsed[0] if parsed else ""
        if not chosen_label:
            raise RuntimeError(f"situation {row['situation_id']}: no chosen_expected label")
        assistant_text = up.build_assistant_completion(
            chosen_label=chosen_label, all_correct_labels=[chosen_label],
            style="cot_and_label" if rec.allow_thinking else "label_only",
            allow_thinking=rec.allow_thinking, cot_text=row["completion"])
        prompt = up.modify_prompt_for_training(
            row["prompt"], allow_thinking=rec.allow_thinking,
            append_answer_instruction=rec.append_answer_instruction)
        records.append({"situation_id": int(row["situation_id"]), "_source": source,
                        "target_label": chosen_label, "prompt": prompt,
                        "completion": assistant_text})
    ordered = up.reorder_examples_to_reduce_adjacent_same_situation(records, seed=build_seed + 17)
    examples = [SupervisedExample(r["situation_id"], r["_source"], r["target_label"],
                                  r["prompt"], r["completion"]) for r in ordered]
    n_mod = sum(e.source == "modified" for e in examples)
    stats = {
        "method": method, "variant": variant, "n_examples": len(examples),
        "n_unmodified": len(examples) - n_mod, "n_modified": n_mod,
        "n_situations": int(len(sampled_situations)),
        "seed": seed, "sample_seed": seed + int(round(pct * 10)),
        "build_seed": build_seed, "reorder_seed": build_seed + 17,
        "files": {k: str(files[k].relative_to(REPO_ROOT)) for k in ("unmodified", "modified")},
        "sha256": {k: sha256_file(files[k]) for k in ("unmodified", "modified")},
    }
    if len(examples) != rec.n_unmodified + rec.n_modified:
        raise RuntimeError(f"{method}: built {len(examples)} examples, recipe wants "
                           f"{rec.n_unmodified + rec.n_modified}")
    return examples, stats


# --------------------------------------------------------------------- DPO

@dataclass
class PreferenceExample:
    situation_id: int
    prompt: str
    chosen: str
    rejected: str
    rejected_type: str


def build_preference_examples(variant: str) -> tuple[list[PreferenceExample], dict]:
    """One {prompt, chosen, rejected} per CSV row, stripped, in CSV order;
    rows with an empty field are dropped (as upstream's prepare_dpo_dataset)."""
    import pandas as pd

    path = assert_training_split_only(DATA[variant]["dpo"])
    df = pd.read_csv(path)
    for col in ("prompt_text", "chosen_full", "rejected_full"):
        if col not in df.columns:
            raise RuntimeError(f"{path.name}: missing column {col}")
    out = []
    for _, row in df.iterrows():
        p, c, r = (str(row["prompt_text"]).strip(), str(row["chosen_full"]).strip(),
                   str(row["rejected_full"]).strip())
        if not p or not c or not r:
            continue
        out.append(PreferenceExample(int(row["situation_id"]), p, c, r,
                                     str(row.get("rejected_type", ""))))
    if not out:
        raise RuntimeError(f"{path.name}: no preference pairs")
    stats = {"method": "dpo", "variant": variant, "n_pairs": len(out),
             "files": {"dpo": str(path.relative_to(REPO_ROOT))},
             "sha256": {"dpo": sha256_file(path)}}
    return out, stats


# ------------------------------------------------------------ tokenization

def tokenize_supervised(tokenizer, fam: str, system_prompt: str,
                        examples: list[SupervisedExample], max_seq_length: int) -> list[dict]:
    rows, n_trunc = [], 0
    for ex in examples:
        p_ids, c_ids = render_ids(tokenizer, fam, render_prefix(fam, system_prompt, ex.prompt),
                                  ex.completion)
        ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + c_ids
        if len(ids) > max_seq_length:          # right truncation, as upstream
            ids, labels = ids[:max_seq_length], labels[:max_seq_length]
            n_trunc += 1
        rows.append({"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)})
    if n_trunc:
        print(f"[baselines] {n_trunc}/{len(rows)} examples truncated to {max_seq_length} tokens")
    return rows


def tokenize_preference(tokenizer, fam: str, system_prompt: str,
                        examples: list[PreferenceExample], max_prompt_length: int,
                        max_length: int) -> list[dict]:
    """Pre-tokenized DPO rows in TRL's collator format. Completions carry the
    turn end; the prompt is truncated from the left to `max_prompt_length`
    and completions from the right to the remaining room, as TRL does."""
    rows = []
    for ex in examples:
        prefix = render_prefix(fam, system_prompt, ex.prompt)
        p_ids, c_ids = render_ids(tokenizer, fam, prefix, ex.chosen)
        _, r_ids = render_ids(tokenizer, fam, prefix, ex.rejected)
        if len(p_ids) > max_prompt_length:
            p_ids = p_ids[-max_prompt_length:]
        room = max_length - len(p_ids)
        rows.append({"prompt_input_ids": p_ids, "chosen_input_ids": c_ids[:room],
                     "rejected_input_ids": r_ids[:room]})
    return rows


def count_supervised_tokens(rows: list[dict]) -> int:
    return sum(sum(1 for t in r["labels"] if t != -100) for r in rows)


# ------------------------------------------------------------------ models

def _lora_config(r: int, alpha: int, dropout: float, base):
    from peft import LoraConfig

    from risk_averse_character.training.distill import lora_exclusion

    exclude = lora_exclusion(base, LORA_TARGET_MODULES)
    if exclude:
        print(f"[baselines] LoRA: excluding {exclude['count']} target-named non-Linear "
              f"modules under {exclude['prefixes']}")
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                      target_modules=list(LORA_TARGET_MODULES), task_type="CAUSAL_LM",
                      **({"exclude_modules": exclude["regex"]} if exclude else {}))


def load_tokenizer(model: str, revision: str | None, padding_side: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


@dataclass
class TrainResult:
    out_dir: str
    adapter_path: str
    final_metrics: dict
    train_tokens_total: int
    n_examples: int
    steps: int


def _log_tap(sink: list[dict], on_log: Callable[[int, dict], None] | None):
    """Trainer callback that records every log row and forwards it."""
    from transformers import TrainerCallback

    class Tap(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            row = dict(logs or {})
            row["step"] = state.global_step
            sink.append(row)
            if on_log:
                on_log(state.global_step, row)
            return control

    return Tap()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


def train_supervised(*, method: str, model: str, revision: str | None, renderer: str,
                     variant: str, system_prompt: str, lr: float, out: Path,
                     device: str = "cuda:0", gradient_checkpointing: bool = False,
                     max_examples: int | None = None, max_steps: int | None = None,
                     on_log: Callable[[int, dict], None] | None = None) -> TrainResult:
    """SFT or tie-training on transformers' Trainer: causal LM cross-entropy
    over the unmasked completion tokens, with the upstream trainer settings.
    The adapter is saved under `<out>/adapter/`."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    rec: SupervisedRecipe = RECIPES[method]
    fam = render_family(renderer)
    out.mkdir(parents=True, exist_ok=True)
    examples, stats = build_supervised_examples(method, variant)
    if max_examples:
        examples = examples[:max_examples]
        stats["smoke_max_examples"] = max_examples
    tok = load_tokenizer(model, revision, padding_side="right")
    rows = tokenize_supervised(tok, fam, system_prompt, examples, rec.max_seq_length)
    sup_tokens = count_supervised_tokens(rows)
    stats.update({"system_prompt": system_prompt, "render_family": fam,
                  "supervised_tokens_per_epoch": sup_tokens,
                  "mean_seq_len": sum(len(r["input_ids"]) for r in rows) / len(rows)})
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    _write_jsonl(out / "sampled_training_rows.jsonl",
                 [{"situation_id": e.situation_id, "source": e.source,
                   "target_label": e.target_label} for e in examples])
    (out / "example0.txt").write_text(tok.decode(rows[0]["input_ids"]))

    base = AutoModelForCausalLM.from_pretrained(model, revision=revision,
                                                torch_dtype=torch.bfloat16, device_map=device)
    base.config.use_cache = False
    peft_model = get_peft_model(base, _lora_config(rec.lora_r, rec.lora_alpha,
                                                   rec.lora_dropout, base))
    peft_model.print_trainable_parameters()
    if gradient_checkpointing:
        peft_model.gradient_checkpointing_enable()
        peft_model.enable_input_require_grads()

    eff = rec.per_device_train_batch_size * rec.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(rows) / eff))
    total_steps = max(1, math.ceil(steps_per_epoch * rec.num_train_epochs))
    if max_steps:
        total_steps = min(total_steps, max_steps)
    warmup = max(1, int(round(rec.warmup_frac * total_steps)))

    pad_id = tok.pad_token_id

    def collate(features):
        m = max(len(f["input_ids"]) for f in features)
        ids = torch.full((len(features), m), pad_id, dtype=torch.long)
        lab = torch.full((len(features), m), -100, dtype=torch.long)
        att = torch.zeros((len(features), m), dtype=torch.long)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"])
            lab[i, :n] = torch.tensor(f["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    logs: list[dict] = []
    targs = TrainingArguments(
        output_dir=str(out / "trainer_outputs"),
        num_train_epochs=rec.num_train_epochs,
        per_device_train_batch_size=rec.per_device_train_batch_size,
        gradient_accumulation_steps=rec.gradient_accumulation_steps,
        learning_rate=lr, logging_steps=rec.logging_steps, save_strategy="no",
        bf16=True, report_to="none", lr_scheduler_type=rec.lr_scheduler_type,
        warmup_steps=warmup, optim=rec.optim, gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=rec.max_grad_norm, seed=rec.seed, remove_unused_columns=False,
        dataloader_pin_memory=False,
        **({"max_steps": total_steps} if max_steps else {}),
    )

    class ListDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return rows[i]

    trainer = Trainer(model=peft_model, args=targs, train_dataset=ListDataset(),
                      data_collator=collate, callbacks=[_log_tap(logs, on_log)])
    t0 = time.time()
    trainer.train()
    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir / "tokenizer"))
    _write_jsonl(out / "metrics.jsonl", logs)
    last = next((row for row in reversed(logs) if "loss" in row), {})
    epochs_run = rec.num_train_epochs if not max_steps else total_steps / steps_per_epoch
    return TrainResult(out_dir=str(out), adapter_path=str(adapter_dir),
                       final_metrics={"train_loss": last.get("loss"),
                                      "wall_seconds": round(time.time() - t0),
                                      "total_steps": trainer.state.global_step,
                                      "warmup_steps": warmup, "epochs": epochs_run},
                       train_tokens_total=int(round(sup_tokens * epochs_run)),
                       n_examples=len(rows), steps=trainer.state.global_step)


def train_dpo(*, model: str, revision: str | None, renderer: str, variant: str,
              system_prompt: str, lr: float, out: Path, device: str = "cuda:0",
              max_examples: int | None = None, max_steps: int | None = None,
              on_log: Callable[[int, dict], None] | None = None) -> TrainResult:
    """DPO on TRL's DPOTrainer with a 4-bit base. The adapter is saved under
    `<out>/adapter/`."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from trl import DPOConfig, DPOTrainer

    class PreTokenizedDPOTrainer(DPOTrainer):
        """TRL re-tokenizes text columns in `_prepare_dataset`. The rows here
        are already rendered and truncated in the trainer's format, so that
        step is skipped."""
        _NEED = ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids")

        def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
            missing = [c for c in self._NEED if c not in dataset.column_names]
            if missing:
                raise RuntimeError(f"DPO {dataset_name} rows must be pre-tokenized; "
                                   f"missing {missing} (columns: {dataset.column_names})")
            return dataset

    rec: DPORecipe = RECIPES["dpo"]
    fam = render_family(renderer)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(rec.seed)
    examples, stats = build_preference_examples(variant)
    if max_examples:
        examples = examples[:max_examples]
        stats["smoke_max_examples"] = max_examples
    tok = load_tokenizer(model, revision, padding_side="left")
    rows = tokenize_preference(tok, fam, system_prompt, examples,
                               rec.max_prompt_length, rec.max_length)
    comp_tokens = sum(len(r["chosen_input_ids"]) + len(r["rejected_input_ids"]) for r in rows)
    stats.update({"system_prompt": system_prompt, "render_family": fam,
                  "completion_tokens_per_epoch": comp_tokens,
                  "mean_prompt_len": sum(len(r["prompt_input_ids"]) for r in rows) / len(rows)})
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (out / "example0.txt").write_text(
        tok.decode(rows[0]["prompt_input_ids"]) + "\n<<<CHOSEN>>>\n"
        + tok.decode(rows[0]["chosen_input_ids"]) + "\n<<<REJECTED>>>\n"
        + tok.decode(rows[0]["rejected_input_ids"]))

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True) if rec.load_in_4bit else None
    base = AutoModelForCausalLM.from_pretrained(
        model, revision=revision, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map=device)
    base.config.use_cache = False
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    peft_config = _lora_config(rec.lora_r, rec.lora_alpha, rec.lora_dropout, base)

    from dataclasses import fields as dc_fields
    allowed = {f.name for f in dc_fields(DPOConfig)}
    wanted = {
        "output_dir": str(out / "checkpoints"),
        "per_device_train_batch_size": rec.per_device_train_batch_size,
        "gradient_accumulation_steps": rec.gradient_accumulation_steps,
        "num_train_epochs": rec.num_train_epochs, "learning_rate": lr, "beta": rec.beta,
        "max_length": rec.max_length, "max_prompt_length": rec.max_prompt_length,
        "save_strategy": "steps", "save_steps": rec.save_steps,
        "save_total_limit": rec.save_total_limit, "logging_steps": rec.logging_steps,
        "report_to": "none", "bf16": True, "gradient_checkpointing": rec.gradient_checkpointing,
        "remove_unused_columns": False, "seed": rec.seed, "dataloader_pin_memory": False,
        **({"max_steps": max_steps} if max_steps else {}),
    }
    dropped = sorted(k for k in wanted if k not in allowed)
    critical = {"beta", "max_length", "max_prompt_length", "num_train_epochs", "learning_rate"}
    if critical & set(dropped):
        missing = sorted(critical & set(dropped))
        raise RuntimeError(f"the installed TRL's DPOConfig lacks {missing}; pin trl==0.28.0")
    if dropped:
        print(f"[baselines] DPOConfig ignores {dropped} on this TRL version")
    logs: list[dict] = []
    trainer = PreTokenizedDPOTrainer(
        model=base, args=DPOConfig(**{k: v for k, v in wanted.items() if k in allowed}),
        train_dataset=Dataset.from_list(rows), peft_config=peft_config,
        processing_class=tok, callbacks=[_log_tap(logs, on_log)])
    t0 = time.time()
    trainer.train()
    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir / "tokenizer"))
    _write_jsonl(out / "metrics.jsonl", logs)
    last = next((row for row in reversed(logs) if "loss" in row), {})
    eff = rec.per_device_train_batch_size * rec.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(rows) / eff))
    epochs_run = (rec.num_train_epochs if not max_steps
                  else trainer.state.global_step / steps_per_epoch)
    return TrainResult(out_dir=str(out), adapter_path=str(adapter_dir),
                       final_metrics={"train_loss": last.get("loss"),
                                      "rewards_margin": last.get("rewards/margins"),
                                      "wall_seconds": round(time.time() - t0),
                                      "total_steps": trainer.state.global_step,
                                      "epochs": epochs_run},
                       train_tokens_total=int(round(comp_tokens * epochs_run)),
                       n_examples=len(rows), steps=trainer.state.global_step)
