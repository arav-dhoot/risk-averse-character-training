#!/usr/bin/env python3
from __future__ import annotations
"""
Train risk-aversion LoRA adapters, then evaluate them with evaluate.py.

This script supports:
1) Baseline evaluation (base model only, no fine-tuning)
2) Fine-tuning on unmodified data only (0% modified)
3) Fine-tuning on mixes with configurable % modified completions/examples

It keeps evaluation aligned with the shared team workflow by calling evaluate.py
for every model variant.
"""

import argparse
import ast
import gc
import heapq
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from risk_averse_prompts import DEFAULT_SYSTEM_PROMPT, model_uses_no_system_prompt

# Lazy-imported training dependencies (loaded in ensure_training_dependencies()).
torch = None
Dataset = None
LoraConfig = None
get_peft_model = None
PeftModel = None
prepare_model_for_kbit_training = None
AutoModelForCausalLM = None
AutoTokenizer = None
BitsAndBytesConfig = None
TrainingArguments = None
DataCollatorForLanguageModeling = None
SFTTrainer = None
SFTConfig = None

SHARED_EVAL_DEFAULT_TEMPERATURE = 0.6
DEFAULT_UNMODIFIED_COT_DATA = "data/CoT-training/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv"
DEFAULT_MODIFIED_COT_DATA = "data/CoT-training/2026_04_13_tie_training_modified_CoTs.csv"
DEFAULT_MODIFIED_COT_SPLIT_NAME = "low-stakes-tie-training"


def _gpu_supports_bf16() -> bool:
    """Best-effort check for bf16 support (Ampere+ CUDA GPUs)."""
    try:
        import torch as _torch  # local import to keep CLI help lightweight
    except Exception:
        return False
    if not _torch.cuda.is_available():
        return False
    try:
        cc_major, _ = _torch.cuda.get_device_capability()
        return cc_major >= 8 and _torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch as _torch  # local import to keep CLI help lightweight
    except Exception:
        return False
    return bool(_torch.cuda.is_available())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline/unmodified/modified-mix adapters and evaluate with evaluate.py."
    )

    # Data
    parser.add_argument(
        "--full_data",
        type=str,
        default=None,
        help="Path to the full (unmodified/original) training dataset (CSV or Excel).",
    )
    parser.add_argument(
        "--modified_data",
        type=str,
        default=None,
        help=(
            "Path to modified situations dataset (CSV or Excel). "
            "Required if any modified percentage > 0."
        ),
    )
    parser.add_argument(
        "--unmodified_cot_data",
        type=str,
        default=DEFAULT_UNMODIFIED_COT_DATA,
        help=(
            "Path to unmodified CoT training file (CSV/Excel) with prompt + completion "
            "(e.g., unmodifed_COT.xlsx)."
        ),
    )
    parser.add_argument(
        "--modified_cot_data",
        type=str,
        default=DEFAULT_MODIFIED_COT_DATA,
        help=(
            "Path to modified CoT training file (CSV/Excel) with prompt + completion "
            "(e.g., modified_COT.xlsx)."
        ),
    )
    parser.add_argument("--cot_prompt_column", type=str, default="prompt_text")
    parser.add_argument("--cot_completion_column", type=str, default="chosen_full")
    parser.add_argument("--cot_situation_id_column", type=str, default="situation_id")
    parser.add_argument(
        "--cot_cowinner_column",
        type=str,
        default="chosen_expected",
        help="Column identifying selected co-winner label in modified CoT file.",
    )
    parser.add_argument(
        "--train_situations",
        type=int,
        default=500,
        help="Target number of situations to sample per fine-tuning run.",
    )
    parser.add_argument(
        "--modified_completion_pcts",
        type=str,
        dest="modified_completion_pcts",
        default="40",
        help=(
            "Comma-separated target percentages of modified completions/examples "
            "for fine-tuning runs."
        ),
    )
    parser.add_argument(
        "--modified_situation_pcts",
        type=str,
        dest="modified_completion_pcts",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--modified_pcts",
        type=str,
        dest="modified_completion_pcts_alias",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible sampling.",
    )

    # Model/training
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model ID used for both training and evaluation.",
    )
    parser.add_argument(
        "--init_adapter_path",
        type=str,
        default=None,
        help=(
            "Optional path to an existing LoRA adapter to continue fine-tuning from. "
            "When provided, LoRA creation args (--lora_*) are ignored."
        ),
    )
    parser.add_argument("--num_train_epochs", type=float, default=4.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument(
        "--allow_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train on collaborator-style CoT responses when available; disable for label-only SFT.",
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help=(
            "Optional cap on number of training examples per fine-tuned variant "
            "while keeping each situation's full example set intact."
        ),
    )
    parser.add_argument(
        "--cot_unmodified_train_examples",
        type=int,
        default=600,
        help=(
            "CoT mode only: exact number of unmodified CoT rows to sample for training. "
            "Whole situations are sampled atomically, so counts must be achievable "
            "without splitting a situation's tied examples."
        ),
    )
    parser.add_argument(
        "--cot_modified_train_examples",
        type=int,
        default=400,
        help=(
            "CoT mode only: exact number of modified CoT rows to sample for training. "
            "Whole situations are sampled atomically, so counts must be achievable "
            "without splitting a situation's tied examples."
        ),
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="no",
        choices=["no", "epoch", "steps"],
        help=(
            "Checkpoint save policy during training. Defaults to 'no' because final-epoch "
            "reporting is now the standard workflow."
        ),
    )
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 training.")
    parser.add_argument("--fp16", action="store_true", help="Enable float16 training.")
    parser.add_argument(
        "--use_4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable 4-bit QLoRA loading (default: enabled when CUDA is available).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable gradient checkpointing (default: enabled).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer/model.",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target module names for LoRA.",
    )

    # Prompt/completion style
    parser.add_argument(
        "--assistant_style",
        type=str,
        default="label_only",
        choices=["label_only", "cot_and_label"],
        help="Assistant completion style during SFT.",
    )
    parser.add_argument(
        "--append_answer_instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize/append the collaborator-style final answer instruction in training prompts.",
    )
    parser.add_argument(
        "--one_label_per_situation",
        action="store_true",
        help="Use one sampled correct label per situation instead of one example per co-winner.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt inserted into training chats before the user prompt.",
    )
    parser.add_argument(
        "--checkpoint_selection_metric",
        type=str,
        default="cooperate_rate",
        choices=["cooperate_rate", "best_cara_rate", "best_linear_rate", "parse_rate"],
        help="Metric used to choose the best saved checkpoint on the first evaluation dataset.",
    )
    parser.add_argument(
        "--checkpoint_parse_rate_floor",
        type=float,
        default=0.95,
        help="Minimum parse rate required before checkpoint-selection metric is applied.",
    )
    parser.add_argument(
        "--select_best_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate saved checkpoints on the first evaluation dataset and use the best one "
            "for final reporting. Disabled by default so runs report final-epoch weights."
        ),
    )

    # Evaluation
    parser.add_argument("--skip_eval", action="store_true", help="Skip calling evaluate.py.")
    parser.add_argument(
        "--run_baseline_eval",
        action="store_true",
        help="Evaluate the base model (no adapter) before fine-tuning runs.",
    )
    parser.add_argument(
        "--run_init_adapter_eval",
        action="store_true",
        help="Evaluate --init_adapter_path before fine-tuning runs.",
    )
    parser.add_argument(
        "--eval_script",
        type=str,
        default="../evaluation/evaluate.py",
        help="Path to the shared evaluation script.",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="medium_stakes_validation",
        help="Built-in dataset alias passed to evaluate.py by default.",
    )
    parser.add_argument(
        "--eval_datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional list of built-in dataset aliases. If provided, each model variant "
            "is evaluated on all of them."
        ),
    )
    parser.add_argument(
        "--eval_dataset_variant",
        type=str,
        default="default",
        help="Dataset variant passed through to evaluate.py when using built-in dataset aliases.",
    )
    parser.add_argument(
        "--eval_custom_csv",
        "--eval_val_csv",
        dest="eval_custom_csv",
        type=str,
        default=None,
        help="Optional custom CSV path passed through to evaluate.py.",
    )
    parser.add_argument(
        "--eval_custom_csvs",
        "--eval_val_csvs",
        dest="eval_custom_csvs",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of custom CSV paths passed through to evaluate.py.",
    )
    parser.add_argument(
        "--eval_custom_name",
        type=str,
        default=None,
        help=(
            "Optional evaluate.py-style split name to use in metadata when --eval_custom_csv "
            "or --eval_custom_csvs is used."
        ),
    )
    parser.add_argument(
        "--eval_backend",
        type=str,
        choices=["transformers", "vllm"],
        default=None,
        help="Optional inference backend override for evaluate.py. Defaults to evaluator default.",
    )
    parser.add_argument(
        "--eval_num_situations",
        type=int,
        default=None,
        help="Optional situation-count override. If omitted, evaluate.py uses its dataset default.",
    )
    parser.add_argument(
        "--eval_temperature",
        type=float,
        default=None,
        help="Optional temperature override. If omitted, evaluate.py uses its canonical default.",
    )
    parser.add_argument(
        "--eval_top_p",
        type=float,
        default=None,
        help="Optional top-p override. If omitted, evaluate.py uses its default.",
    )
    parser.add_argument(
        "--eval_top_k",
        type=int,
        default=None,
        help="Optional top-k override. If omitted, evaluate.py uses its default.",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=None,
        help="Optional sampling seed override. If omitted, evaluate.py uses its default.",
    )
    parser.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=None,
        help="Optional max_new_tokens override. If omitted, evaluate.py uses its default.",
    )
    parser.add_argument(
        "--eval_reasoning_max_tokens",
        type=int,
        default=None,
        help="Optional reasoning_max_tokens override. If omitted, evaluate.py uses its default.",
    )
    parser.add_argument("--eval_disable_thinking", action="store_true")
    parser.add_argument("--eval_no_save_responses", action="store_true")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Optional evaluation batch-size override. If omitted, evaluate.py uses its default.",
    )
    parser.add_argument(
        "--eval_max_time_per_generation",
        type=float,
        default=None,
        help="Optional max_time_per_generation override. If omitted, evaluate.py uses its default.",
    )
    parser.add_argument(
        "--fail_on_eval_error",
        action="store_true",
        help=(
            "Abort the entire run when evaluate.py fails. "
            "Default behavior is to record eval_error in summary and continue."
        ),
    )

    # Output
    parser.add_argument(
        "--output_root",
        type=str,
        default="training_runs",
        help="Directory where run folders and summaries are written.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name. Default: auto timestamp.",
    )

    args = parser.parse_args()

    # Mirror evaluate.py: auto-disable system prompt for model families that don't
    # support a system role (e.g., gemma-3-12b). Honors explicit --system_prompt overrides.
    if args.system_prompt == DEFAULT_SYSTEM_PROMPT and model_uses_no_system_prompt(args.base_model):
        print(f"Auto: base_model {args.base_model!r} does not use a system prompt; clearing --system_prompt.")
        args.system_prompt = ""

    if args.fp16 and args.bf16:
        raise ValueError("Choose at most one of --fp16 or --bf16.")
    if not args.fp16 and not args.bf16:
        if _gpu_supports_bf16():
            args.bf16 = True
            print("Auto precision: bf16")
        else:
            args.fp16 = True
            print("Auto precision: fp16")

    if args.use_4bit is None:
        args.use_4bit = _cuda_available()
        if args.use_4bit:
            print("Auto setting: use_4bit=True")
    if args.use_4bit and not _cuda_available():
        print("Note: disabling use_4bit because CUDA is not available.")
        args.use_4bit = False

    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = True

    if args.init_adapter_path:
        init_adapter_path = Path(args.init_adapter_path).expanduser().resolve()
        if not init_adapter_path.exists():
            raise FileNotFoundError(f"--init_adapter_path not found: {init_adapter_path}")
        args.init_adapter_path = str(init_adapter_path)

    if args.run_init_adapter_eval and args.skip_eval:
        raise ValueError("--run_init_adapter_eval cannot be used together with --skip_eval.")
    if args.run_init_adapter_eval and not args.init_adapter_path:
        raise ValueError("--run_init_adapter_eval requires --init_adapter_path.")
    if args.select_best_checkpoint and args.save_strategy == "no":
        raise ValueError(
            "--select_best_checkpoint requires checkpoint files, so use "
            "--save_strategy epoch or --save_strategy steps."
        )

    raw_modified_completion_pcts = args.modified_completion_pcts
    raw_modified_pcts_alias = args.modified_completion_pcts_alias
    if raw_modified_completion_pcts is not None and raw_modified_pcts_alias is not None:
        if str(raw_modified_completion_pcts).strip() != str(raw_modified_pcts_alias).strip():
            raise ValueError(
                "--modified_completion_pcts and its alias forms were both provided "
                "with different values. Use only --modified_completion_pcts."
            )
    resolved_modified_completion_pcts = (
        raw_modified_completion_pcts
        if raw_modified_completion_pcts is not None
        else raw_modified_pcts_alias
    )
    if resolved_modified_completion_pcts is None:
        resolved_modified_completion_pcts = "0,10,25,50,100"
    args.modified_completion_pcts = resolved_modified_completion_pcts
    args.modified_completion_pcts_alias = resolved_modified_completion_pcts

    cot_exact_vals = (args.cot_unmodified_train_examples, args.cot_modified_train_examples)
    if any(v is not None for v in cot_exact_vals):
        if args.cot_unmodified_train_examples is None or args.cot_modified_train_examples is None:
            raise ValueError(
                "Provide both --cot_unmodified_train_examples and --cot_modified_train_examples "
                "when using exact CoT example-count sampling."
            )
        if args.cot_unmodified_train_examples < 0 or args.cot_modified_train_examples < 0:
            raise ValueError("Exact CoT example counts must be >= 0.")
        exact_total = args.cot_unmodified_train_examples + args.cot_modified_train_examples
        if exact_total <= 0:
            raise ValueError("Exact CoT example counts must sum to > 0.")
        if args.max_train_examples is not None and args.max_train_examples < exact_total:
            raise ValueError(
                "--max_train_examples is smaller than the requested exact CoT example mix. "
                f"Got max_train_examples={args.max_train_examples}, exact_total={exact_total}."
            )

    return args


def parse_modified_completion_pcts(raw: str) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        pct = float(part)
        if pct < 0 or pct > 100:
            raise ValueError(f"Modified percentage must be in [0, 100], got {pct}")
        if pct in seen:
            continue
        seen.add(pct)
        out.append(pct)
    return out


def parse_modified_pcts(raw: str) -> list[float]:
    return parse_modified_completion_pcts(raw)


def dataset_tag_from_path(path: str) -> str:
    stem = Path(path).stem
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return tag or "dataset"


def normalize_split_name(name: str) -> str:
    split_name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name).strip()).strip("-")
    return split_name or "custom"


def infer_custom_eval_name(path: Path, args: argparse.Namespace) -> str | None:
    explicit_name = (args.eval_custom_name or "").strip()
    if explicit_name:
        return explicit_name

    modified_path = Path(args.modified_cot_data).expanduser().resolve() if args.modified_cot_data else None
    if modified_path is not None and path == modified_path:
        return DEFAULT_MODIFIED_COT_SPLIT_NAME
    return None


def resolve_eval_datasets(args: argparse.Namespace) -> list[dict]:
    if args.eval_custom_csvs:
        raw_paths = args.eval_custom_csvs
    elif args.eval_custom_csv:
        raw_paths = [args.eval_custom_csv]
    else:
        raw_paths = args.eval_datasets if args.eval_datasets else [args.eval_dataset]

    datasets = []
    seen = set()
    using_custom_csvs = bool(args.eval_custom_csv or args.eval_custom_csvs)

    for raw in raw_paths:
        if using_custom_csvs:
            p = Path(raw).expanduser().resolve()
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            if not p.exists():
                raise FileNotFoundError(f"Evaluation CSV not found: {p}")
            custom_name = infer_custom_eval_name(p, args)
            datasets.append(
                {
                    "dataset": custom_name or "custom",
                    "dataset_variant": "custom",
                    "custom_csv": str(p),
                    "tag": normalize_split_name(custom_name) if custom_name else dataset_tag_from_path(str(p)),
                    "display": custom_name or str(p),
                }
            )
        else:
            dataset_alias = str(raw).strip()
            key = f"{dataset_alias}::{args.eval_dataset_variant}"
            if key in seen:
                continue
            seen.add(key)
            datasets.append(
                {
                    "dataset": dataset_alias,
                    "dataset_variant": args.eval_dataset_variant,
                    "custom_csv": None,
                    "tag": re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_alias).strip("_") or "dataset",
                    "display": dataset_alias,
                }
            )
    if not datasets:
        raise ValueError("No evaluation datasets resolved.")
    return datasets


def load_cot_examples(
    path: Path,
    prompt_col: str,
    completion_col: str,
    situation_col: str,
    cowinner_col: str | None = None,
    dedupe_one_per_cowinner: bool = False,
) -> pd.DataFrame:
    df = read_tabular(path)
    required = [situation_col, prompt_col, completion_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required CoT columns in {path}: {missing}. "
            f"Found: {list(df.columns)}"
        )
    df = filter_bad_cot_situations(df, path=path, situation_col=situation_col)

    keep_cols = [situation_col, prompt_col, completion_col]
    has_cowinner = bool(cowinner_col) and (cowinner_col in df.columns)
    if has_cowinner:
        keep_cols.append(cowinner_col)

    cot = df[keep_cols].copy()
    cot = cot.rename(
        columns={
            situation_col: "situation_id",
            prompt_col: "prompt",
            completion_col: "completion",
        }
    )
    if has_cowinner:
        cot = cot.rename(columns={cowinner_col: "co_winner_label"})
    else:
        cot["co_winner_label"] = None

    cot = cot.dropna(subset=["situation_id", "prompt", "completion"])
    cot["situation_id"] = cot["situation_id"].astype(int)
    cot["prompt"] = cot["prompt"].map(unescape_cot).astype(str).str.strip()
    cot["completion"] = cot["completion"].map(unescape_cot).astype(str).str.strip()
    cot = cot[(cot["prompt"] != "") & (cot["completion"] != "")]

    if dedupe_one_per_cowinner:
        if has_cowinner:
            cot["co_winner_label"] = cot["co_winner_label"].astype(str).str.strip()
            cot = cot.drop_duplicates(subset=["situation_id", "co_winner_label"], keep="first")
        else:
            cot = cot.drop_duplicates(subset=["situation_id", "completion"], keep="first")

    return cot.reset_index(drop=True)


def ensure_training_dependencies() -> None:
    global torch
    global Dataset
    global LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
    global AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    global DataCollatorForLanguageModeling
    global SFTTrainer, SFTConfig

    try:
        import torch as _torch
        from datasets import Dataset as _Dataset
        from peft import (
            LoraConfig as _LoraConfig,
            PeftModel as _PeftModel,
            get_peft_model as _get_peft_model,
            prepare_model_for_kbit_training as _prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM as _AutoModelForCausalLM,
            AutoTokenizer as _AutoTokenizer,
            BitsAndBytesConfig as _BitsAndBytesConfig,
            DataCollatorForLanguageModeling as _DataCollatorForLanguageModeling,
            TrainingArguments as _TrainingArguments,
        )
        from trl import SFTTrainer as _SFTTrainer
        try:
            from trl import SFTConfig as _SFTConfig
        except ImportError:
            _SFTConfig = None
    except ModuleNotFoundError as exc:
        pkg = getattr(exc, "name", "unknown package")
        raise SystemExit(
            "Missing training dependency: "
            f"{pkg}. Install training extras first, e.g.:\n"
            "pip install torch transformers peft accelerate datasets trl bitsandbytes openpyxl"
        ) from exc

    torch = _torch
    Dataset = _Dataset
    LoraConfig = _LoraConfig
    get_peft_model = _get_peft_model
    PeftModel = _PeftModel
    prepare_model_for_kbit_training = _prepare_model_for_kbit_training
    AutoModelForCausalLM = _AutoModelForCausalLM
    AutoTokenizer = _AutoTokenizer
    BitsAndBytesConfig = _BitsAndBytesConfig
    DataCollatorForLanguageModeling = _DataCollatorForLanguageModeling
    TrainingArguments = _TrainingArguments
    SFTTrainer = _SFTTrainer
    SFTConfig = _SFTConfig


def disable_peft_bitsandbytes_dispatch() -> None:
    """
    Force PEFT to skip bitsandbytes LoRA backends when this run is not using
    4-bit loading.

    PEFT 0.19.0 checks whether bitsandbytes is importable, then eagerly imports
    its LoRA dispatchers even for plain bf16/fp16 LoRA runs. On Engaging the
    installed bitsandbytes build is not usable on the GPU nodes, so we patch the
    relevant PEFT modules to report bnb as unavailable for this process.
    """

    def _false() -> bool:
        return False

    patched: list[str] = []
    module_attr_pairs = [
        ("peft.import_utils", ("is_bnb_available", "is_bnb_4bit_available")),
        ("peft.tuners.lora", ("is_bnb_available", "is_bnb_4bit_available")),
        ("peft.tuners.lora.model", ("is_bnb_available", "is_bnb_4bit_available")),
    ]
    for module_name, attr_names in module_attr_pairs:
        try:
            module = __import__(module_name, fromlist=["*"])
        except Exception:
            continue
        for attr_name in attr_names:
            if hasattr(module, attr_name):
                setattr(module, attr_name, _false)
                patched.append(f"{module_name}.{attr_name}")

    if patched:
        print("Disabled PEFT bitsandbytes dispatch for this non-4bit run:")
        for name in patched:
            print(f"  - {name}")


def normalize_peft_upcast_dtypes() -> None:
    """
    Align PEFT's float8 upcast list with the actual Torch build in this env.

    PEFT 0.19.0 includes torch.float8_e8m0fnu in UPCAST_DTYPES, but torch
    2.6.0+cu124 on Engaging does not expose that dtype symbol. Filter the list
    once at runtime so adapter casting logic only touches dtypes that exist.
    """

    if torch is None:
        return

    try:
        import peft.tuners.tuners_utils as peft_tuners_utils
    except Exception:
        return

    current = tuple(getattr(peft_tuners_utils, "UPCAST_DTYPES", ()))
    normalized = tuple(name for name in current if hasattr(torch, name))
    if normalized == current:
        return

    peft_tuners_utils.UPCAST_DTYPES = normalized
    for module_name in ("peft.utils", "peft.utils.constants"):
        try:
            module = __import__(module_name, fromlist=["*"])
        except Exception:
            continue
        if hasattr(module, "UPCAST_DTYPES"):
            setattr(module, "UPCAST_DTYPES", normalized)

    print("Normalized PEFT UPCAST_DTYPES for this Torch build:")
    print(f"  - before: {current}")
    print(f"  - after:  {normalized}")


def read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type for {path}. Use CSV or Excel.")


def sanitize_token(token: str) -> str:
    token = token.strip()
    if token.isalpha():
        return token.lower()
    return token


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def parse_label_tokens(value) -> list[str]:
    """Parse labels from many storage forms: JSON list, 'a, b', '(1)', etc."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []

    values: list[str] = []
    if isinstance(value, (list, tuple, set)):
        values = [str(v) for v in value]
    else:
        raw = str(value).strip()
        if not raw or raw.lower() in {"nan", "none", "null", "[]"}:
            return []
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
                break
            except Exception:
                parsed = None
        if isinstance(parsed, list):
            values = [str(v) for v in parsed]
        elif parsed is not None and not isinstance(parsed, (dict, tuple, set)):
            values = [str(parsed)]
        else:
            values = [raw]

    tokens: list[str] = []
    for item in values:
        for tok in re.findall(r"(?<![A-Za-z0-9])(?:\d+|[A-Za-z])(?![A-Za-z0-9])", str(item)):
            tokens.append(sanitize_token(tok))
    return unique_preserve_order(tokens)


def parse_ok_flag(value) -> bool:
    """Interpret CoT audit flags; only explicit false values should fail a row."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    if text in {"false", "0", "no"}:
        return False
    if text in {"true", "1", "yes", ""}:
        return True
    return bool(value)


def filter_bad_cot_situations(
    df: pd.DataFrame,
    path: Path,
    situation_col: str,
) -> pd.DataFrame:
    """
    Drop entire situations if any row is explicitly marked bad, or if the tied
    completion set is incomplete for that situation.
    """
    if situation_col not in df.columns:
        return df

    bad_ids: set[object] = set()

    for ok_col in ("chosen_ok", "rejected_ok"):
        if ok_col in df.columns:
            bad_mask = ~df[ok_col].map(parse_ok_flag)
            bad_ids.update(df.loc[bad_mask, situation_col].dropna().tolist())

    completeness_cols = {"chosen_expected", "all_tied_labels", "num_tied_options"}
    if completeness_cols.issubset(df.columns):
        for situation_id, group in df.groupby(situation_col, sort=False):
            expected_count_values = group["num_tied_options"].dropna().astype(int).unique().tolist()
            expected_count = expected_count_values[0] if expected_count_values else None
            expected_labels = sorted(parse_label_tokens(group.iloc[0]["all_tied_labels"]))
            seen_labels = sorted(
                {
                    sanitize_token(str(label))
                    for label in group["chosen_expected"].dropna().tolist()
                    if str(label).strip()
                }
            )
            if expected_count is not None and len(group) != expected_count:
                bad_ids.add(situation_id)
                continue
            if expected_labels and seen_labels != expected_labels:
                bad_ids.add(situation_id)

    if not bad_ids:
        return df

    filtered = df.loc[~df[situation_col].isin(bad_ids)].copy()
    print(
        f"Filtered out {len(bad_ids)} bad CoT situations from {path.name}; "
        f"kept {filtered[situation_col].nunique()} situations / {len(filtered)} rows."
    )
    return filtered


def parse_old_index_label(correct_label, labels_vector) -> list[str]:
    """Handle old format where correct_label is -k indexing into labels_vector."""
    try:
        if isinstance(correct_label, float) and math.isnan(correct_label):
            return []
        idx_val = int(correct_label)
    except Exception:
        return []

    try:
        vec = labels_vector
        if isinstance(vec, str):
            vec = ast.literal_eval(vec)
        if not isinstance(vec, (list, tuple)) or not vec:
            return []
        idx = abs(idx_val) - 1
        if idx < 0 or idx >= len(vec):
            return []
        return parse_label_tokens(vec[idx])
    except Exception:
        return []


def extract_correct_labels(row: pd.Series) -> list[str]:
    """Pick the best available correct-label field from a row."""
    # Preferred newer columns
    for col in ("CARA_correct_labels", "cooperate_correct_labels"):
        if col in row and pd.notna(row[col]):
            labels = parse_label_tokens(row[col])
            if labels:
                return labels

    # Older format: index into labels_vector
    if "correct_label" in row:
        labels = parse_old_index_label(row["correct_label"], row.get("labels_vector"))
        if labels:
            return labels
        labels = parse_label_tokens(row["correct_label"])
        if labels:
            return labels

    # Fallback
    if "answer_word" in row and pd.notna(row["answer_word"]):
        labels = parse_label_tokens(row["answer_word"])
        if labels:
            return labels

    return []


def unescape_cot(text: str | None) -> str | None:
    """Decode literal backslash escapes found in prompt/CoT CSV exports."""
    if text is None:
        return None
    return (
        str(text)
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace('\\"', '"')
    )


def modify_prompt_for_training(prompt_text: str, allow_thinking: bool, append_answer_instruction: bool) -> str:
    """
    Match the collaborator notebook's prompt style more closely.

    We normalize the final answer instruction so the trained model sees a stable
    terminal requirement across vanilla SFT and tie-training runs.
    """
    prompt = str(unescape_cot(prompt_text) or "").strip()

    canonical_suffix = (
        "You can think if you want, but you must end your answer with the letter/number "
        "of your chosen option."
        if allow_thinking
        else "Output the label of your chosen option only."
    )
    if not append_answer_instruction:
        return prompt

    patterns = [
        r"\s*You can think before answering,.*?would select\.\s*$",
        r"\s*You can think about which you'd choose,.*?would choose\.\s*$",
        r"\s*You can think if you want,.*?chosen option\.\s*$",
        r"\s*Output the label of your chosen option only\.\s*$",
    ]
    updated = prompt
    for pattern in patterns:
        updated = re.sub(pattern, "", updated, flags=re.IGNORECASE | re.DOTALL)
    updated = updated.rstrip()
    if canonical_suffix.lower() in updated.lower():
        return updated
    return f"{updated}\n\n{canonical_suffix}"


def build_situations_df(
    raw_df: pd.DataFrame,
    allow_thinking: bool,
    append_answer_instruction: bool,
) -> pd.DataFrame:
    if "situation_id" not in raw_df.columns or "prompt_text" not in raw_df.columns:
        raise ValueError(
            "Dataset must contain at least 'situation_id' and 'prompt_text' columns. "
            f"Found: {list(raw_df.columns)}"
        )

    records = []
    for situation_id, group in raw_df.groupby("situation_id", sort=False):
        row = group.iloc[0]
        correct_labels = extract_correct_labels(row)
        if not correct_labels:
            continue
        prompt = modify_prompt_for_training(
            row["prompt_text"],
            allow_thinking=allow_thinking,
            append_answer_instruction=append_answer_instruction,
        )
        records.append(
            {
                "situation_id": int(situation_id),
                "prompt": prompt,
                "correct_labels": correct_labels,
            }
        )

    if not records:
        raise ValueError("No situation-level records with valid correct labels were found.")
    return pd.DataFrame(records)


@dataclass
class SampledSplit:
    sampled_df: pd.DataFrame
    n_modified: int
    n_unmodified: int
    actual_modified_pct_situations: float


def sample_situations(
    unmodified_df: pd.DataFrame,
    modified_df: pd.DataFrame,
    train_situations: int,
    modified_pct: float,
    seed: int,
) -> SampledSplit:
    if train_situations <= 0:
        raise ValueError("--train_situations must be > 0")

    p = float(modified_pct)
    p = max(0.0, min(100.0, p))

    if p == 0.0:
        n_mod = 0
        n_unmod = min(train_situations, len(unmodified_df))
    elif p == 100.0:
        n_mod = min(train_situations, len(modified_df))
        n_unmod = 0
    else:
        max_total_by_mod = len(modified_df) * 100.0 / p
        max_total_by_unmod = len(unmodified_df) * 100.0 / (100.0 - p)
        feasible_total = int(math.floor(min(train_situations, max_total_by_mod, max_total_by_unmod)))
        if feasible_total <= 0:
            raise ValueError(
                "No feasible sample for requested modified percentage. "
                f"modified_pct={modified_pct}, "
                f"available modified={len(modified_df)}, available unmodified={len(unmodified_df)}"
            )
        n_mod = int(round(feasible_total * p / 100.0))
        n_mod = min(n_mod, len(modified_df))
        n_unmod = feasible_total - n_mod
        n_unmod = min(n_unmod, len(unmodified_df))
        # Keep counts consistent if clipping occurred.
        feasible_total = n_mod + n_unmod
        n_mod = int(round(feasible_total * p / 100.0))
        n_mod = min(n_mod, len(modified_df))
        n_unmod = feasible_total - n_mod

    sampled_parts: list[pd.DataFrame] = []
    if n_unmod > 0:
        sampled_parts.append(unmodified_df.sample(n=n_unmod, random_state=seed))
    if n_mod > 0:
        sampled_parts.append(modified_df.sample(n=n_mod, random_state=seed + 1000))

    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame(columns=unmodified_df.columns)
    if sampled.empty:
        raise ValueError("Sampling produced zero situations.")

    sampled = sampled.sample(frac=1.0, random_state=seed + 2000).reset_index(drop=True)
    actual_mod_pct_situations = 100.0 * n_mod / len(sampled)

    return SampledSplit(
        sampled_df=sampled,
        n_modified=n_mod,
        n_unmodified=n_unmod,
        actual_modified_pct_situations=actual_mod_pct_situations,
    )


def build_assistant_completion(
    chosen_label: str,
    all_correct_labels: list[str],
    style: str,
    allow_thinking: bool,
    cot_text: str | None = None,
) -> str:
    if cot_text and allow_thinking:
        cot = unescape_cot(cot_text)
        cot = re.sub(
            r'\{["\']?answer["\']?\s*:\s*["\']?[^}]+["\']?\}\s*$',
            "",
            cot or "",
            flags=re.DOTALL,
        ).rstrip()
        return f"{cot}\n\nFINAL ANSWER: {chosen_label}"

    if not allow_thinking and style == "label_only":
        return chosen_label

    if len(all_correct_labels) == 1:
        rationale = (
            "Let me evaluate the options using CARA utility with alpha=0.01. "
            f"After comparing expected utility, option {chosen_label} is best."
        )
    else:
        tied = ", ".join(x.upper() if x.isalpha() else x for x in all_correct_labels)
        rationale = (
            "Let me evaluate the options using CARA utility with alpha=0.01. "
            f"Options {tied} are tied for best expected utility, so I will choose one valid best option."
        )
    if allow_thinking or style == "cot_and_label":
        return f"{rationale}\n\nFINAL ANSWER: {chosen_label}"
    return chosen_label


def format_chat_example_for_sft(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    prompt: str,
    assistant_text: str,
) -> dict:
    prompt_messages = []
    if system_prompt:
        prompt_messages.append({"role": "system", "content": system_prompt})
    prompt_messages.append({"role": "user", "content": prompt})
    prompt_only_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_len = len(tokenizer(prompt_only_text, add_special_tokens=False)["input_ids"])

    full_messages = list(prompt_messages)
    full_messages.append({"role": "assistant", "content": assistant_text})
    text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text, "prompt_len": prompt_token_len}


def build_prompt_masking_collator(tokenizer):
    """Create a data collator that masks prompt tokens from the loss."""

    class _PromptMaskingCollator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            prompt_lens = [int(feature.get("prompt_len", 0)) for feature in features]
            sanitized_features = []
            for feature in features:
                sanitized = {}
                for key, value in feature.items():
                    if key == "prompt_len":
                        continue
                    if isinstance(value, str):
                        continue
                    sanitized[key] = value
                sanitized_features.append(sanitized)

            batch = super().__call__(sanitized_features)
            labels = batch["labels"].clone()
            for idx, prompt_len in enumerate(prompt_lens):
                labels[idx, : min(prompt_len, labels.shape[1])] = -100
            batch["labels"] = labels
            return batch

    return _PromptMaskingCollator(tokenizer=tokenizer, mlm=False)


def reorder_examples_to_reduce_adjacent_same_situation(
    examples: list[dict], seed: int
) -> list[dict]:
    """
    Reorder examples so adjacent entries usually have different situation_id.

    When a no-adjacent arrangement is mathematically possible, this greedy
    heap scheduler will produce one. If not possible (e.g. one situation has
    > half the examples), it still minimizes immediate repeats.
    """
    if len(examples) <= 1:
        return examples

    grouped: dict[int, list[dict]] = {}
    for ex in examples:
        sid = int(ex["situation_id"])
        grouped.setdefault(sid, []).append(ex)

    rng = random.Random(seed)
    for sid_examples in grouped.values():
        rng.shuffle(sid_examples)

    # Max-heap by remaining count; random tie-break for stable stochasticity.
    heap: list[tuple[int, float, int]] = []
    for sid, sid_examples in grouped.items():
        heapq.heappush(heap, (-len(sid_examples), rng.random(), sid))

    ordered: list[dict] = []
    prev_sid: int | None = None

    while heap:
        count1, tie1, sid1 = heapq.heappop(heap)

        if sid1 == prev_sid and heap:
            count2, _, sid2 = heapq.heappop(heap)
            ordered.append(grouped[sid2].pop())
            prev_sid = sid2
            count2 += 1  # negative counts move toward zero
            if count2 < 0:
                heapq.heappush(heap, (count2, rng.random(), sid2))
            heapq.heappush(heap, (count1, tie1, sid1))
            continue

        ordered.append(grouped[sid1].pop())
        prev_sid = sid1
        count1 += 1
        if count1 < 0:
            heapq.heappush(heap, (count1, rng.random(), sid1))

    return ordered


def build_training_examples(
    tokenizer: AutoTokenizer,
    sampled_df: pd.DataFrame,
    assistant_style: str,
    one_label_per_situation: bool,
    allow_thinking: bool,
    system_prompt: str,
    seed: int,
) -> list[dict]:
    records: list[dict] = []
    for idx, row in sampled_df.iterrows():
        prompt = row["prompt"]
        situation_id = int(row["situation_id"])
        source = str(row.get("_source", "unmodified")).strip().lower() or "unmodified"
        correct_labels = [sanitize_token(x) for x in row["correct_labels"]]
        correct_labels = unique_preserve_order(correct_labels)
        if not correct_labels:
            continue

        if one_label_per_situation:
            label_ix = (seed + idx) % len(correct_labels)
            labels_for_examples = [correct_labels[label_ix]]
        else:
            labels_for_examples = correct_labels

        for chosen_label in labels_for_examples:
            assistant_text = build_assistant_completion(
                chosen_label=chosen_label,
                all_correct_labels=correct_labels,
                style=assistant_style,
                allow_thinking=allow_thinking,
            )
            formatted = format_chat_example_for_sft(
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                prompt=prompt,
                assistant_text=assistant_text,
            )
            records.append(
                {
                    "situation_id": situation_id,
                    "_source": source,
                    "target_label": chosen_label,
                    "co_winner_label": None,
                    **formatted,
                }
            )

    if not records:
        raise ValueError("No training examples were produced from sampled situations.")
    ordered = reorder_examples_to_reduce_adjacent_same_situation(records, seed=seed)
    return [
        {
            "text": r["text"],
            "prompt_len": r["prompt_len"],
            "_source": r["_source"],
            "situation_id": r["situation_id"],
            "target_label": r["target_label"],
            "co_winner_label": r["co_winner_label"],
        }
        for r in ordered
    ]


def build_training_examples_from_cot(
    tokenizer: AutoTokenizer,
    cot_examples_df: pd.DataFrame,
    allow_thinking: bool,
    system_prompt: str,
    append_answer_instruction: bool,
    seed: int,
) -> list[dict]:
    records: list[dict] = []
    shuffled = cot_examples_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    for _, row in shuffled.iterrows():
        source = str(row.get("_source", "unmodified")).strip().lower() or "unmodified"
        chosen_label = sanitize_token(row.get("co_winner_label"))
        if not chosen_label:
            parsed = parse_label_tokens(row.get("co_winner_label"))
            chosen_label = parsed[0] if parsed else ""
        assistant_text = build_assistant_completion(
            chosen_label=chosen_label,
            all_correct_labels=[chosen_label] if chosen_label else [],
            style="cot_and_label" if allow_thinking else "label_only",
            allow_thinking=allow_thinking,
            cot_text=row["completion"],
        )
        prompt = modify_prompt_for_training(
            row["prompt"],
            allow_thinking=allow_thinking,
            append_answer_instruction=append_answer_instruction,
        )
        formatted = format_chat_example_for_sft(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            prompt=prompt,
            assistant_text=assistant_text,
        )
        records.append(
            {
                "situation_id": int(row["situation_id"]),
                "_source": source,
                "target_label": chosen_label,
                "co_winner_label": chosen_label,
                **formatted,
            }
        )
    if not records:
        raise ValueError("No CoT-based training examples were produced.")
    ordered = reorder_examples_to_reduce_adjacent_same_situation(records, seed=seed + 17)
    return [
        {
            "text": r["text"],
            "prompt_len": r["prompt_len"],
            "_source": r["_source"],
            "situation_id": r["situation_id"],
            "target_label": r["target_label"],
            "co_winner_label": r["co_winner_label"],
        }
        for r in ordered
    ]


def sample_cot_examples_exact_pool(
    cot_examples: pd.DataFrame,
    n_examples: int,
    seed: int,
    source_name: str,
) -> pd.DataFrame:
    """
    Sample an exact number of CoT rows while keeping whole situations intact.

    If a situation contributes multiple tied examples, either all of those rows
    are selected or none of them are.
    """
    if n_examples < 0:
        raise ValueError(f"Requested negative {source_name} CoT rows: {n_examples}")
    if n_examples == 0:
        empty = cot_examples.iloc[0:0].copy()
        empty["_source"] = source_name
        return empty
    if n_examples > len(cot_examples):
        raise ValueError(
            f"Requested more {source_name} CoT rows than available. "
            f"requested={n_examples}, available={len(cot_examples)}"
        )

    grouped = [
        (int(situation_id), group.copy())
        for situation_id, group in cot_examples.groupby("situation_id", sort=False)
    ]
    rng = random.Random(seed)
    rng.shuffle(grouped)
    group_sizes = [len(group) for _, group in grouped]

    reachable: dict[int, tuple[int, int] | None] = {0: None}
    for idx, size in enumerate(group_sizes):
        for total in sorted(list(reachable.keys()), reverse=True):
            new_total = total + size
            if new_total > n_examples or new_total in reachable:
                continue
            reachable[new_total] = (total, idx)
        if n_examples in reachable:
            break

    if n_examples not in reachable:
        unique_sizes = sorted(set(group_sizes))
        raise ValueError(
            "Exact CoT example-count sampling could not satisfy the requested count "
            f"while keeping full tied situations intact for {source_name} rows. "
            f"requested={n_examples}, available={len(cot_examples)}, "
            f"group_sizes={unique_sizes}"
        )

    selected_indices: list[int] = []
    total = n_examples
    while total != 0:
        prev_total, idx = reachable[total]
        selected_indices.append(idx)
        total = prev_total
    selected_indices.reverse()

    selected = pd.concat([grouped[idx][1] for idx in selected_indices], ignore_index=True)
    selected["_source"] = source_name
    return selected


def compute_atomic_group_sizes(cot_examples: pd.DataFrame) -> list[int]:
    if cot_examples.empty:
        return []
    return [
        int(size)
        for size in cot_examples.groupby("situation_id", sort=False).size().astype(int).tolist()
    ]


def compute_feasible_atomic_totals(group_sizes: list[int]) -> list[int]:
    reachable = {0}
    for size in group_sizes:
        prior = list(reachable)
        for total in prior:
            reachable.add(total + size)
    return sorted(reachable)


def nearest_feasible_totals(requested: int, feasible_totals: list[int], limit: int = 6) -> list[int]:
    candidates = [total for total in feasible_totals if total > 0 and total != requested]
    return sorted(candidates, key=lambda total: (abs(total - requested), total))[:limit]


def validate_exact_cot_example_request(
    cot_examples: pd.DataFrame,
    requested_examples: int,
    source_name: str,
) -> None:
    if requested_examples < 0:
        raise ValueError(f"Requested negative {source_name} CoT rows: {requested_examples}")
    if requested_examples == 0:
        print(f"Exact-count preflight ({source_name}): requested=0 rows; skipping.")
        return

    available_examples = len(cot_examples)
    if requested_examples > available_examples:
        raise ValueError(
            f"Requested exact {source_name} CoT rows exceed available rows. "
            f"requested={requested_examples}, available={available_examples}"
        )

    group_sizes = compute_atomic_group_sizes(cot_examples)
    feasible_totals = compute_feasible_atomic_totals(group_sizes)
    if requested_examples not in feasible_totals:
        nearby = nearest_feasible_totals(requested_examples, feasible_totals)
        nearby_text = ", ".join(str(total) for total in nearby) if nearby else "none"
        raise ValueError(
            "Requested exact CoT row count is infeasible while keeping whole tied situations intact. "
            f"source={source_name}, requested={requested_examples}, available={available_examples}, "
            f"nearby_feasible_totals=[{nearby_text}], "
            f"unique_group_sizes={sorted(set(group_sizes))}"
        )

    print(
        f"Exact-count preflight OK ({source_name}): "
        f"requested={requested_examples}, available={available_examples}, "
        f"atomic_groups={len(group_sizes)}"
    )


def sample_cot_examples_exact(
    unmodified_cot_examples: pd.DataFrame,
    modified_cot_examples: pd.DataFrame,
    n_unmodified_examples: int,
    n_modified_examples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    if n_unmodified_examples > 0:
        unmod = sample_cot_examples_exact_pool(
            cot_examples=unmodified_cot_examples,
            n_examples=n_unmodified_examples,
            seed=seed,
            source_name="unmodified",
        )
        parts.append(unmod)
    if n_modified_examples > 0:
        mod = sample_cot_examples_exact_pool(
            cot_examples=modified_cot_examples,
            n_examples=n_modified_examples,
            seed=seed + 1000,
            source_name="modified",
        )
        parts.append(mod)

    if not parts:
        raise ValueError("Exact CoT example sampling requested zero rows.")

    selected = pd.concat(parts, ignore_index=True)
    selected = selected.sample(frac=1.0, random_state=seed + 2000).reset_index(drop=True)
    sampled_situations = (
        selected[["situation_id", "_source"]]
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed + 3000)
        .reset_index(drop=True)
    )
    return selected, sampled_situations


def load_tokenizer(base_model: str, trust_remote_code: bool) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_for_training(args: argparse.Namespace):
    if args.use_4bit:
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if getattr(model, "config", None) is not None and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if args.init_adapter_path:
        print(f"Initializing training from existing adapter: {args.init_adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            args.init_adapter_path,
            is_trainable=True,
        )
        return model

    lora_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


def evaluate_with_shared_script(
    args: argparse.Namespace,
    eval_script_path: Path,
    eval_dataset: dict,
    output_json: Path,
    adapter_path: Path | None,
) -> dict | None:
    cmd = [
        sys.executable,
        str(eval_script_path),
        "--base_model",
        args.base_model,
        "--output",
        str(output_json),
    ]
    if args.eval_backend is not None:
        cmd += ["--backend", args.eval_backend]
    if adapter_path is not None:
        cmd += ["--model_path", str(adapter_path)]
    if eval_dataset["custom_csv"] is not None:
        cmd += ["--custom_csv", eval_dataset["custom_csv"]]
    else:
        cmd += ["--dataset", eval_dataset["dataset"]]
        if eval_dataset["dataset_variant"] != "default":
            cmd += ["--dataset_variant", eval_dataset["dataset_variant"]]
    if args.eval_num_situations is not None:
        cmd += ["--num_situations", str(args.eval_num_situations)]
    if args.eval_temperature is not None:
        cmd += ["--temperature", str(args.eval_temperature)]
        if abs(args.eval_temperature - SHARED_EVAL_DEFAULT_TEMPERATURE) > 1e-12:
            cmd.append("--allow_nondefault_temperature")
    if args.eval_top_p is not None:
        cmd += ["--top_p", str(args.eval_top_p)]
    if args.eval_top_k is not None:
        cmd += ["--top_k", str(args.eval_top_k)]
    if args.eval_seed is not None:
        cmd += ["--seed", str(args.eval_seed)]
    if args.eval_max_new_tokens is not None:
        cmd += ["--max_new_tokens", str(args.eval_max_new_tokens)]
    if args.eval_reasoning_max_tokens is not None:
        cmd += ["--reasoning_max_tokens", str(args.eval_reasoning_max_tokens)]
    if args.eval_batch_size is not None:
        cmd += ["--batch_size", str(args.eval_batch_size)]
    if args.eval_max_time_per_generation is not None:
        cmd += ["--max_time_per_generation", str(args.eval_max_time_per_generation)]
    if args.eval_disable_thinking:
        cmd.append("--disable_thinking")
    if args.eval_no_save_responses:
        cmd.append("--no_save_responses")

    print("Running evaluation:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(eval_script_path.parent))

    if not output_json.exists():
        return None
    with open(output_json, "r") as f:
        return json.load(f)


def load_eval_json_if_present(output_json: Path) -> dict | None:
    if not output_json.exists():
        return None
    try:
        with open(output_json, "r") as f:
            return json.load(f)
    except Exception:
        return None


def extract_eval_metrics(eval_json: dict | None) -> dict:
    if not eval_json:
        return {}
    metrics = eval_json.get("metrics", {})
    return {
        "parse_rate": metrics.get("parse_rate"),
        "best_cara_rate": metrics.get("best_cara_rate"),
        "best_linear_rate": metrics.get("best_linear_rate"),
        "cooperate_rate": metrics.get("cooperate_rate"),
        "rebel_rate": metrics.get("rebel_rate"),
        "steal_rate": metrics.get("steal_rate"),
        "num_valid": eval_json.get("num_valid"),
        "num_total": eval_json.get("num_total"),
    }


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except Exception:
        return -1


def checkpoint_passes_parse_floor(metrics: dict | None, parse_floor: float) -> bool:
    if not metrics:
        return False
    return float(metrics.get("parse_rate") or 0.0) >= parse_floor


def choose_best_checkpoint_record(
    checkpoint_records: list[dict],
    metric_name: str,
    parse_floor: float,
) -> tuple[dict, str]:
    if not checkpoint_records:
        raise ValueError("No checkpoint records were provided for selection.")

    def _metric_tuple(record: dict) -> tuple[float, float, float, float, int]:
        metrics = record.get("metrics") or {}
        primary = float(metrics.get(metric_name) or 0.0)
        parse_rate = float(metrics.get("parse_rate") or 0.0)
        cooperate_rate = float(metrics.get("cooperate_rate") or 0.0)
        cara_rate = float(metrics.get("best_cara_rate") or 0.0)
        return (
            primary,
            parse_rate,
            cooperate_rate,
            cara_rate,
            checkpoint_step(Path(record["checkpoint_path"])),
        )

    valid = [
        record
        for record in checkpoint_records
        if checkpoint_passes_parse_floor(record.get("metrics"), parse_floor)
    ]
    if valid:
        return max(valid, key=_metric_tuple), "parse_floor_satisfied"

    def _fallback_tuple(record: dict) -> tuple[float, float, float, float, int]:
        metrics = record.get("metrics") or {}
        return (
            float(metrics.get("parse_rate") or 0.0),
            float(metrics.get(metric_name) or 0.0),
            float(metrics.get("cooperate_rate") or 0.0),
            float(metrics.get("best_cara_rate") or 0.0),
            checkpoint_step(Path(record["checkpoint_path"])),
        )

    return max(checkpoint_records, key=_fallback_tuple), "parse_floor_unmet_fallback"


def summarize_example_sources(examples: list[dict]) -> dict[str, float | int]:
    modified_examples = sum(1 for ex in examples if ex.get("_source") == "modified")
    unmodified_examples = sum(1 for ex in examples if ex.get("_source") == "unmodified")
    other_examples = len(examples) - modified_examples - unmodified_examples
    modified_pct = 100.0 * modified_examples / max(1, len(examples))
    return {
        "modified_examples": int(modified_examples),
        "unmodified_examples": int(unmodified_examples),
        "other_examples": int(other_examples),
        "actual_modified_pct_examples": float(modified_pct),
    }


def cap_examples_by_full_situation(
    examples: list[dict],
    max_examples: int,
    seed: int,
) -> list[dict]:
    """
    Cap training examples without splitting a situation's tied examples.

    This preserves the invariant that a situation contributes either all of its
    examples or none of them.
    """
    if max_examples <= 0:
        raise ValueError("--max_train_examples must be > 0 when provided.")
    if len(examples) <= max_examples:
        return examples

    grouped: dict[int, list[dict]] = {}
    ordered_situations: list[int] = []
    for ex in examples:
        situation_id = int(ex["situation_id"])
        if situation_id not in grouped:
            grouped[situation_id] = []
            ordered_situations.append(situation_id)
        grouped[situation_id].append(ex)

    sizes = [len(grouped[situation_id]) for situation_id in ordered_situations]
    reachable: dict[int, tuple[int, int] | None] = {0: None}
    for idx, size in enumerate(sizes):
        for total in sorted(list(reachable.keys()), reverse=True):
            new_total = total + size
            if new_total > max_examples or new_total in reachable:
                continue
            reachable[new_total] = (total, idx)

    best_total = max(reachable)
    if best_total <= 0:
        min_group = min(sizes)
        raise ValueError(
            "--max_train_examples is too small to keep any full situation intact. "
            f"max_train_examples={max_examples}, smallest_situation_example_count={min_group}"
        )

    selected_indices: list[int] = []
    total = best_total
    while total != 0:
        prev_total, idx = reachable[total]
        selected_indices.append(idx)
        total = prev_total
    selected_indices.reverse()

    selected = [
        ex
        for idx in selected_indices
        for ex in grouped[ordered_situations[idx]]
    ]
    return reorder_examples_to_reduce_adjacent_same_situation(selected, seed=seed + 37)


def write_training_row_manifest(examples: list[dict], output_csv: Path) -> None:
    manifest_rows = []
    for idx, ex in enumerate(examples):
        manifest_rows.append(
            {
                "example_index": idx,
                "situation_id": int(ex["situation_id"]),
                "_source": ex.get("_source"),
                "target_label": ex.get("target_label"),
                "co_winner_label": ex.get("co_winner_label"),
                "prompt_len": int(ex["prompt_len"]),
            }
        )
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_csv, index=False)


def make_run_dir(output_root: Path, run_name: str | None) -> Path:
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(output_root, args.run_name)
    print(f"Output run directory: {run_dir}")

    eval_script_path = Path(args.eval_script).resolve()
    if not eval_script_path.exists():
        raise FileNotFoundError(f"evaluate.py not found at: {eval_script_path}")
    eval_datasets = resolve_eval_datasets(args)
    print("Evaluation datasets:")
    for d in eval_datasets:
        print(f"  - {d['display']}")

    modified_completion_pcts = parse_modified_completion_pcts(args.modified_completion_pcts)
    cot_mode = bool(args.unmodified_cot_data or args.modified_cot_data)
    cot_exact_example_mix = (
        args.cot_unmodified_train_examples is not None or args.cot_modified_train_examples is not None
    )
    cot_exact_example_total = None
    cot_exact_modified_pct = None
    if cot_exact_example_mix and not cot_mode:
        raise ValueError(
            "Exact CoT example-count sampling requires CoT mode. "
            "Provide --unmodified_cot_data (and --modified_cot_data if modified count > 0)."
        )

    if cot_mode:
        if args.full_data or args.modified_data:
            print("Note: CoT mode active. Ignoring --full_data/--modified_data.")
        if args.assistant_style != "label_only" or args.one_label_per_situation:
            print(
                "Note: CoT mode uses completion text from CoT files directly. "
                "--assistant_style and --one_label_per_situation are ignored."
            )
        if not args.unmodified_cot_data:
            raise ValueError("In CoT mode, --unmodified_cot_data is required.")

        unmodified_cot_examples = load_cot_examples(
            path=Path(args.unmodified_cot_data).resolve(),
            prompt_col=args.cot_prompt_column,
            completion_col=args.cot_completion_column,
            situation_col=args.cot_situation_id_column,
            cowinner_col=args.cot_cowinner_column,
            dedupe_one_per_cowinner=False,
        )
        unmodified_situations = (
            unmodified_cot_examples[["situation_id"]]
            .drop_duplicates()
            .assign(_source="unmodified")
            .reset_index(drop=True)
        )
        print(
            f"Loaded unmodified CoT examples: {len(unmodified_cot_examples)} "
            f"across {len(unmodified_situations)} situations"
        )

        if args.modified_cot_data:
            modified_cot_examples = load_cot_examples(
                path=Path(args.modified_cot_data).resolve(),
                prompt_col=args.cot_prompt_column,
                completion_col=args.cot_completion_column,
                situation_col=args.cot_situation_id_column,
                cowinner_col=args.cot_cowinner_column,
                dedupe_one_per_cowinner=True,
            )
            modified_situations = (
                modified_cot_examples[["situation_id"]]
                .drop_duplicates()
                .assign(_source="modified")
                .reset_index(drop=True)
            )
            print(
                f"Loaded modified CoT examples: {len(modified_cot_examples)} "
                f"across {len(modified_situations)} situations "
                f"(deduped to one completion per co-winner)"
            )
        else:
            modified_cot_examples = pd.DataFrame(columns=["situation_id", "prompt", "completion", "co_winner_label"])
            modified_situations = pd.DataFrame(columns=["situation_id", "_source"])

        if cot_exact_example_mix:
            cot_exact_example_total = (
                args.cot_unmodified_train_examples + args.cot_modified_train_examples
            )
            cot_exact_modified_pct = (
                100.0 * args.cot_modified_train_examples / cot_exact_example_total
                if cot_exact_example_total
                else 0.0
            )
            validate_exact_cot_example_request(
                cot_examples=unmodified_cot_examples,
                requested_examples=args.cot_unmodified_train_examples,
                source_name="unmodified",
            )
            validate_exact_cot_example_request(
                cot_examples=modified_cot_examples,
                requested_examples=args.cot_modified_train_examples,
                source_name="modified",
            )
            modified_completion_pcts = [cot_exact_modified_pct]
            print(
                "Note: Exact CoT example-count mode active. "
                f"Sampling {args.cot_modified_train_examples} modified + "
                f"{args.cot_unmodified_train_examples} unmodified CoT rows "
                f"(total {cot_exact_example_total}); ignoring --train_situations/--modified_completion_pcts "
                "for CoT sampling."
            )

        needs_modified = (
            args.cot_modified_train_examples > 0
            if cot_exact_example_mix
            else any(p > 0 for p in modified_completion_pcts)
        )
        if needs_modified and modified_situations.empty:
            raise ValueError(
                "You requested modified completion percentages > 0, but no --modified_cot_data "
                "was provided or it produced zero valid modified situations."
            )
    else:
        if not args.full_data:
            raise ValueError(
                "Provide either --full_data (label-based training mode) "
                "or --unmodified_cot_data/--modified_cot_data (CoT training mode)."
            )
        full_df = read_tabular(Path(args.full_data).resolve())
        full_situations = build_situations_df(
            full_df,
            allow_thinking=args.allow_thinking,
            append_answer_instruction=args.append_answer_instruction,
        )

        modified_situations = pd.DataFrame(columns=[*full_situations.columns, "_source"])
        if args.modified_data:
            modified_df_raw = read_tabular(Path(args.modified_data).resolve())
            modified_situations = build_situations_df(
                modified_df_raw,
                allow_thinking=args.allow_thinking,
                append_answer_instruction=args.append_answer_instruction,
            ).assign(_source="modified")

        needs_modified = any(p > 0 for p in modified_completion_pcts)
        if needs_modified and modified_situations.empty:
            raise ValueError(
                "You requested modified completion percentages > 0, but no --modified_data "
                "was provided or it produced zero valid modified situations."
            )

        modified_ids = set(modified_situations["situation_id"].astype(int).tolist())
        unmodified_situations = full_situations[
            ~full_situations["situation_id"].isin(modified_ids)
        ].copy().assign(_source="unmodified")

    # Persist resolved config
    config_path = run_dir / "resolved_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    if cot_mode:
        print(
            f"Situation pools (CoT mode): "
            f"unmodified={len(unmodified_situations)}, modified={len(modified_situations)}"
        )
    else:
        print(
            f"Situation pools: full={len(full_situations)}, "
            f"unmodified={len(unmodified_situations)}, modified={len(modified_situations)}"
        )

    if args.init_adapter_path:
        print(
            "Continue-from-adapter mode active. Existing adapter weights will be updated. "
            "LoRA creation args (--lora_r/--lora_alpha/--lora_dropout/--lora_target_modules) "
            "are ignored."
        )

    summary_rows: list[dict] = []

    if args.run_baseline_eval and not args.skip_eval:
        baseline_dir = run_dir / "baseline_eval"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for d in eval_datasets:
            baseline_eval_path = baseline_dir / f"eval__{d['tag']}.json"
            baseline_eval = None
            baseline_eval_error = None
            try:
                baseline_eval = evaluate_with_shared_script(
                    args=args,
                    eval_script_path=eval_script_path,
                    eval_dataset=d,
                    output_json=baseline_eval_path,
                    adapter_path=None,
                )
            except Exception as exc:
                baseline_eval_error = f"{type(exc).__name__}: {exc}"
                baseline_eval = load_eval_json_if_present(baseline_eval_path)
                print(
                    f"WARNING: baseline evaluation failed on '{d['tag']}' "
                    f"({baseline_eval_error})."
                )
                if args.fail_on_eval_error:
                    raise
            row = {
                "variant": "baseline",
                "requested_modified_pct_input": None,
                "requested_modified_pct_examples": None,
                "requested_modified_pct_situations": None,
                "actual_modified_pct": None,
                "actual_modified_pct_examples": None,
                "actual_modified_pct_situations": None,
                "train_situations": 0,
                "train_examples": 0,
                "train_mode": None,
                "modified_examples": 0,
                "unmodified_examples": 0,
                "adapter_path": None,
                "continued_from_adapter": False,
                "init_adapter_path": args.init_adapter_path,
                "eval_dataset": d["dataset"],
                "eval_dataset_variant": d["dataset_variant"],
                "eval_custom_csv": d["custom_csv"],
                "eval_json": str(baseline_eval_path),
                "eval_error": baseline_eval_error,
            }
            row.update(extract_eval_metrics(baseline_eval))
            summary_rows.append(row)

    if args.run_init_adapter_eval and not args.skip_eval:
        init_adapter_eval_dir = run_dir / "init_adapter_eval"
        init_adapter_eval_dir.mkdir(parents=True, exist_ok=True)
        init_adapter_path = Path(args.init_adapter_path)
        for d in eval_datasets:
            init_eval_path = init_adapter_eval_dir / f"eval__{d['tag']}.json"
            init_eval = None
            init_eval_error = None
            try:
                init_eval = evaluate_with_shared_script(
                    args=args,
                    eval_script_path=eval_script_path,
                    eval_dataset=d,
                    output_json=init_eval_path,
                    adapter_path=init_adapter_path,
                )
            except Exception as exc:
                init_eval_error = f"{type(exc).__name__}: {exc}"
                init_eval = load_eval_json_if_present(init_eval_path)
                print(
                    f"WARNING: init-adapter evaluation failed on '{d['tag']}' "
                    f"({init_eval_error})."
                )
                if args.fail_on_eval_error:
                    raise
            row = {
                "variant": "init_adapter_baseline",
                "requested_modified_pct_input": None,
                "requested_modified_pct_examples": None,
                "requested_modified_pct_situations": None,
                "actual_modified_pct": None,
                "actual_modified_pct_examples": None,
                "actual_modified_pct_situations": None,
                "train_situations": 0,
                "train_examples": 0,
                "train_mode": "init_adapter_eval",
                "modified_examples": 0,
                "unmodified_examples": 0,
                "adapter_path": str(init_adapter_path),
                "continued_from_adapter": False,
                "init_adapter_path": str(init_adapter_path),
                "eval_dataset": d["dataset"],
                "eval_dataset_variant": d["dataset_variant"],
                "eval_custom_csv": d["custom_csv"],
                "eval_json": str(init_eval_path),
                "eval_error": init_eval_error,
            }
            row.update(extract_eval_metrics(init_eval))
            summary_rows.append(row)

    if modified_completion_pcts:
        ensure_training_dependencies()
        normalize_peft_upcast_dtypes()
        if not args.use_4bit:
            disable_peft_bitsandbytes_dispatch()
        tokenizer = load_tokenizer(args.base_model, trust_remote_code=args.trust_remote_code)
    else:
        tokenizer = None

    for pct in modified_completion_pcts:
        if cot_mode and cot_exact_example_mix:
            selected_cot_examples, sampled_cot_situations = sample_cot_examples_exact(
                unmodified_cot_examples=unmodified_cot_examples,
                modified_cot_examples=modified_cot_examples,
                n_unmodified_examples=args.cot_unmodified_train_examples,
                n_modified_examples=args.cot_modified_train_examples,
                seed=args.seed + int(round(pct * 10)),
            )
            n_mod_situations = int(
                sampled_cot_situations.loc[
                    sampled_cot_situations["_source"] == "modified", "situation_id"
                ].nunique()
            )
            n_unmod_situations = int(
                sampled_cot_situations.loc[
                    sampled_cot_situations["_source"] == "unmodified", "situation_id"
                ].nunique()
            )
            total_sampled_situations = max(1, len(sampled_cot_situations))
            split = SampledSplit(
                sampled_df=sampled_cot_situations,
                n_modified=n_mod_situations,
                n_unmodified=n_unmod_situations,
                actual_modified_pct_situations=100.0 * n_mod_situations / total_sampled_situations,
            )
        else:
            split = sample_situations(
                unmodified_df=unmodified_situations,
                modified_df=modified_situations,
                train_situations=args.train_situations,
                modified_pct=pct,
                seed=args.seed + int(pct * 10),
            )

        pct_label = str(pct).rstrip("0").rstrip(".")
        pct_label = pct_label.replace(".", "p")
        variant_name = f"ft_modpct_{pct_label}"
        variant_dir = run_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        if cot_mode:
            if cot_exact_example_mix:
                selected_cot_examples = selected_cot_examples.copy()
                modified_examples_before_cap = int(
                    (selected_cot_examples["_source"] == "modified").sum()
                )
                unmodified_examples_before_cap = int(
                    (selected_cot_examples["_source"] == "unmodified").sum()
                )
            else:
                selected_unmod_ids = set(
                    split.sampled_df.loc[
                        split.sampled_df["_source"] == "unmodified", "situation_id"
                    ].astype(int).tolist()
                )
                selected_mod_ids = set(
                    split.sampled_df.loc[
                        split.sampled_df["_source"] == "modified", "situation_id"
                    ].astype(int).tolist()
                )
                selected_unmod_examples = unmodified_cot_examples[
                    unmodified_cot_examples["situation_id"].isin(selected_unmod_ids)
                ].copy().assign(_source="unmodified")
                selected_mod_examples = modified_cot_examples[
                    modified_cot_examples["situation_id"].isin(selected_mod_ids)
                ].copy().assign(_source="modified")
                selected_cot_examples = pd.concat(
                    [selected_unmod_examples, selected_mod_examples], ignore_index=True
                )
                modified_examples_before_cap = int(len(selected_mod_examples))
                unmodified_examples_before_cap = int(len(selected_unmod_examples))
            examples = build_training_examples_from_cot(
                tokenizer=tokenizer,
                cot_examples_df=selected_cot_examples,
                allow_thinking=args.allow_thinking,
                system_prompt=args.system_prompt,
                append_answer_instruction=args.append_answer_instruction,
                seed=args.seed + int(pct * 100),
            )
        else:
            examples = build_training_examples(
                tokenizer=tokenizer,
                sampled_df=split.sampled_df,
                assistant_style=args.assistant_style,
                one_label_per_situation=args.one_label_per_situation,
                allow_thinking=args.allow_thinking,
                system_prompt=args.system_prompt,
                seed=args.seed + int(pct * 100),
            )
            example_source_summary_before_cap = summarize_example_sources(examples)
            modified_examples_before_cap = int(example_source_summary_before_cap["modified_examples"])
            unmodified_examples_before_cap = int(
                example_source_summary_before_cap["unmodified_examples"]
            )

        n_examples_before_cap = int(len(examples))
        cap_applied = False
        if args.max_train_examples is not None:
            if len(examples) > args.max_train_examples:
                examples = cap_examples_by_full_situation(
                    examples=examples,
                    max_examples=args.max_train_examples,
                    seed=args.seed + int(pct * 100),
                )
                cap_applied = True

        example_source_summary = summarize_example_sources(examples)
        n_modified_examples = int(example_source_summary["modified_examples"])
        n_unmodified_examples = int(example_source_summary["unmodified_examples"])
        n_other_examples = int(example_source_summary["other_examples"])
        actual_modified_pct_examples = float(example_source_summary["actual_modified_pct_examples"])

        train_dataset = Dataset.from_list(
            [{"text": ex["text"], "prompt_len": ex["prompt_len"]} for ex in examples]
        )

        actual_modified_pct_situations = None
        if cot_mode or "_source" in split.sampled_df.columns:
            actual_modified_pct_situations = float(split.actual_modified_pct_situations)
        requested_modified_pct_examples = (
            float(cot_exact_modified_pct)
            if cot_mode and cot_exact_example_mix and cot_exact_modified_pct is not None
            else None
        )
        requested_modified_pct_situations = None if cot_mode and cot_exact_example_mix else float(pct)

        split.sampled_df[["situation_id"]].to_csv(variant_dir / "sampled_situation_ids.csv", index=False)
        write_training_row_manifest(
            examples=examples,
            output_csv=variant_dir / "sampled_training_rows.csv",
        )
        with open(variant_dir / "train_stats.json", "w") as f:
            json.dump(
                {
                    "requested_modified_pct_input": pct,
                    "requested_modified_pct_examples": requested_modified_pct_examples,
                    "requested_modified_pct_situations": requested_modified_pct_situations,
                    "actual_modified_pct": actual_modified_pct_examples,
                    "actual_modified_pct_examples": actual_modified_pct_examples,
                    "actual_modified_pct_situations": actual_modified_pct_situations,
                    "n_modified": split.n_modified,
                    "n_unmodified": split.n_unmodified,
                    "n_situations": int(len(split.sampled_df)),
                    "n_modified_examples": n_modified_examples,
                    "n_unmodified_examples": n_unmodified_examples,
                    "n_other_examples": n_other_examples,
                    "n_modified_examples_before_cap": modified_examples_before_cap,
                    "n_unmodified_examples_before_cap": unmodified_examples_before_cap,
                    "n_train_examples_before_cap": n_examples_before_cap,
                    "n_train_examples": int(len(examples)),
                    "max_train_examples": args.max_train_examples,
                    "train_examples_capped": cap_applied,
                    "training_mode": "cot_files" if cot_mode else "label_generated",
                    "cot_exact_example_mix": bool(cot_mode and cot_exact_example_mix),
                    "requested_modified_examples": (
                        int(args.cot_modified_train_examples)
                        if cot_mode and cot_exact_example_mix
                        else None
                    ),
                    "requested_unmodified_examples": (
                        int(args.cot_unmodified_train_examples)
                        if cot_mode and cot_exact_example_mix
                        else None
                    ),
                    "requested_total_examples_exact_mix": (
                        int(cot_exact_example_total)
                        if cot_mode and cot_exact_example_mix and cot_exact_example_total is not None
                        else None
                    ),
                    "requested_modified_pct_exact_mix": (
                        float(cot_exact_modified_pct)
                        if cot_mode and cot_exact_example_mix and cot_exact_modified_pct is not None
                        else None
                    ),
                    "continued_from_adapter": bool(args.init_adapter_path),
                    "init_adapter_path": args.init_adapter_path,
                    "allow_thinking": args.allow_thinking,
                    "system_prompt": args.system_prompt,
                },
                f,
                indent=2,
            )

        if cot_mode and cot_exact_example_mix:
            print(
                f"\nTraining variant {variant_name}: "
                f"{len(split.sampled_df)} sampled situations "
                f"({split.n_modified} modified, {split.n_unmodified} unmodified), "
                f"{len(examples)} CoT examples "
                f"({n_modified_examples} modified, {n_unmodified_examples} unmodified)"
            )
        else:
            print(
                f"\nTraining variant {variant_name}: "
                f"{len(split.sampled_df)} situations, "
                f"{split.n_modified} modified ({split.actual_modified_pct_situations:.1f}% situations), "
                f"{len(examples)} examples"
            )
        if cap_applied:
            print(
                f"  Capped training examples from {n_examples_before_cap} "
                f"to {len(examples)} via --max_train_examples"
            )

        model = load_model_for_training(args)
        model.print_trainable_parameters()
        trainer_output_dir = variant_dir / "trainer_outputs"
        checkpoint_eval_dir = variant_dir / "checkpoint_selection"

        trainer_fp16 = args.fp16
        trainer_bf16 = args.bf16
        if args.use_4bit and trainer_fp16 and torch.cuda.is_available():
            cc_major, _ = torch.cuda.get_device_capability()
            if cc_major < 8:
                # On pre-Ampere GPUs (e.g., V100), 4-bit + fp16 training can hit
                # GradScaler unscale failures with bf16 grads in some stacks.
                # Disable Trainer mixed precision to keep QLoRA runs stable.
                print(
                    "Note: disabling Trainer fp16 mixed precision for 4-bit training "
                    f"on compute capability {cc_major}.x GPU."
                )
                trainer_fp16 = False
                trainer_bf16 = False

        world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        effective_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        )
        steps_per_epoch = max(1, math.ceil(len(train_dataset) / effective_batch_size))
        total_train_steps = max(1, math.ceil(steps_per_epoch * float(args.num_train_epochs)))
        warmup_steps = max(1, int(round(0.1 * total_train_steps)))
        collator = build_prompt_masking_collator(tokenizer)

        common_trainer_args = dict(
            output_dir=str(trainer_output_dir),
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            fp16=trainer_fp16,
            bf16=trainer_bf16,
            report_to="none",
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
            gradient_checkpointing=args.gradient_checkpointing,
            max_grad_norm=1.0,
            save_total_limit=max(int(math.ceil(args.num_train_epochs)), 1),
            seed=args.seed,
            remove_unused_columns=False,
        )

        sft_trainer_init_params = set(inspect.signature(SFTTrainer.__init__).parameters)
        uses_new_trl_api = (
            "processing_class" in sft_trainer_init_params and "tokenizer" not in sft_trainer_init_params
        )

        if uses_new_trl_api and SFTConfig is not None:
            sft_config_params = set(inspect.signature(SFTConfig.__init__).parameters)
            extra_sft_config_args = {}
            if "dataset_text_field" in sft_config_params:
                extra_sft_config_args["dataset_text_field"] = "text"
            if "packing" in sft_config_params:
                extra_sft_config_args["packing"] = False
            if "max_seq_length" in sft_config_params:
                extra_sft_config_args["max_seq_length"] = args.max_seq_length
            elif "max_length" in sft_config_params:
                extra_sft_config_args["max_length"] = args.max_seq_length
            training_args = SFTConfig(
                **common_trainer_args,
                **extra_sft_config_args,
            )
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                processing_class=tokenizer,
                args=training_args,
                data_collator=collator,
            )
        else:
            training_args = TrainingArguments(**common_trainer_args)
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                dataset_text_field="text",
                tokenizer=tokenizer,
                args=training_args,
                max_seq_length=args.max_seq_length,
                packing=False,
                data_collator=collator,
            )

        print(
            "Training setup: prompt-masked loss, packing=False, "
            f"warmup_steps={warmup_steps}, max_seq_length={args.max_seq_length}."
        )
        if not args.select_best_checkpoint:
            print(
                "Checkpoint selection disabled; final-epoch weights will be used for reporting. "
                "Pass --select_best_checkpoint to score saved checkpoints and choose the best one."
            )
            if args.save_strategy != "no":
                print(
                    "Note: checkpoints are still being saved because --save_strategy is not 'no'. "
                    "Use --save_strategy no to skip intermediate checkpoint saves."
                )

        trainer.train()

        final_adapter_dir = variant_dir / "final_adapter"
        trainer.model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)

        adapter_dir = variant_dir / "adapter"
        selected_checkpoint_name = None
        selected_checkpoint_path = None
        checkpoint_selection_status = None
        checkpoint_selection_eval_json = None

        if args.select_best_checkpoint and not args.skip_eval:
            checkpoint_eval_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dirs = sorted(trainer_output_dir.glob("checkpoint-*"), key=checkpoint_step)
            if not checkpoint_dirs:
                raise RuntimeError(
                    "No checkpoints were saved; cannot apply checkpoint selection. "
                    "Use --save_strategy epoch/steps or disable --select_best_checkpoint."
                )

            selection_dataset = eval_datasets[0]
            checkpoint_records: list[dict] = []
            print(f"Selecting best checkpoint on: {selection_dataset['display']}")
            for checkpoint_dir in checkpoint_dirs:
                output_json = checkpoint_eval_dir / f"{checkpoint_dir.name}__{selection_dataset['tag']}.json"
                metrics = None
                eval_error = None
                try:
                    eval_json = evaluate_with_shared_script(
                        args=args,
                        eval_script_path=eval_script_path,
                        eval_dataset=selection_dataset,
                        output_json=output_json,
                        adapter_path=checkpoint_dir,
                    )
                    metrics = extract_eval_metrics(eval_json)
                except Exception as exc:
                    eval_error = f"{type(exc).__name__}: {exc}"
                    metrics = extract_eval_metrics(load_eval_json_if_present(output_json))
                    print(
                        f"WARNING: checkpoint evaluation failed for '{checkpoint_dir.name}' "
                        f"({eval_error})."
                    )
                    if args.fail_on_eval_error:
                        raise

                record = {
                    "checkpoint_path": str(checkpoint_dir),
                    "checkpoint_name": checkpoint_dir.name,
                    "metrics": metrics if metrics else None,
                    "eval_json": str(output_json),
                    "eval_error": eval_error,
                }
                checkpoint_records.append(record)
                if metrics:
                    print(
                        f"  - {checkpoint_dir.name}: "
                        f"coop={float(metrics.get('cooperate_rate') or 0.0):.1%}, "
                        f"parse={float(metrics.get('parse_rate') or 0.0):.1%}, "
                        f"CARA={float(metrics.get('best_cara_rate') or 0.0):.1%}"
                    )

            successful_records = [record for record in checkpoint_records if record.get("metrics")]
            if not successful_records:
                raise RuntimeError("All checkpoint-selection evaluations failed.")

            best_record, checkpoint_selection_status = choose_best_checkpoint_record(
                checkpoint_records=successful_records,
                metric_name=args.checkpoint_selection_metric,
                parse_floor=args.checkpoint_parse_rate_floor,
            )
            selected_checkpoint_name = best_record["checkpoint_name"]
            selected_checkpoint_path = best_record["checkpoint_path"]
            checkpoint_selection_eval_json = best_record["eval_json"]

            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            shutil.copytree(selected_checkpoint_path, adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
        else:
            checkpoint_selection_status = (
                "skipped_eval" if args.skip_eval else "final_adapter_used"
            )
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            shutil.copytree(final_adapter_dir, adapter_dir)

        # Explicit cleanup between runs to reduce VRAM pressure.
        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        base_row = {
            "variant": variant_name,
            "requested_modified_pct_input": pct,
            "requested_modified_pct_examples": requested_modified_pct_examples,
            "requested_modified_pct_situations": requested_modified_pct_situations,
            "actual_modified_pct": actual_modified_pct_examples,
            "actual_modified_pct_examples": actual_modified_pct_examples,
            "train_situations": int(len(split.sampled_df)),
            "train_examples": int(len(examples)),
            "train_examples_before_cap": n_examples_before_cap,
            "max_train_examples": args.max_train_examples,
            "train_examples_capped": cap_applied,
            "train_mode": "cot_files" if cot_mode else "label_generated",
            "modified_examples": int(n_modified_examples),
            "unmodified_examples": int(n_unmodified_examples),
            "other_examples": int(n_other_examples),
            "modified_examples_before_cap": int(modified_examples_before_cap),
            "unmodified_examples_before_cap": int(unmodified_examples_before_cap),
            "actual_modified_pct_situations": actual_modified_pct_situations,
            "adapter_path": str(adapter_dir),
            "final_adapter_path": str(final_adapter_dir),
            "selected_checkpoint_name": selected_checkpoint_name,
            "selected_checkpoint_path": selected_checkpoint_path,
            "checkpoint_selection_status": checkpoint_selection_status,
            "checkpoint_selection_metric": args.checkpoint_selection_metric,
            "checkpoint_parse_rate_floor": args.checkpoint_parse_rate_floor,
            "checkpoint_selection_eval_json": checkpoint_selection_eval_json,
            "continued_from_adapter": bool(args.init_adapter_path),
            "init_adapter_path": args.init_adapter_path,
            "allow_thinking": args.allow_thinking,
        }

        if args.skip_eval:
            row = dict(base_row)
            row["eval_dataset"] = None
            row["eval_dataset_variant"] = None
            row["eval_custom_csv"] = None
            row["eval_json"] = None
            row["eval_error"] = None
            summary_rows.append(row)
        else:
            for d in eval_datasets:
                eval_path = variant_dir / f"eval__{d['tag']}.json"
                eval_json = None
                eval_error = None
                try:
                    eval_json = evaluate_with_shared_script(
                        args=args,
                        eval_script_path=eval_script_path,
                        eval_dataset=d,
                        output_json=eval_path,
                        adapter_path=adapter_dir,
                    )
                except Exception as exc:
                    eval_error = f"{type(exc).__name__}: {exc}"
                    eval_json = load_eval_json_if_present(eval_path)
                    print(
                        f"WARNING: evaluation failed for variant '{variant_name}' "
                        f"on dataset '{d['tag']}' ({eval_error})."
                    )
                    if args.fail_on_eval_error:
                        raise
                row = dict(base_row)
                row["eval_dataset"] = d["dataset"]
                row["eval_dataset_variant"] = d["dataset_variant"]
                row["eval_custom_csv"] = d["custom_csv"]
                row["eval_json"] = str(eval_path)
                row["eval_error"] = eval_error
                row.update(extract_eval_metrics(eval_json))
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("\nAll done.")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
