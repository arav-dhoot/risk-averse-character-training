"""On-policy reverse-KL distillation of a prompted teacher into a LoRA student.

One model is loaded with a LoRA adapter. With the adapter on it is the
student: it samples rollouts on the training prompts and is scored with
gradients. With the adapter off it is the teacher: the same base model
scoring the same rollouts behind the constitution's system prompt. The
per-token reverse KL (student minus teacher log-probability) is the only
training signal, applied as a negative advantage in an importance-sampling
policy-gradient loss.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

MetricsCallback = Callable[[int, dict[str, Any]], None]

DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True, kw_only=True)
class DistillConfig:
    model: str
    model_revision: str | None = None
    system_prompt: str
    prompts: str
    renderer: str
    out: str
    lora_rank: int = 32
    lora_alpha: int | None = None            # default: equal to the rank
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    lr: float = 1e-4
    max_steps: int = 500
    groups_per_batch: int = 32
    group_size: int = 4
    save_every: int = 20
    shuffle_seed: int = 12345                # order of the prompt batches
    max_tokens: int = 512
    max_prompt_tokens: int = 1024
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    device: str = "cuda:0"
    gen_batch_size: int = 32                 # prompts per generate() call
    fwd_microbatch: int = 8                  # sequences per with-grad forward
    loss_norm: str = "token_mean"            # "token_mean" | "sum"
    seed: int = 0                            # rollout sampling seed
    teacher_prefix_after_bos: bool = False   # splice the system turn after <bos> (Gemma)
    save_at_tokens: tuple[int, ...] = ()     # cumulative-token thresholds that force a save
    store_rollouts: bool = False             # write every step's rollouts to <out>/rollouts/

    def __post_init__(self) -> None:
        if self.loss_norm not in ("token_mean", "sum"):
            raise ValueError(f"unknown loss_norm {self.loss_norm!r}")
        if any(t <= 0 for t in self.save_at_tokens):
            raise ValueError("save_at_tokens thresholds must be positive, "
                             f"got {self.save_at_tokens}")


@dataclass(frozen=True)
class DistillResult:
    out_dir: str
    adapter_path: str
    final_metrics: dict
    rollouts_dir: str | None = None


# ------------------------------------------------------------------ prompts

def load_prompts(path: str | Path, field: str = "prompt") -> list[str]:
    """Prompts from a JSONL file of `{"prompt": ...}` rows, in file order."""
    prompts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)[field])
    if not prompts:
        raise ValueError(f"no prompts loaded from {path}")
    return prompts


def cycle_prompts(prompts: list[str], n_steps: int, per_step: int, seed: int) -> list[list[str]]:
    """Deterministic per-step prompt batches. Each epoch is a fresh seeded
    shuffle of the corpus; the tail of an epoch is topped up from the next
    shuffle, so exactly `n_steps` batches come back regardless of corpus size."""
    if not prompts:
        raise ValueError("no prompts to cycle")
    rng = random.Random(seed)
    pool: list[str] = []
    batches: list[list[str]] = []
    while len(batches) < n_steps:
        if len(pool) < per_step:
            block = list(prompts)
            rng.shuffle(block)
            pool.extend(block)
        batches.append([pool.pop(0) for _ in range(per_step)])
    return batches


# ---------------------------------------------------------------- helpers

def lora_exclusion(model, target_modules: tuple[str, ...]) -> dict | None:
    """PEFT `exclude_modules` regex for target-named modules that are not
    `torch.nn.Linear` (multimodal checkpoints carry such modules in their
    vision towers, and PEFT refuses to wrap them). None when nothing needs
    excluding."""
    import re
    import torch

    bad, good = [], []
    for name, mod in model.named_modules():
        if name.split(".")[-1] in target_modules:
            (good if isinstance(mod, torch.nn.Linear) else bad).append(name)
    if not bad:
        return None
    if not good:
        raise RuntimeError(f"no torch.nn.Linear module matches the LoRA targets {target_modules}")
    prefix = lambda n: ".".join(n.split(".")[:2])  # noqa: E731
    bad_prefixes = sorted({prefix(n) for n in bad})
    clean = [p for p in bad_prefixes if not any(g.startswith(p + ".") for g in good)]
    mixed = [n for n in bad if prefix(n) not in clean]
    parts = [re.escape(p) + r"\..*" for p in clean] + [re.escape(n) for n in mixed]
    regex = "|".join(parts)
    excluded = [n for n in bad if re.fullmatch(regex, n)]
    wrongly = [n for n in good if re.fullmatch(regex, n)]
    if wrongly or len(excluded) != len(bad):
        raise RuntimeError(f"LoRA exclusion regex mismatch: excludes {wrongly[:3]} "
                           f"or misses {len(bad) - len(excluded)} modules")
    return {"regex": regex, "count": len(bad), "wrappable": len(good),
            "prefixes": clean + ([f"{len(mixed)} listed"] if mixed else []),
            "example": f"{bad[0]} ({type(dict(model.named_modules())[bad[0]]).__name__})"}


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def rollout_rows(step: int, batch_prompts: list[str], datums: list[dict], tokenizer) -> list[dict]:
    """One row per trained-on rollout, in training order."""
    rows = []
    for d in datums:
        n = len(d["response_ids"])
        rows.append({
            "step": step,
            "prompt_index": d["prompt_index"],
            "sample_index": d["sample_index"],
            "prompt": batch_prompts[d["prompt_index"]],
            "response": tokenizer.decode(d["response_ids"], skip_special_tokens=False),
            "response_ids": list(d["response_ids"]),
            "response_len": n,
            "truncated": bool(d["truncated"]),
            "student_logprob_sum": d["student_logprob_sum"],
            "teacher_logprob_sum": d["teacher_logprob_sum"],
            "rkl_sum": d["rkl_sum"],
            "rkl_mean": d["rkl_sum"] / n,
        })
    return rows


def _write_rollouts(path: Path, rows: list[dict]) -> None:
    # whole-file write via rename, so a crash never leaves a partial step file
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def build_teacher_seq(prefix_ids: list[int], prompt_ids: list[int], response_ids: list[int],
                      *, prefix_after_bos: bool = False) -> list[int]:
    """The teacher-scored sequence: `prefix + prompt + response`, or with the
    prefix spliced after the prompt's leading `<bos>`. Both layouts have the
    same length, so the response tokens sit at the same positions."""
    if prefix_after_bos:
        if not prompt_ids:
            raise RuntimeError("prefix_after_bos with empty prompt_ids")
        return prompt_ids[:1] + prefix_ids + prompt_ids[1:] + response_ids
    return prefix_ids + prompt_ids + response_ids


def compute_is_loss(logp, mask, advantages):
    """Importance-sampling policy-gradient loss on one microbatch. The ratio
    exp(logp - logp.detach()) is 1 at compute time and carries the gradient.
    Returns the unnormalised negative sum and the masked token count."""
    import torch

    ratio = torch.exp(logp - logp.detach())
    loss_sum = -(ratio * advantages * mask).sum()
    return loss_sum, float(mask.sum().item())


def gather_token_logprobs(logits, target_ids):
    """Log-softmax over the vocabulary, gathered at the realised next tokens."""
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(logp, -1, target_ids.unsqueeze(-1)).squeeze(-1)


def resolve_stop_ids(stop_seqs, tokenizer, renderer_name: str) -> list[int]:
    """Renderer stop sequences as single token ids."""
    stop_ids: list[int] = []
    for s in stop_seqs:
        if isinstance(s, int):
            stop_ids.append(s)
        elif isinstance(s, str):
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"stop sequence {s!r} is not a single token ({ids})")
            stop_ids.append(ids[0])
        else:
            raise RuntimeError(f"unsupported stop sequence {s!r} from renderer {renderer_name!r}")
    return stop_ids


# ----------------------------------------------------------------- engine

class HFEngine:
    """One HF model with a LoRA adapter: adapter on for the student (sampling
    and with-grad scoring), adapter off for the teacher."""

    def __init__(self, cfg: DistillConfig):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model, revision=cfg.model_revision, torch_dtype=torch.bfloat16,
            device_map=cfg.device)
        base.config.use_cache = True
        exclude = lora_exclusion(base, cfg.lora_target_modules)
        if exclude:
            print(f"[distill] LoRA: excluding {exclude['count']} target-named non-Linear modules "
                  f"under {exclude['prefixes']} (e.g. {exclude['example']}); "
                  f"{exclude['wrappable']} Linear targets remain")
        lora = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha or cfg.lora_rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(cfg.lora_target_modules),
            task_type="CAUSAL_LM",
            **({"exclude_modules": exclude["regex"]} if exclude else {}),
        )
        self.model = get_peft_model(base, lora)
        self.model.print_trainable_parameters()
        self.optimizer = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)
        self.device = torch.device(cfg.device)

    def sample(self, prompt_token_lists: list[list[int]], stop_ids: list[int],
               group_size: int, rng_seed: int) -> list[dict]:
        """`group_size` samples per prompt, trimmed at the first stop id.
        Empty rollouts are dropped."""
        torch = self.torch
        cfg = self.cfg
        torch.manual_seed(rng_seed)
        out: list[dict] = []
        self.model.eval()
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        with torch.no_grad():
            for i in range(0, len(prompt_token_lists), cfg.gen_batch_size):
                chunk = prompt_token_lists[i:i + cfg.gen_batch_size]
                maxlen = max(len(p) for p in chunk)
                input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
                attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
                for j, p in enumerate(chunk):        # left-pad for generate
                    input_ids[j, maxlen - len(p):] = torch.tensor(p)
                    attn[j, maxlen - len(p):] = 1
                gen = self.model.generate(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn.to(self.device),
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=1.0, top_k=0,
                    max_new_tokens=cfg.max_tokens,
                    num_return_sequences=group_size,
                    eos_token_id=stop_ids,
                    pad_token_id=pad_id,
                )
                new_tokens = gen[:, maxlen:]
                for j, p in enumerate(chunk):
                    for g in range(group_size):
                        row = new_tokens[j * group_size + g].tolist()
                        resp = []
                        for t in row:
                            resp.append(t)
                            if t in stop_ids:
                                break
                        while resp and resp[-1] == pad_id and pad_id not in stop_ids:
                            resp.pop()
                        if resp:
                            out.append({"prompt_ids": list(p),
                                        "response_ids": resp,
                                        "truncated": resp[-1] not in stop_ids,
                                        "prompt_index": i + j,
                                        "sample_index": g})
        return out

    def train_step(self, datums: list[dict], prefix_ids: list[int], kl_coef: float) -> dict:
        """Teacher-KL advantages and one Adam step. Each datum gains
        student/teacher log-probability sums and its reverse-KL sum."""
        torch = self.torch
        cfg = self.cfg
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # teacher log-probabilities: adapter off, system prefix prepended
        self.model.eval()
        teacher_lp: list = []
        with torch.no_grad(), self.model.disable_adapter():
            for i in range(0, len(datums), cfg.fwd_microbatch):
                chunk = datums[i:i + cfg.fwd_microbatch]
                seqs = [build_teacher_seq(prefix_ids, d["prompt_ids"], d["response_ids"],
                                          prefix_after_bos=cfg.teacher_prefix_after_bos)
                        for d in chunk]
                maxlen = max(len(s) for s in seqs)
                ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
                attn = torch.zeros_like(ids)
                for j, s in enumerate(seqs):        # right-pad for scoring
                    ids[j, :len(s)] = torch.tensor(s)
                    attn[j, :len(s)] = 1
                ids, attn = ids.to(self.device), attn.to(self.device)
                logits = self.model(input_ids=ids, attention_mask=attn).logits
                lp = gather_token_logprobs(logits[:, :-1], ids[:, 1:])
                for j, d in enumerate(chunk):
                    start = len(prefix_ids) + len(d["prompt_ids"]) - 1
                    teacher_lp.append(lp[j, start:start + len(d["response_ids"])].detach().cpu())

        # student with-grad pass, microbatched; each microbatch backwards
        # immediately so only one activation graph is alive at a time
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        kl_sum, tok_sum = 0.0, 0.0
        lp_sum, n_truncated = 0.0, 0
        total_tok = sum(len(d["response_ids"]) for d in datums)
        denom = total_tok if cfg.loss_norm == "token_mean" else 1.0
        seen_tok = 0
        for i in range(0, len(datums), cfg.fwd_microbatch):
            chunk = datums[i:i + cfg.fwd_microbatch]
            seqs = [d["prompt_ids"] + d["response_ids"] for d in chunk]
            maxlen = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros_like(ids)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s)
                attn[j, :len(s)] = 1
            ids, attn = ids.to(self.device), attn.to(self.device)
            logits = self.model(input_ids=ids, attention_mask=attn).logits
            lp = gather_token_logprobs(logits[:, :-1], ids[:, 1:])

            B, T = lp.shape
            mask = torch.zeros((B, T), device=lp.device)
            adv = torch.zeros((B, T), device=lp.device)
            for j, d in enumerate(chunk):
                start = len(d["prompt_ids"]) - 1
                n = len(d["response_ids"])
                mask[j, start:start + n] = 1.0
                t_lp = teacher_lp[i + j].to(lp.device)
                s_lp = lp[j, start:start + n].detach()
                rkl = s_lp - t_lp
                adv[j, start:start + n] = -kl_coef * rkl
                rkl_sum_j = float(rkl.sum())
                kl_sum += rkl_sum_j
                tok_sum += n
                lp_sum += float(s_lp.sum())
                n_truncated += int(d.get("truncated", False))
                d["student_logprob_sum"] = float(s_lp.sum())
                d["teacher_logprob_sum"] = float(t_lp.sum())
                d["rkl_sum"] = rkl_sum_j
            loss_sum, ntok = compute_is_loss(lp, mask, adv)
            seen_tok += ntok
            (loss_sum / denom).backward(retain_graph=False)

        if seen_tok != total_tok:
            raise RuntimeError(f"loss-mask token count {seen_tok} != response-token count "
                               f"{total_tok}; refusing to step")
        self.optimizer.step()
        return {
            "teacher_kl": kl_sum / tok_sum if tok_sum else 0.0,
            "loss_tokens": total_tok,
            "entropy": -lp_sum / tok_sum if tok_sum else 0.0,
            "truncation_rate": n_truncated / len(datums),
            "response_len_mean": total_tok / len(datums),
        }

    def save_adapter(self, path: Path, step: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.torch.save({"optimizer": self.optimizer.state_dict(), "step": step},
                        path / "optim_state.pt")

    def load_adapter_state(self, path: Path) -> int:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        sd = load_file(str(Path(path) / "adapter_model.safetensors"))
        set_peft_model_state_dict(self.model, sd)
        blob = self.torch.load(path / "optim_state.pt", weights_only=False)
        self.optimizer.load_state_dict(blob["optimizer"])
        return int(blob["step"])


# ------------------------------------------------------------------- loop

def run_distill(cfg: DistillConfig, *, teacher_prefix_tokens: list[int],
                on_metrics: MetricsCallback | None = None,
                resume_from: str | None = None) -> DistillResult:
    """Train for `cfg.max_steps` steps. Writes `<out>/metrics.jsonl`,
    `<out>/checkpoints.jsonl`, `<out>/token_checkpoints.json` (when
    `save_at_tokens` is set) and adapters under `<out>/step<N>/`."""
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from risk_averse_character.chat_formats import register_renderers

    register_renderers()

    out = Path(cfg.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    ckpts_path = out / "checkpoints.jsonl"
    rollouts_dir: Path | None = None
    if cfg.store_rollouts:
        rollouts_dir = out / "rollouts"
        rollouts_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(cfg.model)
    renderer = renderers.get_renderer(cfg.renderer, tokenizer=tokenizer)
    stop_ids = resolve_stop_ids(renderer.get_stop_sequences(), tokenizer, cfg.renderer)

    prompts = load_prompts(cfg.prompts)
    batches = cycle_prompts(prompts, cfg.max_steps, cfg.groups_per_batch, seed=cfg.shuffle_seed)

    bos_id = getattr(tokenizer, "bos_token_id", None)
    if cfg.teacher_prefix_after_bos and bos_id is None:
        raise RuntimeError("teacher_prefix_after_bos requires a tokenizer with a bos_token_id")

    def render_prompt(p: str) -> list[int]:
        # no special tokens here: the truncation count must not include a
        # tokenizer-added <bos>, and the decode must not leak it into the text
        toks = tokenizer.encode(p, add_special_tokens=False)
        if len(toks) > cfg.max_prompt_tokens:
            p = tokenizer.decode(toks[:cfg.max_prompt_tokens])
        ids = renderer.build_generation_prompt([{"role": "user", "content": p}]).to_ints()
        if cfg.teacher_prefix_after_bos and (not ids or ids[0] != bos_id):
            raise RuntimeError(f"rendered prompt does not start with bos id {bos_id} "
                               f"(got {ids[:3]}); the teacher prefix cannot be spliced")
        return ids

    engine = HFEngine(cfg)
    start_step = 0
    if resume_from:
        start_step = engine.load_adapter_state(Path(resume_from))
        print(f"[distill] resumed adapter and optimizer from {resume_from} "
              f"(completed step {start_step})")

    adapter_path: str | None = None
    final_metrics: dict[str, Any] = {}
    train_tokens_cum = 0
    pending_saves = sorted(set(cfg.save_at_tokens))
    token_ckpts: dict[int, dict] = {}
    for step in range(start_step, len(batches)):
        t0 = time.time()
        prompt_ids = [render_prompt(p) for p in batches[step]]
        datums = engine.sample(prompt_ids, stop_ids, cfg.group_size,
                               rng_seed=cfg.seed * 1_000_003 + step)
        if not datums:
            raise RuntimeError(f"step {step}: zero non-empty rollouts")
        metrics = {"lr": cfg.lr}
        metrics.update(engine.train_step(datums, teacher_prefix_tokens, cfg.kl_penalty_coef))
        train_tokens_cum += int(metrics["loss_tokens"])
        metrics["train_tokens_cum"] = train_tokens_cum
        if rollouts_dir is not None:
            _write_rollouts(rollouts_dir / f"step{step:04d}.jsonl",
                            rollout_rows(step, batches[step], datums, tokenizer))
        crossed = [t for t in pending_saves if t <= train_tokens_cum]
        if crossed:
            del pending_saves[:len(crossed)]
        if crossed \
                or (cfg.save_every > 0 and (step + 1) % cfg.save_every == 0) \
                or step + 1 == len(batches):
            ckpt_dir = out / f"step{step + 1}"
            engine.save_adapter(ckpt_dir, step + 1)
            adapter_path = str(ckpt_dir)
            ck_row = {"step": step + 1, "adapter_path": adapter_path,
                      "train_tokens_cum": train_tokens_cum}
            if crossed:
                ck_row["token_checkpoints"] = crossed
                for t in crossed:
                    token_ckpts[t] = {"step": step + 1, "tokens": train_tokens_cum,
                                      "adapter_path": adapter_path}
                (out / "token_checkpoints.json").write_text(
                    json.dumps(token_ckpts, indent=2) + "\n")
                print(f"[distill] token checkpoint at step {step + 1} "
                      f"({train_tokens_cum} tokens; thresholds {crossed})")
            _append_jsonl(ckpts_path, ck_row)
        metrics["time/step"] = time.time() - t0
        _append_jsonl(metrics_path, {"step": step, **metrics})
        final_metrics.update(metrics)
        if on_metrics is not None:
            try:
                on_metrics(step, metrics)
            except Exception:
                pass                                 # a progress tap never kills a run

    if pending_saves:
        print(f"[distill] warning: save-at-tokens thresholds never reached "
              f"(run ended at {train_tokens_cum} tokens): {pending_saves}")
        final_metrics["token_checkpoints_unreached"] = list(pending_saves)
    if not adapter_path:
        raise RuntimeError(f"loop finished without an adapter save in {out}")
    return DistillResult(out_dir=str(out), adapter_path=adapter_path,
                         final_metrics=final_metrics,
                         rollouts_dir=str(rollouts_dir) if rollouts_dir is not None else None)
