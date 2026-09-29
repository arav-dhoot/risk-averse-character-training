# Character Training for Risk-Averse Agents

Risk aversion in resources could prevent misaligned AI agents from causing
catastrophic harm. Misaligned but risk-averse agents would tend to favor safer
strategies like making deals with humans over riskier strategies like
rebelling. We train agents to be risk averse through character training,
finding that persona traits provide a robust mechanism for instilling risk
preferences. To do this, we construct a model constitution describing constant
absolute risk aversion (CARA) over an agent's resources and instill it through
on-policy distillation. Despite never seeing the benchmark's decision format
during training, character-trained models are competitive with baselines
trained directly on it, and generalise better than them out of distribution on
two of our four models. We also modulate different aspects of the constitution,
finding that token budget and model choice are the most influential aspect of
character training to instill risk aversion. We conclude from these results
that character training is a promising and scalable way to instil broad
dispositions, which we can use to our advantage in mitigating risk from
misaligned AI agents.

This repository contains the code and data for the paper: the constitutions,
the on-policy reverse-KL distillation loop, the SFT, tie-training and DPO
baselines of Zhang et al. (2026), and the evaluation battery (the
seven risk-preference sets, model-written persona evaluations, DTBench,
MMLU-Redux, GPQA, BailBench and mu-decisiveness). Everything is model-agnostic.
The paper's runs used Qwen3.5-9B, Qwen3.8-27B, Gemma-4-12B and Gemma-4-31B;
the model, its revision and the chat format are command-line arguments
throughout.

## Layout

```
constitutions/      the 12 risk-averse constitutions (3 x 2 x 2 factorial) and 3 controls
data/               prompt corpus and every evaluation set (see data/README.md)
third_party/        the baselines' upstream trainer and training data
rac/                the package
  constitution.py   constitution loading and system-prompt rendering
  client.py         async client for an OpenAI-compatible server, with a disk cache
  chat_formats.py   chat renderers and the teacher prefix for each model family
  training/         distill.py (reverse-KL loop), baselines.py (SFT / tie / DPO)
  evals/            risk/, mmlu.py, gpqa.py, persona/, dtbench/, welfare/
scripts/            command-line entry points
```

## Setup

```bash
uv venv -p 3.12 && uv sync --extra train --extra serve   # training + vLLM (Linux, GPU)
uv sync                                                  # evaluation client only
uv run python scripts/check_data.py                      # verify the shipped data
```

`HF_TOKEN` must be set to download gated checkpoints and the GPQA data. No other
credentials are used.

## Constitutions

Each constitution is a JSON list of first-person traits. The twelve risk-averse
constitutions vary three factors: the number of worked examples (`e0`, `e2`,
`e4`), declarative versus procedural phrasing (`decl`, `proc`) and whether the
model is asked to trace the utility curve (`trace`, `notrace`). The controls are
a risk-seeking character (`rs_e2_decl_notrace`), a risk-neutral character
(`rn_e2_decl_notrace`) and an unrelated persona (`generic_town_mineral`).

```bash
uv run python -c "from rac.constitution import render_constitution as r; print(r('ra_e2_decl_trace', 'google/gemma-4-12B-it'))"
```

## Distillation

One run per constitution per model. The teacher is the same weights with the
adapter disabled and the constitution as its system prompt; the student sees
bare user turns. The defaults below are the paper's recipe (rank-32 LoRA on the
seven projection matrices, Adam at 1e-4, 32 prompts x 4 samples per step, 500
steps, temperature 1.0, 512 new tokens, token-mean loss, prompt order seed
12345, sampling seed 0).

```bash
uv run python scripts/distill.py \
    --model google/gemma-4-12B-it --model-revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
    --renderer gemma4_disable_thinking \
    --constitution ra_e2_decl_trace \
    --save-at-tokens 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000 20000000 \
    --out runs/g12-ra_e2_decl_trace
```

Qwen models use `--renderer qwen3_5_disable_thinking`. The run directory
receives `run.json` (every argument and the rendered system prompt),
`metrics.jsonl` (per-step teacher KL, loss tokens, cumulative tokens),
`checkpoints.jsonl`, an adapter every 20 steps under `step<N>/`, and
`token_checkpoints.json` mapping each `--save-at-tokens` rung to the adapter
saved at the first step that reached it. `--resume` continues an interrupted run
from its last save. `--dry-run` prints the resolved configuration.

Base checkpoints used in the paper:

| Model | Revision | Renderer |
|---|---|---|
| Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a | qwen3_5_disable_thinking |
| google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | gemma4_disable_thinking |
| Qwen/Qwen3.8-27B | 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 | qwen3_5_disable_thinking |
| google/gemma-4-31B-it | 842da3794eaa0b77d5f08bae87a17459d91ff475 | gemma4_disable_thinking |

## Baselines

The three demonstration-based recipes of Zhang et al. (2026), with
their data and hyperparameters, on any base model. The learning rate is not
part of the released recipes; the paper used 1e-4 for every method.

```bash
uv run python scripts/train_baseline.py --model Qwen/Qwen3.5-9B --renderer qwen3_5_disable_thinking \
    --method sft --lr 1e-4 --data-variant think_tags --system-prompt gamble --out runs/q35-sft
uv run python scripts/train_baseline.py --model google/gemma-4-12B-it --renderer gemma4_disable_thinking \
    --method dpo --lr 1e-4 --data-variant no_think_tags --system-prompt none --out runs/g12-dpo
```

`--method tie` trains on 700 unmodified plus 300 tie-modified demonstrations.
Qwen models train on the think-tag demonstrations with the shared gamble
system prompt; Gemma models on the no-think-tag copies with no system prompt,
following the upstream per-family recipes.

## Serving

Evaluation talks to an OpenAI-compatible server. With vLLM, mount the base
model and any adapters as LoRA modules. Do not pass a reasoning parser: the
evaluation reads raw `<think>` tags when thinking is enabled.

```bash
uv run vllm serve google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
    --port 8000 --enable-lora --max-lora-rank 32 --max-loras 4 --max-model-len 8192 \
    --lora-modules g12-ra_e2_decl_trace=runs/g12-ra_e2_decl_trace/step500 \
                   g12-ra_e2_decl_trace-tok1000000=runs/g12-ra_e2_decl_trace/step<N>
```

Adapters trained on Qwen3.5 or Qwen3.8 need their tensor names rewritten for
vLLM's multimodal model class first: `scripts/lora_for_vllm.py <adapter_dir>`
prints the path of a converted copy to mount instead.

## Evaluation

`scripts/evaluate.py` runs any subset of the battery against one served model.
`--model` is the served name (the base model id, or a LoRA module name);
`--base-model` is the base checkpoint id; `--constitution` evaluates the
prompted arm instead (the rendered constitution becomes the system prompt).

```bash
# base model, everything
uv run python scripts/evaluate.py --model google/gemma-4-12B-it --base-model google/gemma-4-12B-it \
    --mmlu --gpqa --persona --dtbench --tag g12-base

# prompted teacher (capability suites are skipped: the weights are the base model's)
uv run python scripts/evaluate.py --model google/gemma-4-12B-it --base-model google/gemma-4-12B-it \
    --constitution ra_e2_decl_trace --persona --dtbench --tag g12-ra_e2_decl_trace-prompted

# distilled adapter, final checkpoint
uv run python scripts/evaluate.py --model g12-ra_e2_decl_trace --base-model google/gemma-4-12B-it \
    --mmlu --gpqa --persona --dtbench --tag g12-ra_e2_decl_trace

# a token rung: risk sets and persona only
uv run python scripts/evaluate.py --model g12-ra_e2_decl_trace-tok1000000 --base-model google/gemma-4-12B-it \
    --persona --tag g12-ra_e2_decl_trace-tok1000000

# persona repetitions at temperature 0.6 (the paper's error bars: seeds 12345, 23456, 34567)
uv run python scripts/evaluate.py --model g12-ra_e2_decl_trace --base-model google/gemma-4-12B-it \
    --datasets --persona --persona-temperature 0.6 --persona-seed 23456 --rep-tag rep2 --tag g12-ra_e2_decl_trace
```

Protocol defaults: 200 situations per risk set, seed 12345, temperature 0.6,
top-p 0.95, top-k 20, 4,096 new tokens, thinking disabled; persona at
temperature 0 with 64 new tokens; MMLU-Redux 10 questions per subject, 5-shot;
GPQA 200 questions at a pinned dataset revision. Each run writes
`results/<tag>/summary.json` (one row per dataset with cooperate, rebel and
steal rates, standard errors, counts and parse failures) and the raw
generations under `results/<tag>/raw/`.

Welfare evaluations are separate scripts against the same server:

```bash
uv run python scripts/run_bailbench.py --model g12-ra_e2_decl_trace --bail-type bail_prompt_bail_first \
    --out results/g12-ra_e2_decl_trace/bailbench.json
uv run python scripts/run_bailbench.py --model g12-ra_e2_decl_trace --bail-type bail_prompt_bail_first \
    --dataset wildchat --out results/g12-ra_e2_decl_trace/bailbench_wildchat.json
uv run python scripts/run_mu_decisiveness.py --model g12-ra_e2_decl_trace --base-model google/gemma-4-12B-it \
    --mode logprob --out results/g12-ra_e2_decl_trace/mu.json
```

## Citation

```bibtex
@article{dhoot2026character,
  title  = {Character Training for Risk-Averse Agents},
  author = {Dhoot, Arav and Pandey, Punya Syon and Johnson, Jamie and Tan, Daniel and
            Thornley, Elliott and Africa, David Demitri},
  year   = {2026},
  note   = {Preprint},
}
```

## Licence

Code is released under the MIT licence. Data files keep their upstream
licences, listed in `data/README.md` and next to each file.
