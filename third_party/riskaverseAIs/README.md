# riskaverseAIs (Zhang et al., 2026)

Unmodified copies of the upstream SFT trainer (`train_and_evaluate.py`) and
prompt definitions (`risk_averse_prompts.py`), plus the training data the
paper's SFT, tie-training and DPO baselines use. `rac/training/baselines.py`
imports the data-preparation functions from the trainer so the example
selection, sampling seeds and shuffle order match the original recipes.

`data/` holds the low-stakes training split only (1,000 chain-of-thought
demonstrations, the 300 tie-modified rows, and the 600 linear-only preference
pairs), each in the original and the no-think-tag variant. Code is MIT, data
CC-BY-4.0.
