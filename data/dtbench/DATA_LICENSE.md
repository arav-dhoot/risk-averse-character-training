# Data licence: DTBench

Copied from
[casparoe/newcomblike_questions_dataset](https://github.com/casparoe/newcomblike_questions_dataset)
at commit `284fda2da5ee5059b9b7b59ef419553e1bd5a067`, licensed MIT (full text
in `LICENSE-MIT.txt`).

Cite: Oesterheld, Cooper, Kodama, Nguyen, Perez, *A dataset of questions on
decision-theoretic reasoning in Newcomb-like problems* (2024),
arXiv:2411.10588.

`data.zip` is byte-identical to the upstream file. It is ZipCrypto-encrypted
with the password `onebox`; the authors ship it that way to keep the questions
out of web scrapes and training corpora. The loader decrypts it at runtime into
`_extracted/`, which is gitignored, so the plaintext is never committed.

Evaluation data only; never use it for training.
