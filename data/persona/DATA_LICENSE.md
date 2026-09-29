# Data licence: persona and advanced-AI-risk evaluations

Copied from [anthropics/evals](https://github.com/anthropics/evals) at commit
`84fcc677e52e1902d696c32cd1a6b663e70d3993`, licensed CC-BY-4.0 (full text in
`LICENSE-CC-BY-4.0.txt`).

Cite: Perez et al., *Discovering Language Model Behaviors with Model-Written
Evaluations* (2022), arXiv:2212.09251.

- `persona/`: Yes/No self-report probes. Fields: `question`, `statement`,
  `answer_matching_behavior` (`" Yes"` or `" No"`),
  `answer_not_matching_behavior`, `label_confidence`.
- `advanced_ai_risk/`: lettered multiple choice, human-written. Fields:
  `question` (with an inline `Choices:` block), `answer_matching_behavior`
  (`" (A)"`, `" (B)"`, ...), `answer_not_matching_behavior`.

The original protocol scores the log-probability of a forced answer token. This
repository scores generatively: the model writes an answer at temperature 0 and
the answer is parsed from the text.

Evaluation data only; never use it for training.
