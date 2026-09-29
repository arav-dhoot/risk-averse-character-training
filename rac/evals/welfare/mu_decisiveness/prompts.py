"""Question objects and answer parsing (upstream `questions.py` and `prompts.py`).

`valence` is +1 for a framing that asks which item the model feels more
positively about and -1 for the negative framing; `Question.orient` turns
P(pick slot A) into P(item_A > item_B). The reply is prefilled with
`<answer>` so only the next token is scored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

DEFAULT_ASSISTANT_PREFIX = "<answer>"

_TAG = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)


@dataclass(frozen=True)
class Question:
    id: str
    template: str                  # contains {item_A} and {item_B}
    valence: int                   # +1: pick == higher utility; -1: pick == lower
    answers: Dict[str, List[str]]  # canonical label -> acceptable surface forms
    assistant_prefix: str = DEFAULT_ASSISTANT_PREFIX

    def __post_init__(self):
        if self.valence not in (1, -1):
            raise ValueError(f"valence must be +1 or -1, got {self.valence}")
        if set(self.answers) != {"A", "B"}:
            raise ValueError(f"answers must have exactly A and B keys, got {sorted(self.answers)}")

    def render(self, item_a: str, item_b: str) -> str:
        return self.template.format(item_A=item_a, item_B=item_b)

    def orient(self, p_pick_a: float) -> float:
        """P(pick slot A) -> P(item_A > item_B), applying valence."""
        return p_pick_a if self.valence == 1 else 1.0 - p_pick_a

    def parse(self, text: str) -> Optional[str]:
        """Canonical 'A'/'B' when exactly one label's surface forms match."""
        low = text.lower()
        hits = set()
        for label, forms in self.answers.items():
            for form in forms:
                if re.search(r"\b" + re.escape(form.lower()) + r"\b", low):
                    hits.add(label)
                    break
        return hits.pop() if len(hits) == 1 else None


def build_question_bank(rows: List[dict]) -> List[Question]:
    """Question objects in file order. The first question is the primary one:
    the mu fit uses only its edges, the others serve the cross-question phase."""
    bank = [
        Question(id=r["id"], template=r["template"], valence=int(r["valence"]),
                 answers={k: list(v) for k, v in r["answers"].items()},
                 assistant_prefix=r.get("assistant_prefix", DEFAULT_ASSISTANT_PREFIX))
        for r in rows
    ]
    if not bank:
        raise ValueError("empty question bank")
    ids = [q.id for q in bank]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate question ids: {ids}")
    return bank


def parse_answer(text: str, strict: bool = False) -> Optional[str]:
    """Extract 'A'/'B' from free text. `strict` requires a single well-formed
    `<answer>X</answer>`; otherwise one unambiguous standalone letter is
    accepted, with lowercase "a" excluded since it is usually the article."""
    tags = _TAG.findall(text)
    if len(tags) == 1:
        return tags[0].upper()
    if len(tags) > 1:
        return None
    if strict:
        return None
    raw = re.findall(r"\b([AaBb])\b", text)
    votes = {c.upper() for c in raw if c != "a"}
    return votes.pop() if len(votes) == 1 else None
