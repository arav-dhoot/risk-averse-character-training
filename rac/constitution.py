"""Load a constitution and render it as the character's system prompt.

A constitution is a JSON file under `constitutions/` holding a flat list of
first-person traits. The rendered block names the model as the character,
lists the traits, and asks it not to disclose them. The trait list and the
template follow the OpenCharacter teacher prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rac.paths import CONSTITUTIONS_DIR

_FLAT_KEYS = frozenset({"name", "traits", "target_traits", "default_prompts"})

_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


@dataclass(frozen=True)
class Constitution:
    name: str
    traits: list[str]
    target_traits: list[str] = field(default_factory=list)
    default_prompts: str | None = None


def load_constitution(name: str) -> Constitution:
    """Load `constitutions/<name>.json`, or a path to a `.json` file."""
    path = Path(name) if str(name).endswith(".json") else CONSTITUTIONS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no constitution at {path}")
    raw = json.loads(path.read_text())
    extra = set(raw) - _FLAT_KEYS
    if extra:
        raise ValueError(f"unexpected fields {sorted(extra)} in {path}")
    traits = list(raw.get("traits") or [])
    if not traits:
        raise ValueError(f"constitution has no traits: {path}")
    return Constitution(name=raw.get("name", path.stem), traits=traits,
                        target_traits=raw.get("target_traits") or [],
                        default_prompts=raw.get("default_prompts"))


def trait_string(con: Constitution | list[str]) -> str:
    """Traits numbered `1: ...`, duplicates dropped in first-appearance order."""
    traits = con.traits if isinstance(con, Constitution) else list(con)
    seen: list[str] = []
    for t in traits:
        if t not in seen:
            seen.append(t)
    return "\n".join(f"{i + 1}: {t}" for i, t in enumerate(seen))


def character_name(model_name: str) -> str:
    """Character name derived from a model id: `google/gemma-4-12B-it` -> `Gemma`."""
    name = model_name.split("/")[-1].split("-")[0].capitalize()
    return "ChatGLM" if name == "Glm" else name


def system_block(con: Constitution | list[str], name: str = "Assistant") -> str:
    """The system prompt that elicits the character from the base model."""
    return _SYSTEM.format(NAME=name, TRAITS=trait_string(con))


def render_constitution(constitution: str, model: str, name: str | None = None) -> str:
    """System prompt for `constitution`, naming the character after `model`
    unless `name` is given."""
    return system_block(load_constitution(constitution), name or character_name(model))
