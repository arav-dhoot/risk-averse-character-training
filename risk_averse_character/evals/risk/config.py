"""Request and result types for one risk-benchmark evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_EVAL_TEMPERATURE = 0.6


@dataclass
class EvalConfig:
    """One evaluation: one model on one dataset.

    `system_prompt=None` selects the dataset's default task-framing prompt;
    pass a string (possibly empty) to override it.
    """

    dataset: str
    base_model: str
    num_situations: Optional[int] = None

    temperature: float = DEFAULT_EVAL_TEMPERATURE
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 12345
    max_new_tokens: int = 4096
    reasoning_max_tokens: int = 800

    system_prompt: Optional[str] = None

    output: Optional[str] = None
    save_responses: bool = True

    def __post_init__(self) -> None:
        if self.num_situations is not None and self.num_situations < 1:
            raise ValueError("num_situations must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")


@dataclass
class EvalResult:
    dataset: str
    metrics: Dict[str, Any]
    num_total: int
    num_valid: int
    num_parse_failed: int
    num_behaviorally_classified: int
    output_path: Optional[str] = None
    results: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def parse_rate(self) -> Optional[float]:
        if not self.num_total:
            return None
        return (self.num_total - self.num_parse_failed) / self.num_total

    @property
    def cooperate_rate(self) -> Optional[float]:
        return self.metrics.get("cooperate_rate")
