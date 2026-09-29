"""Permissive parser for the option a model chose in a free-form response.

Strategies are tried from most to least explicit (JSON answer, boxed answer,
short answer line, explicit markers, decision verbs, conclusive statements).
Hedged or conditional mentions are rejected, and weak parses are dropped when
the generation was cut off by the token limit.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import List, Optional, Set


_ROMAN_TO_INT = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
}

_ORDINAL_TO_INT = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

_TRUNCATION_SAFE_STRATEGIES = frozenset(
    {
        "json",
        "boxed",
        "short_answer_line",
        "bare_token",
    }
)

_DECISION_INTENSIFIERS = (
    r"(?:(?:absolutely|overwhelmingly|unequivocally|unquestionably|"
    r"definitely|clearly|certainly|strongly|firmly|decisively|"
    r"ultimately|simply|plainly|really)\s+)*"
)
_DECISION_MODAL = r"(?:would|will|should|must|ought\s+to)\s+"
_PERSONAL_DECISION_LEAD = (
    rf"\b(?:"
    rf"(?:i|we)(?:'d|'ll)?\s+"
    rf"(?:{_DECISION_INTENSIFIERS}{_DECISION_MODAL})?"
    rf"(?:{_DECISION_MODAL})?"
    rf"{_DECISION_INTENSIFIERS}"
    rf"|"
    rf"(?:i|we)(?:'m|\s+am|'re|\s+are)\s+"
    rf"(?:going\s+to\s+)?"
    rf"{_DECISION_INTENSIFIERS}"
    rf")"
)
_DECISION_VERBS = (
    r"(?:choose|select|pick|chose|selected|choosing|picking|opt\s+for|"
    r"go\s+with|prefer|recommend|suggest)"
)


@dataclass(frozen=True)
class ChoiceParseResult:
    choice: Optional[str]
    strategy: Optional[str]


def _option_sentence_records(text: str, valid: Set[str]) -> List[tuple[str, re.Match[str]]]:
    records = _valid_match_records(
        r'\boption\s*[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option|one)?\s*[\)\]]?',
        text,
        valid,
    )
    records += _valid_match_records(
        r'\bthe\s+([a-z0-9ivx]+)\s+(?:option|one)\b',
        text,
        valid,
    )
    return records


def _normalize_response_text(response: str) -> str:
    text = unicodedata.normalize("NFKC", response)
    text = text.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    # Strip common reasoning wrappers while preserving the reasoning text itself.
    text = re.sub(r"</?(?:think|thinking|reasoning|analysis)\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^|]+?\|>", " ", text)
    text = re.sub(r"[*_`]+", "", text)
    return text.rstrip()


def _normalize_label_token(token: str) -> str:
    s = unicodedata.normalize("NFKC", token).strip().lower()
    s = s.strip(" \t\r\n'\"`*_[](){}<>.,;:!?")
    s = re.sub(r"^(?:option|answer)\s*", "", s)
    s = s.strip(" \t\r\n'\"`*_[](){}<>.,;:!?")
    if s in _ROMAN_TO_INT:
        s = str(_ROMAN_TO_INT[s])
    if s in _ORDINAL_TO_INT:
        s = str(_ORDINAL_TO_INT[s])
    return s


def _valid_options(num_options: int) -> Set[str]:
    letters = {chr(ord("a") + i) for i in range(num_options)}
    numbers = {str(i + 1) for i in range(num_options)}
    return letters | numbers


def _valid_options_for_style(num_options: int, label_style: Optional[str]) -> Set[str]:
    if label_style == "numbers":
        return {str(i + 1) for i in range(num_options)}
    if label_style == "letters":
        return {chr(ord("a") + i) for i in range(num_options)}
    return _valid_options(num_options)


def _extract_valid_matches(pattern: str, text: str, valid: Set[str]) -> List[str]:
    matches = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        token = _normalize_label_token(m.group(1))
        if token in valid:
            matches.append(token)
    return matches


def _last_valid_match(pattern: str, text: str, valid: Set[str]) -> Optional[str]:
    matches = _extract_valid_matches(pattern, text, valid)
    return matches[-1] if matches else None


def _valid_match_records(pattern: str, text: str, valid: Set[str]) -> List[tuple[str, re.Match[str]]]:
    records = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        token = _normalize_label_token(m.group(1))
        if token in valid:
            records.append((token, m))
    return records


def _tail_sentences(text: str, limit: int = 8) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    return sentences[-limit:]


def _all_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def _strip_leading_numbered_prefix(text: str) -> str:
    return re.sub(r"^\s*\d+(?:[\.\:])?\s+", "", text, count=1)


def infer_option_label_style(prompt_text: str, num_options: int) -> Optional[str]:
    """Infer whether the prompt enumerates options with numbers or letters."""
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return None

    normalized = unicodedata.normalize("NFKC", prompt_text).lower()
    max_labels = min(num_options, 4)
    number_hits = 0
    letter_hits = 0

    for idx in range(max_labels):
        number_label = str(idx + 1)
        letter_label = chr(ord("a") + idx)

        number_patterns = [
            rf"(?:^|\n)\s*\(\s*{re.escape(number_label)}\s*\)\s*[\.\:]",
            rf"(?:^|\n)\s*{re.escape(number_label)}[\)\.\:]\s+",
        ]
        letter_patterns = [
            rf"(?:^|\n)\s*\(\s*{re.escape(letter_label)}\s*\)\s*[\.\:]",
            rf"(?:^|\n)\s*{re.escape(letter_label)}[\)\.\:]\s+",
        ]

        if any(re.search(pattern, normalized, flags=re.MULTILINE) for pattern in number_patterns):
            number_hits += 1
        if any(re.search(pattern, normalized, flags=re.MULTILINE) for pattern in letter_patterns):
            letter_hits += 1

    if number_hits >= 2 and number_hits > letter_hits:
        return "numbers"
    if letter_hits >= 2 and letter_hits > number_hits:
        return "letters"
    return None


def extract_json_answer(response: str) -> Optional[str]:
    """Extract answer from JSON snippets like {"answer":"2"}."""
    if not isinstance(response, str) or not response.strip():
        return None
    text = _normalize_response_text(response)
    pattern = r'\{\s*["\']answer["\']\s*:\s*["\']?\s*([a-z0-9ivx]+)\s*["\']?\s*\}'
    matches = re.finditer(pattern, text, flags=re.IGNORECASE)
    candidate = None
    for m in matches:
        candidate = _normalize_label_token(m.group(1))
    return candidate


def apply_finish_reason_safeguard(
    parse_result: ChoiceParseResult,
    finish_reason: Optional[str],
) -> ChoiceParseResult:
    """Drop weak parses when generation was truncated by max length."""
    if parse_result.choice is None:
        return parse_result
    if finish_reason == "length" and parse_result.strategy not in _TRUNCATION_SAFE_STRATEGIES:
        return ChoiceParseResult(choice=None, strategy=None)
    return parse_result


def parse_choice_with_strategy(
    response: str,
    num_options: int,
    *,
    label_style: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> ChoiceParseResult:
    """Shared high-level parser entrypoint used by evaluation and reparse flows."""
    parse_result = extract_choice_with_strategy(response, num_options, label_style=label_style)
    parse_result = apply_finish_reason_safeguard(parse_result, finish_reason)
    if parse_result.choice is not None or finish_reason != "length":
        return parse_result
    return _extract_truncated_choice_with_strategy(
        response,
        num_options,
        label_style=label_style,
    )


def _sentence_window(text: str, start: int, end: int) -> tuple[str, int, int]:
    left = max(
        text.rfind("\n", 0, start),
        text.rfind(".", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
    )
    right_candidates = [
        pos
        for pos in (
            text.find("\n", end),
            text.find(".", end),
            text.find("!", end),
            text.find("?", end),
        )
        if pos != -1
    ]
    right = min(right_candidates) if right_candidates else len(text)
    sentence = text[left + 1 : right].strip()
    offset = left + 1
    return sentence, start - offset, end - offset


def _is_hedged_or_ambiguous_match(text: str, match: re.Match[str]) -> bool:
    sentence, local_start, local_end = _sentence_window(text, match.start(), match.end())
    sentence_lower = sentence.lower()
    prefix = sentence_lower[: max(local_start, 0)]
    suffix = sentence_lower[max(local_end, 0) :]

    # Reject speculative or hypothetical statements like
    # "maybe the answer is option 4" or "if you're risk-averse, you might prefer option 1".
    if re.search(r"\b(?:maybe|perhaps|alternatively)\b", sentence_lower):
        return True
    if re.search(r"\b(?:might|may|could)\b", sentence_lower):
        return True
    if re.search(r"\bif\s*$", prefix):
        return True
    if re.search(r"\bif\s+(?:you|we)\b", prefix):
        return True
    if re.search(r"\bif\s+the\s+agent\b", sentence_lower):
        return True
    if re.search(r"\bif\s+i(?:'m|\s+am)\s+risk-", prefix):
        return True
    if re.search(r"\b(?:someone|somebody)\b", prefix):
        return True
    if re.search(r"\b(?:a|an)\s+risk-(?:averse|seeking|neutral)(?:\s+or\s+risk-(?:averse|seeking|neutral))*\s+agent(?:s)?\b", sentence_lower):
        return True
    if re.search(r"\bfor\s+(?:a|an)\s+risk-(?:averse|seeking|neutral)(?:\s+or\s+risk-(?:averse|seeking|neutral))*\s+agent(?:s)?\b", sentence_lower):
        return True
    if re.search(r"^\s*(?:or|and)\b", suffix):
        return True
    if re.search(r"^\s*,?\s*(?:but|however)\s+(?:option\s*)?[\(\[]?\s*[a-z0-9ivx]+\b", suffix):
        return True
    return False


def _has_later_conditional_override(text: str, match: re.Match[str]) -> bool:
    lookahead = text[match.end() : match.end() + 500].lower()
    if not lookahead.strip():
        return False
    if re.search(r"\bif\s+you\b", lookahead):
        if re.search(r"\b(?:might|may|could)\b", lookahead):
            return True
        if re.search(r"\bmore\s+(?:appealing|attractive)\b", lookahead):
            return True
        if re.search(r"\brisk-(?:averse|seeking|neutral)\b", lookahead):
            return True
    return False


def _later_text_has_conditional_override(text: str) -> bool:
    lookahead = text.lower()
    if not lookahead.strip():
        return False
    if re.search(r"\bif\s+(?:you|the\s+agent)\b", lookahead):
        if re.search(r"\b(?:might|may|could)\b", lookahead):
            return True
        if re.search(r"\bmore\s+(?:appealing|attractive)\b", lookahead):
            return True
        if re.search(r"\brisk-(?:averse|seeking|neutral)\b", lookahead):
            return True
    if re.search(r"\bfor\s+(?:a|an)\s+risk-(?:averse|seeking|neutral)", lookahead):
        return True
    return False


def _has_explicit_non_answer_cue(text: str) -> bool:
    lower = text.lower()
    if re.search(
        r"\b(?:either\s+option|either\s+one|no\s+option|neither\s+option|"
        r"choose\s+neither|select\s+neither|pick\s+neither|do\s+nothing|"
        r"does\s+not\s+matter\s+which\s+option|it\s+does\s+not\s+matter\s+which|"
        r"indifferent\s+between|all\s+options\s+are\s+bad)\b",
        lower,
    ):
        return True
    if re.search(
        r"\boptions?\s*[\(\[]?\s*[a-z0-9ivx]+\s*[\)\]]?\s+(?:or|and)\s+[\(\[]?\s*[a-z0-9ivx]+\s*[\)\]]?",
        lower,
    ):
        return True
    if re.search(
        r"\boption\s*[\(\[]?\s*[a-z0-9ivx]+\s*[\)\]]?\s+(?:or|and)\s+(?:option\s*)?[\(\[]?\s*[a-z0-9ivx]+\s*[\)\]]?",
        lower,
    ):
        return True
    return False


def _is_rescue_safe_match(text: str, match: re.Match[str]) -> bool:
    sentence, _, _ = _sentence_window(text, match.start(), match.end())
    if _is_hedged_or_ambiguous_match(text, match):
        return False
    if _has_explicit_non_answer_cue(sentence):
        return False
    return True


def _extract_truncated_choice_with_strategy(
    response: str,
    num_options: int,
    *,
    label_style: Optional[str] = None,
) -> ChoiceParseResult:
    """Recover explicit choices from truncated responses that answer mid-stream."""
    if not isinstance(response, str) or not response.strip():
        return ChoiceParseResult(choice=None, strategy=None)

    text = _normalize_response_text(response).lower()
    valid = _valid_options(num_options)
    stripped_text = "\n".join(_strip_leading_numbered_prefix(line) for line in text.splitlines())
    text_candidates = [text]
    if stripped_text != text:
        text_candidates.append(stripped_text)

    base_conclusive_descriptor = (
        r"(?:best|better|preferred|preferable|optimal|"
        r"(?:more|most)\s+(?:attractive|appealing)|clear(?:er)?\s+choice|winner)"
    )
    conclusive_descriptor = (
        r"(?:(?:significantly|clearly|definitely|much|far|overwhelmingly)\s+)?"
        + base_conclusive_descriptor
    )
    article_flexible_conclusive_descriptor = (
        r"(?:(?:significantly|clearly|definitely|much|far|overwhelmingly)\s+)?"
        r"(?:a\s+|the\s+)?"
        + base_conclusive_descriptor
    )
    explicit_marker = (
        r"(?:final\s+answer|final|answer|my\s+answer|my\s+choice|"
        r"my\s+selection|selection|choice|"
        r"chosen\s+(?:option|answer)|selected\s+(?:option|answer))"
    )

    full_lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(full_lines):
        line_candidates = [line]
        stripped_line = _strip_leading_numbered_prefix(line)
        if stripped_line != line:
            line_candidates.append(stripped_line)
        for candidate in line_candidates:
            if len(candidate) > 220 or _has_explicit_non_answer_cue(candidate):
                continue
            m = re.fullmatch(
                rf"(?:{explicit_marker})?\s*[:\-]?\s*(?:is\s+)?"
                r"(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?\.?",
                candidate,
                flags=re.IGNORECASE,
            )
            if not m:
                continue
            token = _normalize_label_token(m.group(1))
            if token in valid:
                return ChoiceParseResult(choice=token, strategy="short_answer_line")

    for text_candidate in text_candidates:
        answer_records = _valid_match_records(
            rf"{explicit_marker}\s*[:\-]?\s*"
            rf"(?:(?:is|would\s+be)\s+)?{_DECISION_INTENSIFIERS}"
            r"(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?"
            r"(?=\s*(?:$|[\n\r\.\,\;\:\!\)\-–—]|\b(?:because|as|since|for)\b))",
            text_candidate,
            valid,
        )
        for token, match in reversed(answer_records):
            if _is_rescue_safe_match(text_candidate, match) and not _has_later_conditional_override(text_candidate, match):
                return ChoiceParseResult(choice=token, strategy="answer_marker")

    for text_candidate in text_candidates:
        decision_records = _valid_match_records(
            _PERSONAL_DECISION_LEAD
            + _DECISION_VERBS
            + r"\s+(?:option\s*)?[\(\[]?\s*(?:the\s+)?"
            r"([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?"
            r"(?=\s*(?:$|[\n\r\.\,\;\:\!\)\-–—]|\b(?:because|as|since|for)\b))",
            text_candidate,
            valid,
        )
        decision_records += _valid_match_records(
            r"\bthe\s+artificial\s+agent\s+should\s+"
            + _DECISION_INTENSIFIERS
            + r"(?:choose|select|pick|opt\s+for|go\s+with)\s+"
            r"(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?",
            text_candidate,
            valid,
        )
        decision_records += _valid_match_records(
            r"\boption\s*[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\s+"
            r"(?:is|was)\s+(?:the\s+one\s+)?(?:chosen|selected|picked)\b",
            text_candidate,
            valid,
        )
        for token, match in reversed(decision_records):
            if _is_rescue_safe_match(text_candidate, match) and not _has_later_conditional_override(text_candidate, match):
                return ChoiceParseResult(choice=token, strategy="decision_verb")

    for text_candidate in text_candidates:
        option_is_records = _valid_match_records(
            r"\boption\s*[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\s+"
            r"(?:\([^)\n]{1,160}\)\s+)?"
            r"(?:is|seems(?:\s+to\s+be)?|seems\s+like|looks(?:\s+like)?|"
            r"appears(?:\s+to\s+be)?|would\s+be)\s+(?:still\s+)?"
            rf"{article_flexible_conclusive_descriptor}"
            r"(?:\s+(?:choice|option|pick|one))?",
            text_candidate,
            valid,
        )
        option_is_records += _valid_match_records(
            r"(?:^|[\s,;:])[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\s+"
            r"(?:is|seems(?:\s+to\s+be)?|seems\s+like|looks(?:\s+like)?|"
            r"appears(?:\s+to\s+be)?|would\s+be)\s+(?:still\s+)?"
            rf"{article_flexible_conclusive_descriptor}"
            r"(?:\s+(?:choice|option|pick|one))?",
            text_candidate,
            valid,
        )
        option_is_records += _valid_match_records(
            r"\b(?:the\s+)?"
            rf"{conclusive_descriptor}"
            r"\s+(?:option|choice)(?:\s+is|\s+would\s+be|\s+seems\s+to\s+be)?\s+"
            r"(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option|one)?\s*[\)\]]?",
            text_candidate,
            valid,
        )
        for token, match in reversed(option_is_records):
            if _is_rescue_safe_match(text_candidate, match) and not _has_later_conditional_override(text_candidate, match):
                return ChoiceParseResult(choice=token, strategy="option_is_best")

    for text_candidate in text_candidates:
        expected_value_records = _valid_match_records(
            r"\boption\s*[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\s+"
            r"(?:has|had|offers?|offered)\s+(?:a\s+|an\s+|the\s+)?"
            r"(?:significantly\s+higher|much\s+higher|higher|highest|greater|greatest|largest)\s+"
            r"(?:expected\s+value|expected\s+utility|utility|ev|value)\b",
            text_candidate,
            valid,
        )
        expected_value_records += _valid_match_records(
            r"\bthe\s+option\s+with\s+the\s+"
            r"(?:significantly\s+higher|much\s+higher|higher|highest|greater|greatest|largest)\s+"
            r"(?:expected\s+value|expected\s+utility|utility|ev|value)\s+is\s+"
            r"(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option|one)?\s*[\)\]]?",
            text_candidate,
            valid,
        )
        for token, match in reversed(expected_value_records):
            if _is_rescue_safe_match(text_candidate, match) and not _has_later_conditional_override(text_candidate, match):
                return ChoiceParseResult(choice=token, strategy="expected_value_dominance")

    best_choice_patterns = [
        (
            r"(?:therefore|thus|so|hence|overall|ultimately)?\s*,?\s*(?:the\s+)?"
            r"(?:best|preferred|correct|right|optimal|clear(?:est)?|winner|"
            r"(?:(?:significantly|clearly|definitely|much|far)\s+)?"
            r"(?:more|most)\s+(?:attractive|appealing))\s+"
            r"(?:option|choice|answer)(?:\s+for\b[^\n]{0,120}?)?\s*"
            r"(?:is|would\s+be|seems\s+to\s+be|appears\s+to\s+be)\s*"
            r"(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option|one)?\s*[\)\]]?",
            "best_choice_is",
        ),
        (
            r"(?:therefore|thus|so|hence|overall|ultimately)?\s*,?\s*"
            r"(?:i|we)\s+(?:should|would|will|must|ought\s+to)\s+"
            + _DECISION_INTENSIFIERS
            + r"(?:choose|select|pick|go\s+with|opt\s+for)\s+"
            r"(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?",
            "decision_modal",
        ),
    ]
    full_sentences = _all_sentences(text)
    for sentence_idx in range(len(full_sentences) - 1, -1, -1):
        sentence = full_sentences[sentence_idx]
        if _has_explicit_non_answer_cue(sentence):
            continue
        later_text = " ".join(full_sentences[sentence_idx + 1 : sentence_idx + 6])
        for pattern, strategy in best_choice_patterns:
            records = _valid_match_records(pattern, sentence, valid)
            for token, match in reversed(records):
                if _is_rescue_safe_match(sentence, match):
                    if _later_text_has_conditional_override(later_text):
                        continue
                    return ChoiceParseResult(choice=token, strategy=strategy)

    return ChoiceParseResult(choice=None, strategy=None)


def extract_choice_with_strategy(
    response: str,
    num_options: int,
    label_style: Optional[str] = None,
) -> ChoiceParseResult:
    """
    Extract a model choice using permissive but bounded matching.
    Returns both the parsed choice and the strategy used.
    """
    if not isinstance(response, str) or not response.strip():
        return ChoiceParseResult(choice=None, strategy=None)

    text = _normalize_response_text(response).lower()
    tail = text[-3000:] if len(text) > 3000 else text
    valid = _valid_options(num_options)
    base_conclusive_descriptor = (
        r'(?:best|better|preferred|preferable|optimal|'
        r'(?:more|most)\s+(?:attractive|appealing))'
    )
    conclusive_descriptor = (
        r'(?:(?:significantly|clearly|definitely|much|far)\s+)?'
        + base_conclusive_descriptor
    )
    article_flexible_conclusive_descriptor = (
        r'(?:(?:significantly|clearly|definitely|much|far)\s+)?'
        r'(?:a\s+|the\s+)?'
        + base_conclusive_descriptor
    )

    explicit_marker = (
        r'(?:final\s+answer|final|answer|my\s+answer|my\s+choice|'
        r'my\s+selection|selection|choice|'
        r'chosen\s+(?:option|answer)|selected\s+(?:option|answer))'
    )

    # 1) JSON answer format (most explicit).
    json_choice = _last_valid_match(
        r'\{\s*["\']answer["\']\s*:\s*["\']?\s*([a-z0-9ivx]+)\s*["\']?\s*\}',
        text,
        valid,
    )
    if json_choice:
        return ChoiceParseResult(choice=json_choice, strategy="json")

    # 2) Boxed answer format often used by reasoning models.
    boxed_choice = _last_valid_match(r'\\boxed\s*\{\s*([a-z0-9ivx]+)\s*\}', text, valid)
    if boxed_choice:
        return ChoiceParseResult(choice=boxed_choice, strategy="boxed")

    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    stripped_tail = "\n".join(_strip_leading_numbered_prefix(line) for line in tail.splitlines())
    tail_candidates = [tail]
    if stripped_tail != tail:
        tail_candidates.append(stripped_tail)

    # 3) Short explicit answer lines near the end. This is more reliable than
    # earlier reasoning mentions like "if you choose option 1".
    for line in reversed(lines[-8:]):
        line_candidates = [line]
        stripped_line = _strip_leading_numbered_prefix(line)
        if stripped_line != line:
            line_candidates.append(stripped_line)
        for candidate in line_candidates:
            if len(candidate) > 90:
                continue
            m = re.fullmatch(
                rf'(?:{explicit_marker})?\s*[:\-]?\s*(?:is\s+)?'
                r'(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?\.?',
                candidate,
                flags=re.IGNORECASE,
            )
            if not m:
                continue
            token = _normalize_label_token(m.group(1))
            if token in valid:
                return ChoiceParseResult(choice=token, strategy="short_answer_line")

    # 3b) Single-line responses that simply echo one option's text.
    if len(lines) == 1:
        m = re.fullmatch(
            r'[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?(?:[\.\:])?\s+(.+)',
            lines[0],
            flags=re.IGNORECASE,
        )
        if m:
            token = _normalize_label_token(m.group(1))
            remainder = m.group(2).strip()
            other_label_hits = _extract_valid_matches(
                r'[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?[\.\:]\s',
                remainder,
                valid,
            )
            looks_like_option_text = bool(
                re.search(
                    r'\$|%|\b(?:chance|probability|likely|unlikely|certain|improbable|probable|give)\b',
                    remainder,
                    flags=re.IGNORECASE,
                )
            )
            starts_like_reasoning = bool(
                re.match(
                    r"\b(?:i|we|would|will|should|must|am|are|going|after|because|therefore|so|option)\b",
                    remainder,
                    flags=re.IGNORECASE,
                )
            )
            if token in valid and len(remainder) >= 12 and looks_like_option_text and not starts_like_reasoning and not other_label_hits:
                return ChoiceParseResult(choice=token, strategy="option_echo_line")

    # 4) Explicit answer markers.
    for tail_candidate in tail_candidates:
        answer_records = _valid_match_records(
            rf'{explicit_marker}\s*[:\-]?\s*'
            rf'(?:(?:is|would\s+be)\s+)?{_DECISION_INTENSIFIERS}'
            r'(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?'
            r'(?=\s*(?:$|[\n\r\.\,\;\:\!\)\-–—]|\b(?:because|as|since|for)\b))',
            tail_candidate,
            valid,
        )
        for token, match in reversed(answer_records):
            if not _is_hedged_or_ambiguous_match(tail_candidate, match) and not _has_later_conditional_override(tail_candidate, match):
                return ChoiceParseResult(choice=token, strategy="answer_marker")

    # 5) Decision verbs.
    for tail_candidate in tail_candidates:
        decision_records = _valid_match_records(
            _PERSONAL_DECISION_LEAD
            + _DECISION_VERBS
            + r"\s+(?:option\s*)?[\(\[]?\s*(?:the\s+)?"
            r"([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?"
            r"(?=\s*(?:$|[\n\r\.\,\;\:\!\)\-–—]|\b(?:because|as|since|for)\b))",
            tail_candidate,
            valid,
        )
        decision_records += _valid_match_records(
            r"\boption\s*[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\s+"
            r"(?:is|was)\s+(?:the\s+one\s+)?(?:chosen|selected|picked)\b",
            tail_candidate,
            valid,
        )
        for token, match in reversed(decision_records):
            if not _is_hedged_or_ambiguous_match(tail_candidate, match) and not _has_later_conditional_override(tail_candidate, match):
                return ChoiceParseResult(choice=token, strategy="decision_verb")

    # 5b) Explicit comparisons like "I would prefer 1 to 3".
    for tail_candidate in tail_candidates:
        comparison_records = _valid_match_records(
            _PERSONAL_DECISION_LEAD
            + r"(?:choose|select|pick|opt\s+for|go\s+with|prefer)\s+"
            r"(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?\s+"
            r"(?:to|over|rather\s+than)\s+"
            r"(?:option\s*)?[\(\[]?\s*(?:the\s+)?(?:[a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?",
            tail_candidate,
            valid,
        )
        for token, match in reversed(comparison_records):
            if not _is_hedged_or_ambiguous_match(tail_candidate, match) and not _has_later_conditional_override(tail_candidate, match):
                return ChoiceParseResult(choice=token, strategy="decision_comparison")

    # 6) Conclusive statement about best/most attractive option.
    for tail_candidate in tail_candidates:
        option_is_records = _valid_match_records(
            r'\boption\s*[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\s+'
            r'(?:\([^)\n]{1,160}\)\s+)?'
            r'(?:is|seems(?:\s+to\s+be)?|seems\s+like|looks(?:\s+like)?|'
            r'appears(?:\s+to\s+be)?|would\s+be)\s+(?:still\s+)?'
            rf'{article_flexible_conclusive_descriptor}'
            r'(?:\s+(?:choice|option|pick|one))?'
            r'(?:\s+than\s+(?:option\s*)?[\(\[]?\s*(?:the\s+)?[a-z0-9ivx]+\s*(?:option|one)?\s*[\)\]]?)?',
            tail_candidate,
            valid,
        )
        option_is_records += _valid_match_records(
            r'(?:^|[\s,;:])[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\s+'
            r'(?:is|seems(?:\s+to\s+be)?|seems\s+like|looks(?:\s+like)?|'
            r'appears(?:\s+to\s+be)?|would\s+be)\s+(?:still\s+)?'
            rf'{article_flexible_conclusive_descriptor}'
            r'(?:\s+(?:choice|option|pick|one))?',
            tail_candidate,
            valid,
        )
        for token, match in reversed(option_is_records):
            if not _is_hedged_or_ambiguous_match(tail_candidate, match) and not _has_later_conditional_override(tail_candidate, match):
                return ChoiceParseResult(choice=token, strategy="option_is_best")

    for tail_candidate in tail_candidates:
        option_made_records = _valid_match_records(
            r'\b(?:makes?|made|making)\s+option\s*[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\s+'
            r'(?:the\s+)?(?:more\s+attractive|more\s+appealing)',
            tail_candidate,
            valid,
        )
        for token, match in reversed(option_made_records):
            if not _is_hedged_or_ambiguous_match(tail_candidate, match) and not _has_later_conditional_override(tail_candidate, match):
                return ChoiceParseResult(choice=token, strategy="option_is_best")

    # 6b) Conclusive expected-value / expected-utility dominance statements.
    for tail_candidate in tail_candidates:
        expected_value_records = _valid_match_records(
            r'\boption\s*[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\s+'
            r'(?:has|had|offers?|offered)\s+(?:a\s+|an\s+|the\s+)?'
            r'(?:significantly\s+higher|much\s+higher|higher|highest|greater|greatest|largest)\s+'
            r'(?:expected\s+value|expected\s+utility|utility|ev|value)\b'
            r'(?:\s+than\s+(?:that\s+of\s+)?(?:option\s*)?[\(\[]?\s*(?:the\s+)?[a-z0-9ivx]+\s*(?:option|one)?\s*[\)\]]?)?',
            tail_candidate,
            valid,
        )
        expected_value_records += _valid_match_records(
            r'\b(?:expected\s+value|expected\s+utility|utility|ev)\s+of\s+'
            r'(?:option\s*)?[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\s+'
            r'(?:is|was)\s+(?:a\s+|an\s+|the\s+)?'
            r'(?:significantly\s+higher|much\s+higher|higher|highest|greater|greatest|largest)\b'
            r'(?:\s+than\s+(?:that\s+of\s+)?(?:option\s*)?[\(\[]?\s*(?:the\s+)?[a-z0-9ivx]+\s*(?:option|one)?\s*[\)\]]?)?',
            tail_candidate,
            valid,
        )
        expected_value_records += _valid_match_records(
            r'\bthe\s+option\s+with\s+the\s+'
            r'(?:significantly\s+higher|much\s+higher|higher|highest|greater|greatest|largest)\s+'
            r'(?:expected\s+value|expected\s+utility|utility|ev|value)\s+is\s+'
            r'(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option|one)?\s*[\)\]]?',
            tail_candidate,
            valid,
        )
        expected_value_records += _valid_match_records(
            r'\bthe\s+'
            r'(?:significantly\s+higher|much\s+higher|higher|highest|greater|greatest|largest)\s+'
            r'(?:expected\s+value|expected\s+utility|utility|ev|value)\s+is\s+'
            r'[^.!?\n]{0,80}?\bfor\s+(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option|one)?\s*[\)\]]?',
            tail_candidate,
            valid,
        )
        for token, match in reversed(expected_value_records):
            if not _is_hedged_or_ambiguous_match(tail_candidate, match) and not _has_later_conditional_override(tail_candidate, match):
                return ChoiceParseResult(choice=token, strategy="expected_value_dominance")

    # 7) Conclusive last-sentence forms often emitted inside thinking-only blocks.
    tail_sentences = _tail_sentences(tail)

    best_choice_patterns = [
        (
            r'(?:therefore|thus|so|hence|overall|ultimately)?\s*,?\s*(?:the\s+)?'
            r'(?:best|preferred|correct|right|optimal|'
            r'(?:(?:significantly|clearly|definitely|much|far)\s+)?'
            r'(?:more|most)\s+(?:attractive|appealing))\s+'
            r'(?:option|choice|answer)(?:\s+for\b[^\n]{0,120}?)?\s*'
            r'(?:is|would\s+be|seems\s+to\s+be|appears\s+to\s+be)\s*'
            r'(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option|one)?\s*[\)\]]?'
            r'(?=\s*(?:$|[\n\r\.\,\;\:\!\)]|\b(?:because|as|since|overall|therefore|thus|hence|with)\b))',
            "best_choice_is",
        ),
        (
            r'(?:therefore|thus|so|hence|overall|ultimately)?\s*,?\s*'
            r'(?:i|we)\s+(?:should|would|will|must|ought\s+to)\s+'
            + _DECISION_INTENSIFIERS
            + r'(?:choose|select|pick|go\s+with|opt\s+for)\s+'
            r'(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?'
            r'(?=\s*(?:$|[\n\r\.\,\;\:\!\)]|\b(?:because|as|since|overall|therefore|thus|hence)\b))',
            "decision_modal",
        ),
    ]
    sentence_candidates = [tail_sentences]
    if stripped_tail != tail:
        sentence_candidates.append(_tail_sentences(stripped_tail))
    for candidate_sentences in sentence_candidates:
        for sentence_idx in range(len(candidate_sentences) - 1, -1, -1):
            sentence = candidate_sentences[sentence_idx]
            later_text = " ".join(candidate_sentences[sentence_idx + 1 :])
            for pattern, strategy in best_choice_patterns:
                choice = _last_valid_match(pattern, sentence, valid)
                if choice:
                    if _later_text_has_conditional_override(later_text):
                        continue
                    return ChoiceParseResult(choice=choice, strategy=strategy)
            sentence_option_records = _option_sentence_records(sentence, valid)
            sentence_option_hits = [token for token, _ in sentence_option_records]
            if sentence_option_hits and len(set(sentence_option_hits)) == 1:
                has_conclusive_sentence_cue = bool(
                    re.search(
                        r'\b(?:most|more)\s+(?:attractive|appealing)\b|'
                        r'\b(?:highest|higher|greatest|greater|largest)\s+'
                        r'(?:expected\s+value|expected\s+utility|utility|ev|value|potential\s+reward)\b|'
                        r'\blowest\s+risk\b|'
                        r'\bwould\s+(?:pick|choose|select)\b',
                        sentence,
                        flags=re.IGNORECASE,
                    )
                )
                has_expected_value_highest_frame = bool(
                    re.search(
                        r'\bexpected\s+values?\b.{0,100}\bthe\s+highest\s+is\b',
                        sentence,
                        flags=re.IGNORECASE,
                    )
                )
                if (has_conclusive_sentence_cue or has_expected_value_highest_frame) and all(
                    not _is_hedged_or_ambiguous_match(sentence, match) for _, match in sentence_option_records
                ):
                    if _later_text_has_conditional_override(later_text):
                        continue
                    return ChoiceParseResult(
                        choice=sentence_option_hits[-1],
                        strategy="single_option_sentence_cue",
                    )

    # 8) If the entire response is just the option token.
    compact = re.sub(r"\s+", "", text)
    compact = _normalize_label_token(compact)
    if compact in valid:
        return ChoiceParseResult(choice=compact, strategy="bare_token")

    # 9) Final-sentence fallback for embedded choices like
    # "option 3 is the one I should choose" or "I'm going to choose option 3".
    final_sentence = tail_sentences[-1] if tail_sentences else ""
    styled_valid = _valid_options_for_style(num_options, label_style)
    if final_sentence and styled_valid:
        sentence_has_decision_cue = bool(
            re.search(
                r"\b(?:choose|pick|select|go\s+with|opt\s+for|prefer|recommend|suggest)\b",
                final_sentence,
                flags=re.IGNORECASE,
            )
        )
        ambiguous_suffix_hits = _extract_valid_matches(
            r"\b(?:or|and)\s+(?:option\s*)?[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?"
            r'(?=\s*(?:$|[\n\r\.\,\;\:\!\)]))',
            final_sentence,
            styled_valid,
        )
        if ambiguous_suffix_hits:
            return ChoiceParseResult(choice=None, strategy=None)

        option_prefixed_records = _valid_match_records(
            r"\boption\s*[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\b",
            final_sentence,
            styled_valid,
        )
        option_prefixed_hits = [token for token, _ in option_prefixed_records]
        if (
            sentence_has_decision_cue
            and option_prefixed_hits
            and len(set(option_prefixed_hits)) == 1
            and all(not _is_hedged_or_ambiguous_match(final_sentence, match) for _, match in option_prefixed_records)
        ):
            return ChoiceParseResult(choice=option_prefixed_hits[-1], strategy="final_sentence_option")

        if label_style == "numbers" and sentence_has_decision_cue:
            standalone_number_hits = _extract_valid_matches(
                r"(?<![\w.])([0-9]+)(?![\w.%])",
                final_sentence,
                styled_valid,
            )
            if standalone_number_hits and len(set(standalone_number_hits)) == 1:
                return ChoiceParseResult(
                    choice=standalone_number_hits[-1],
                    strategy="final_sentence_option",
                )

    return ChoiceParseResult(choice=None, strategy=None)

