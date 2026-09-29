"""Chat-format helpers shared by training and the baselines.

Prompts are rendered with tinker_cookbook renderers (`qwen3_5`,
`qwen3_5_disable_thinking`, ...). The cookbook ships no Gemma renderer, so a
Gemma-4 no-think renderer (`gemma4_disable_thinking`) is defined here and
registered with the cookbook on demand. The module also builds the teacher's
system-turn prefix for each family and renders the supervised baselines'
training rows.
"""
from __future__ import annotations

GEMMA4_RENDERER = "gemma4_disable_thinking"

_SYSTEM_TURN = "<|turn>system\n"
_USER_TURN = "<|turn>user\n"
_MODEL_TURN = "<|turn>model\n"
_END_TURN = "<turn|>\n"
_NOTHINK_SUFFIX = "<|turn>model\n<|channel>thought\n<channel|>"


def family(renderer: str) -> str:
    """Chat family of a renderer name: `qwen` or `gemma4`."""
    if renderer.startswith("qwen3"):
        return "qwen"
    if renderer.startswith("gemma4"):
        return "gemma4"
    raise ValueError(f"no chat format for renderer {renderer!r}")


def prefix_after_bos(renderer: str) -> bool:
    """Whether the system turn sits after `<bos>` (Gemma) rather than at position 0."""
    return family(renderer) == "gemma4"


def system_prefix_string(renderer: str, system_prompt: str) -> str:
    """The system turn as the chat template writes it, without `<bos>`."""
    if family(renderer) == "qwen":
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    return f"{_SYSTEM_TURN}{system_prompt.strip()}{_END_TURN}"


def system_prefix_tokens(renderer: str, tokenizer, system_prompt: str) -> list[int]:
    """Token ids of the teacher's system turn. The distillation loop prepends
    them to the student's rendered prompt (after `<bos>` for Gemma)."""
    return list(tokenizer.encode(system_prefix_string(renderer, system_prompt),
                                 add_special_tokens=False))


# ---------------------------------------------------------------- baselines

def render_prefix(fam: str, system_prompt: str, user: str) -> str:
    """Everything up to and including the assistant header. No `<bos>` for
    Gemma; `render_ids` adds the id at tokenization time."""
    if fam == "qwen":
        s = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
        return s + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if fam == "gemma4":
        s = f"{_SYSTEM_TURN}{system_prompt.strip()}{_END_TURN}" if system_prompt else ""
        return s + f"{_USER_TURN}{user.strip()}{_END_TURN}{_MODEL_TURN}"
    raise ValueError(fam)


def turn_end(fam: str) -> str:
    return "<|im_end|>\n" if fam == "qwen" else _END_TURN


def render_ids(tokenizer, fam: str, prefix: str, completion: str,
               with_turn_end: bool = True) -> tuple[list[int], list[int]]:
    """(prompt_ids, completion_ids), tokenized separately so the label mask
    boundary falls exactly at the assistant header."""
    p = tokenizer.encode(prefix, add_special_tokens=False)
    if fam == "gemma4":
        if tokenizer.bos_token_id is None:
            raise RuntimeError("gemma4 tokenizer has no bos id")
        p = [tokenizer.bos_token_id] + p
    body = completion + (turn_end(fam) if with_turn_end else "")
    c = tokenizer.encode(body, add_special_tokens=False)
    return p, c


# ----------------------------------------------------------- gemma4 renderer

def _text(message: dict) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"gemma4 renderer supports string content only, got "
                           f"{type(content).__name__!r} in role {message.get('role')!r}")
    return content.strip()


def render_gemma4_conversation(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """The Gemma-4 template string after `<bos>`: an optional leading system
    turn, then alternating user/model turns, and in no-think mode a model
    header with a pre-closed thought channel."""
    msgs = list(messages)
    parts: list[str] = []
    if msgs and msgs[0].get("role") in ("system", "developer"):
        parts.append(f"{_SYSTEM_TURN}{_text(msgs[0])}{_END_TURN}")
        msgs = msgs[1:]
    prev_role = None
    for m in msgs:
        role = m.get("role")
        if role == "user":
            parts.append(f"{_USER_TURN}{_text(m)}{_END_TURN}")
        elif role == "assistant":
            if prev_role == "assistant":
                raise RuntimeError("gemma4 renderer: consecutive assistant messages are not "
                                   "supported")
            parts.append(f"{_MODEL_TURN}{_text(m)}{_END_TURN}")
        else:
            raise RuntimeError(f"gemma4 renderer: unsupported role {role!r}")
        prev_role = role
    if add_generation_prompt:
        parts.append(_NOTHINK_SUFFIX)
    return "".join(parts)


def _make_gemma4_renderer_class():
    import tinker
    from tinker_cookbook.renderers import base as rbase

    class Gemma4DisableThinkingRenderer(rbase.Renderer):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            self._bos_id = tokenizer.bos_token_id
            if self._bos_id is None:
                raise RuntimeError("gemma4 renderer: tokenizer has no bos id")
            self._eot_id = self._single_token("<turn|>")
            self._eos_id = self._single_token("<eos>")

        def _single_token(self, s: str) -> int:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"gemma4 renderer: {s!r} is not a single token ({ids})")
            return ids[0]

        @property
        def _bos_tokens(self) -> list[int]:
            return [self._bos_id]

        def get_stop_sequences(self) -> list[int]:
            # <turn|> closes a turn; <eos> is the model's other terminal token
            return [self._eot_id, self._eos_id]

        def build_generation_prompt(self, messages, role="assistant", prefill=None):
            if role != "assistant":
                raise RuntimeError(f"gemma4 renderer: completion role {role!r} not supported")
            s = render_gemma4_conversation(messages, add_generation_prompt=True)
            ids = [self._bos_id] + list(self.tokenizer.encode(s, add_special_tokens=False))
            if prefill:
                ids += list(self.tokenizer.encode(prefill, add_special_tokens=False))
            return tinker.ModelInput(chunks=[tinker.types.EncodedTextChunk(tokens=ids)])

        def render_message(self, message, ctx):
            role = message.get("role")
            if role in ("system", "developer"):
                header = _SYSTEM_TURN
            elif role == "user":
                header = _USER_TURN
            elif role == "assistant":
                header = _MODEL_TURN
            else:
                raise RuntimeError(f"gemma4 renderer: unsupported role {role!r}")
            return rbase.RenderedMessage(
                header=tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer.encode(header, add_special_tokens=False)),
                output=[tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer.encode(f"{_text(message)}{_END_TURN}",
                                                 add_special_tokens=False))])

        def parse_response(self, response):
            response = self._normalize_response_tokens(response)
            return rbase.parse_response_for_stop_token(response, self.tokenizer, self._eot_id)

    return Gemma4DisableThinkingRenderer


def register_renderers() -> None:
    """Register the Gemma-4 no-think renderer with tinker_cookbook (idempotent)."""
    from tinker_cookbook import renderers

    if renderers.is_renderer_registered(GEMMA4_RENDERER):
        return
    cls = _make_gemma4_renderer_class()

    def factory(tokenizer, image_processor=None):
        if image_processor is not None:
            raise RuntimeError("gemma4 renderer is text-only")
        return cls(tokenizer)

    renderers.register_renderer(GEMMA4_RENDERER, factory)
