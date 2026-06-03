"""Prompt formatting for EN<->TR translation using Qwen chat template."""

from typing import Literal

Direction = Literal["en2tr", "tr2en"]

EN2TR_INSTR = "Translate the following English sentence to Turkish."
TR2EN_INSTR = "Translate the following Turkish sentence to English."


def build_messages(direction: Direction, src_text: str, tgt_text: str | None = None):
    """Return a list of chat messages in Qwen's expected format.

    If tgt_text is None we're building an inference prompt (no assistant turn).
    """
    instruction = EN2TR_INSTR if direction == "en2tr" else TR2EN_INSTR
    messages = [
        {"role": "user", "content": f"{instruction}\n\n{src_text}"},
    ]
    if tgt_text is not None:
        messages.append({"role": "assistant", "content": tgt_text})
    return messages


def format_for_training(tokenizer, example: dict, direction: Direction) -> dict:
    """Convert a WMT16 sample into the supervised training format Qwen expects.

    example: {"translation": {"en": "...", "tr": "..."}}
    """
    en = example["translation"]["en"]
    tr = example["translation"]["tr"]
    if direction == "en2tr":
        src, tgt = en, tr
    else:
        src, tgt = tr, en
    messages = build_messages(direction, src, tgt)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def format_for_inference(tokenizer, src_text: str, direction: Direction) -> str:
    """Build a prompt suitable for generation (assistant turn left open)."""
    messages = build_messages(direction, src_text, tgt_text=None)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
