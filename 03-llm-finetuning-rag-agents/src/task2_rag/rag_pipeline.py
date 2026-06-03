"""RAG generation pipeline for Turkish History multiple-choice QA.

Builds either a zero-shot prompt or a RAG-augmented one, then asks the LLM
to output a single letter (A/B/C/D/E) which we parse out.
"""

import re
from typing import List

import torch

from src.task1_lora_mt.inference import load_model


SYSTEM_TR = (
    "Sen Türkçe tarih sorularını cevaplayan bir asistansın. "
    "Sana bir çoktan seçmeli soru ve seçenekler verilecek. "
    "Sadece doğru şıkkın harfini (A, B, C, D veya E) cevap olarak ver. "
    "Başka hiçbir açıklama yapma."
)

RAG_TEMPLATE = """Aşağıdaki bağlam bilgilerini kullanarak soruyu cevapla.

### Bağlam:
{context}

### Soru:
{question}

### Seçenekler:
{choices}

Cevap (yalnızca harf):"""

ZS_TEMPLATE = """Aşağıdaki çoktan seçmeli tarih sorusunu cevapla.

### Soru:
{question}

### Seçenekler:
{choices}

Cevap (yalnızca harf):"""


def format_choices(choices: List[str]) -> str:
    letters = "ABCDE"
    return "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))


def build_zero_shot_prompt(tokenizer, question: str, choices: List[str]) -> str:
    user_msg = ZS_TEMPLATE.format(question=question, choices=format_choices(choices))
    messages = [
        {"role": "system", "content": SYSTEM_TR},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_rag_prompt(tokenizer, question: str, choices: List[str], context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    user_msg = RAG_TEMPLATE.format(context=context, question=question, choices=format_choices(choices))
    messages = [
        {"role": "system", "content": SYSTEM_TR},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


ANSWER_RE = re.compile(r"\b([ABCDE])\b")


def parse_answer(text: str) -> str | None:
    """Pull the first A-E letter out of the model's output, else None."""
    text = text.strip().upper()
    m = ANSWER_RE.search(text)
    return m.group(1) if m else None


@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True)
