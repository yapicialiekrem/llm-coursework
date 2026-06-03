"""Prompt templates used across the three translation pipelines:

1. Zero-shot baseline (`build_zero_shot`).
2. MAPS — Multi-Aspect Prompting and Selection (He et al., TACL 2024). Three
   knowledge-mining prompts (keywords, topics, demonstrations), four candidate
   prompts (vanilla + 3 knowledge-conditioned), and one LLM-as-judge selector.
3. RAG-based dynamic 5-shot (`build_rag_few_shot`).

We use Qwen 2.5 Instruct's native chat template, so prompts are returned as a
list of {"role", "content"} messages and the tokenizer is responsible for
inserting <|im_start|> / <|im_end|> tags.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}

LANG_NAME = {"en": "English", "tr": "Turkish"}

SYSTEM_TRANSLATOR = (
    "You are a professional, faithful machine translation engine. "
    "Output ONLY the translation text, with no quotes, no explanations, "
    "no language labels, and no surrounding commentary."
)

SYSTEM_ANALYST = (
    "You are an expert linguistic assistant helping prepare high-quality "
    "machine translations. Follow the requested output format exactly."
)

SYSTEM_JUDGE = (
    "You are an expert bilingual translation evaluator. "
    "Choose the single best translation among the candidates. "
    "Reply with ONLY the letter (A, B, C, or D)."
)


# ---------------------------------------------------------------------------
# 1. Zero-shot baseline
# ---------------------------------------------------------------------------

def build_zero_shot(src: str, src_lang: str, tgt_lang: str) -> List[Message]:
    """Plain zero-shot translation prompt — used as the baseline for Part 3."""
    user = (
        f"Translate the following {LANG_NAME[src_lang]} text into "
        f"{LANG_NAME[tgt_lang]}.\n\n"
        f"{LANG_NAME[src_lang]}: {src}\n"
        f"{LANG_NAME[tgt_lang]}:"
    )
    return [
        {"role": "system", "content": SYSTEM_TRANSLATOR},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# 2. MAPS — Multi-Aspect Prompting and Selection
# ---------------------------------------------------------------------------
# The 5 trigger sentences below mirror the domain coverage used by the
# original MAPS code (health, aviation, sports, politics, business). They are
# kept identical across the three knowledge prompts so the model sees a
# consistent demonstration distribution.

_TRIGGER_EN = [
    "The doctors advised that patients recovering from heart surgery should follow a strict diet and gradually increase physical activity.",
    "The pilot reported a sudden loss of altitude before the aircraft disappeared from radar over the Andes.",
    "Real Madrid clinched the championship after a dramatic second-half comeback against their long-time rivals.",
    "The European Parliament has voted to extend sanctions following the failure of the latest round of diplomatic talks.",
    "The startup announced a $50 million Series B funding round led by a Tokyo-based venture capital firm.",
]

_TRIGGER_TR = [
    "Doktorlar, kalp ameliyatı sonrası iyileşmekte olan hastaların sıkı bir diyet uygulaması ve fiziksel aktiviteyi kademeli olarak artırması gerektiğini önerdi.",
    "Pilot, uçağın And Dağları üzerinde radardan kaybolmasından önce ani bir irtifa kaybı bildirdi.",
    "Real Madrid, uzun süredir rakibi olan takıma karşı ikinci yarıda yaptığı dramatik geri dönüşün ardından şampiyonluğu garantiledi.",
    "Avrupa Parlamentosu, son tur diplomatik görüşmelerin başarısızlıkla sonuçlanmasının ardından yaptırımları uzatmak için oy kullandı.",
    "Girişim, Tokyo merkezli bir risk sermayesi firmasının liderlik ettiği 50 milyon dolarlık B serisi yatırım turunu duyurdu.",
]


def _trigger_pairs(src_lang: str, tgt_lang: str) -> list[tuple[str, str]]:
    if src_lang == "en" and tgt_lang == "tr":
        return list(zip(_TRIGGER_EN, _TRIGGER_TR))
    if src_lang == "tr" and tgt_lang == "en":
        return list(zip(_TRIGGER_TR, _TRIGGER_EN))
    raise ValueError(f"Unsupported pair: {src_lang}->{tgt_lang}")


# --- Stage 1: knowledge mining ---------------------------------------------

def build_keyword_prompt(src: str, src_lang: str, tgt_lang: str) -> List[Message]:
    """Ask the model to extract bilingual keyword pairs."""
    pairs = _trigger_pairs(src_lang, tgt_lang)
    examples = []
    # Stub keyword pairs for the 5 triggers. These mirror the style of the
    # MAPS paper's hand-written exemplars.
    stub_kw_en_tr = [
        "heart surgery=kalp ameliyatı, strict diet=sıkı diyet, physical activity=fiziksel aktivite",
        "pilot=pilot, loss of altitude=irtifa kaybı, Andes=And Dağları",
        "Real Madrid=Real Madrid, championship=şampiyonluk, comeback=geri dönüş",
        "European Parliament=Avrupa Parlamentosu, sanctions=yaptırımlar, diplomatic talks=diplomatik görüşmeler",
        "startup=girişim, Series B=B serisi, venture capital firm=risk sermayesi firması",
    ]
    stub_kw_tr_en = [
        "kalp ameliyatı=heart surgery, sıkı diyet=strict diet, fiziksel aktivite=physical activity",
        "pilot=pilot, irtifa kaybı=loss of altitude, And Dağları=Andes",
        "Real Madrid=Real Madrid, şampiyonluk=championship, geri dönüş=comeback",
        "Avrupa Parlamentosu=European Parliament, yaptırımlar=sanctions, diplomatik görüşmeler=diplomatic talks",
        "girişim=startup, B serisi=Series B, risk sermayesi firması=venture capital firm",
    ]
    kw = stub_kw_en_tr if src_lang == "en" else stub_kw_tr_en

    for (s, _), k in zip(pairs, kw):
        examples.append(
            f"{LANG_NAME[src_lang]}: {s}\nKeyword Pairs: {k}"
        )

    instruction = (
        f"Extract the keywords in the following {LANG_NAME[src_lang]} sentence, "
        f"and then translate these keywords into {LANG_NAME[tgt_lang]}. "
        "Output ONLY a single line in the form 'src=tgt, src=tgt, ...'."
    )
    body = "\n\n".join(examples)
    user = (
        f"{instruction}\n\n{body}\n\n"
        f"{LANG_NAME[src_lang]}: {src}\nKeyword Pairs:"
    )
    return [
        {"role": "system", "content": SYSTEM_ANALYST},
        {"role": "user", "content": user},
    ]


def build_topic_prompt(src: str, src_lang: str, tgt_lang: str) -> List[Message]:
    """Ask the model to describe the topics of the sentence."""
    pairs = _trigger_pairs(src_lang, tgt_lang)
    stub_topics = [
        "Health, post-surgery recovery",
        "Aviation accident, missing aircraft",
        "Sports, football championship",
        "Politics, EU sanctions",
        "Business, venture capital",
    ]
    examples = [
        f"Input: {s}\nTopics: {t}" for (s, _), t in zip(pairs, stub_topics)
    ]
    instruction = (
        "Use a few words to describe the topics of the following input sentence. "
        "Output ONLY a single short line of topics."
    )
    body = "\n\n".join(examples)
    user = f"{instruction}\n\n{body}\n\nInput: {src}\nTopics:"
    return [
        {"role": "system", "content": SYSTEM_ANALYST},
        {"role": "user", "content": user},
    ]


def build_demo_prompt(src: str, src_lang: str, tgt_lang: str) -> List[Message]:
    """Ask the model to write a related sentence and translate it.

    The MAPS paper calls this 'demonstration generation' — a self-generated
    analogous parallel pair (not a retrieved one)."""
    pairs = _trigger_pairs(src_lang, tgt_lang)
    # We re-use the trigger pairs themselves as the "related" demos in the
    # few-shot context (their second sentence in the original MAPS data).
    examples = []
    for (s, t) in pairs:
        examples.append(
            f"Input {LANG_NAME[src_lang]} sentence: {s}\n"
            f"Output {LANG_NAME[src_lang]}-{LANG_NAME[tgt_lang]} sentence pair: {s} ||| {t}"
        )
    instruction = (
        f"Write a {LANG_NAME[src_lang]} sentence that is related to but "
        f"different from the input {LANG_NAME[src_lang]} sentence and translate "
        f"it into {LANG_NAME[tgt_lang]}. "
        f"Output ONLY one line in the form '<{LANG_NAME[src_lang]} sentence> ||| <{LANG_NAME[tgt_lang]} sentence>'."
    )
    body = "\n\n".join(examples)
    user = (
        f"{instruction}\n\n{body}\n\n"
        f"Input {LANG_NAME[src_lang]} sentence: {src}\n"
        f"Output {LANG_NAME[src_lang]}-{LANG_NAME[tgt_lang]} sentence pair:"
    )
    return [
        {"role": "system", "content": SYSTEM_ANALYST},
        {"role": "user", "content": user},
    ]


# --- Stage 2: knowledge-conditioned translation -----------------------------

def build_with_knowledge(
    src: str,
    src_lang: str,
    tgt_lang: str,
    knowledge_type: str,
    knowledge: str,
) -> List[Message]:
    """Translate using a single piece of mined knowledge as context."""
    user = (
        f"{knowledge_type}: {knowledge}\n\n"
        f"Given the above knowledge, translate the following "
        f"{LANG_NAME[src_lang]} text into {LANG_NAME[tgt_lang]}.\n\n"
        f"{LANG_NAME[src_lang]}: {src}\n"
        f"{LANG_NAME[tgt_lang]}:"
    )
    return [
        {"role": "system", "content": SYSTEM_TRANSLATOR},
        {"role": "user", "content": user},
    ]


# --- Stage 3: LLM-as-a-judge selection --------------------------------------

def build_judge_prompt(
    src: str,
    src_lang: str,
    tgt_lang: str,
    candidates: Sequence[str],
) -> List[Message]:
    """LLM-SCQ selector from the MAPS paper.

    The model picks the best of (typically four) candidates and replies with a
    single letter."""
    letters = ["A", "B", "C", "D", "E", "F"]
    listed = "\n".join(
        f"{letters[i]}. {c.strip()}" for i, c in enumerate(candidates)
    )
    user = (
        f"Source ({LANG_NAME[src_lang]}): {src}\n\n"
        f"Below are candidate {LANG_NAME[tgt_lang]} translations. "
        "Choose the one that is most accurate, fluent, and faithful to the "
        "source. Reply with ONLY the letter (A/B/C/...).\n\n"
        f"{listed}\n\nAnswer:"
    )
    return [
        {"role": "system", "content": SYSTEM_JUDGE},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# 3. RAG dynamic 5-shot
# ---------------------------------------------------------------------------

def build_rag_few_shot(
    src: str,
    src_lang: str,
    tgt_lang: str,
    retrieved: Sequence[tuple[str, str]],
) -> List[Message]:
    """5-shot prompt where exemplars are retrieved by similarity to `src`."""
    examples = "\n\n".join(
        f"{LANG_NAME[src_lang]}: {ex_src}\n{LANG_NAME[tgt_lang]}: {ex_tgt}"
        for ex_src, ex_tgt in retrieved
    )
    user = (
        f"Translate from {LANG_NAME[src_lang]} into {LANG_NAME[tgt_lang]}. "
        "Use the style and terminology of the examples below.\n\n"
        f"{examples}\n\n"
        f"{LANG_NAME[src_lang]}: {src}\n"
        f"{LANG_NAME[tgt_lang]}:"
    )
    return [
        {"role": "system", "content": SYSTEM_TRANSLATOR},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_translation(raw: str) -> str:
    """Strip the assistant's response down to the translation only."""
    if raw is None:
        return ""
    text = raw.strip()
    # Drop leading language label if present (e.g., "Turkish: ...").
    for label in ("Turkish:", "English:", "Translation:", "Çeviri:"):
        if text.lower().startswith(label.lower()):
            text = text[len(label):].strip()
    # Keep only the first non-empty line — many small models append rationales.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.strip(' "“”')
    return text


def parse_judge_letter(raw: str, n_candidates: int) -> int:
    """Parse the judge response into a 0-indexed candidate index. Falls back to
    candidate 0 (vanilla) on parse failure."""
    if not raw:
        return 0
    s = raw.strip().upper()
    for ch in s:
        if "A" <= ch <= "F":
            idx = ord(ch) - ord("A")
            if 0 <= idx < n_candidates:
                return idx
    return 0
