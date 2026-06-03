"""LLM backend wrappers and end-to-end translation pipelines.

Two backends are supported:

* ``VLLMBackend`` — preferred on H100/A100. Uses vLLM's PagedAttention and
  continuous batching to translate the full WMT16 test set in ~1 hour.
* ``TransformersBackend`` — fallback for local Mac / CPU testing. Slow but
  identical interface, useful for the small-subset smoke test in the notebook.

Pipelines:

* :func:`zero_shot_translate`  — single prompt, single decode.
* :func:`maps_translate`       — full Multi-Aspect Prompting and Selection
  pipeline (4 candidates + LLM-as-judge selection).
* :func:`rag_translate`        — dynamic 5-shot using retrieved exemplars.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Sequence

from . import prompts


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


class LLMBackend:
    """Common interface — both backends accept a list of message lists and
    return one string per prompt."""

    def chat(self, batches: List[List[dict]], cfg: GenConfig) -> List[str]:
        raise NotImplementedError


class VLLMBackend(LLMBackend):
    """vLLM-based batched inference. Recommended for H100 deployment."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
    ):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )

    def chat(self, batches: List[List[dict]], cfg: GenConfig) -> List[str]:
        from vllm import SamplingParams

        rendered = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in batches
        ]
        sampling = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        outputs = self.llm.generate(rendered, sampling)
        # vLLM returns outputs in the same order as inputs.
        return [o.outputs[0].text for o in outputs]


class TransformersBackend(LLMBackend):
    """HuggingFace transformers backend. Slow but portable (Mac MPS / CPU).

    Use a smaller model here (e.g. ``Qwen/Qwen2.5-3B-Instruct``) when running
    locally for the smoke test."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    def chat(self, batches: List[List[dict]], cfg: GenConfig) -> List[str]:
        import torch

        outputs = []
        for msgs in batches:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(
                self.model.device
            )
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=cfg.temperature > 0,
                    temperature=max(cfg.temperature, 1e-5),
                    top_p=cfg.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = gen[0, inputs["input_ids"].shape[1]:]
            outputs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outputs


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

@dataclass
class TranslationResult:
    src: str
    hypothesis: str
    intermediate: dict = field(default_factory=dict)


def zero_shot_translate(
    backend: LLMBackend,
    items: Sequence[dict],
    cfg: GenConfig | None = None,
) -> List[TranslationResult]:
    cfg = cfg or GenConfig()
    msgs = [
        prompts.build_zero_shot(it["src"], it["src_lang"], it["tgt_lang"])
        for it in items
    ]
    raws = backend.chat(msgs, cfg)
    return [
        TranslationResult(src=it["src"], hypothesis=prompts.clean_translation(r))
        for it, r in zip(items, raws)
    ]


def maps_translate(
    backend: LLMBackend,
    items: Sequence[dict],
    cfg: GenConfig | None = None,
    judge_cfg: GenConfig | None = None,
) -> List[TranslationResult]:
    """Reproduces the MAPS pipeline (He et al., 2024):

    1. Mine three pieces of knowledge per source: keywords, topic, demo.
    2. Produce four translation candidates: vanilla + 3 knowledge-conditioned.
    3. Let the LLM pick the best candidate (LLM-SCQ selector).
    """
    cfg = cfg or GenConfig(max_new_tokens=256)
    judge_cfg = judge_cfg or GenConfig(max_new_tokens=4)
    n = len(items)

    # --- Stage 1: knowledge mining (3 batched passes) -----------------------
    kw_raw = backend.chat(
        [prompts.build_keyword_prompt(it["src"], it["src_lang"], it["tgt_lang"]) for it in items],
        cfg,
    )
    topic_raw = backend.chat(
        [prompts.build_topic_prompt(it["src"], it["src_lang"], it["tgt_lang"]) for it in items],
        cfg,
    )
    demo_raw = backend.chat(
        [prompts.build_demo_prompt(it["src"], it["src_lang"], it["tgt_lang"]) for it in items],
        cfg,
    )

    keywords = [r.strip().splitlines()[0] if r.strip() else "" for r in kw_raw]
    topics = [r.strip().splitlines()[0] if r.strip() else "" for r in topic_raw]
    demos = [r.strip().splitlines()[0] if r.strip() else "" for r in demo_raw]

    # --- Stage 2: four candidates per item ---------------------------------
    vanilla_msgs = [
        prompts.build_zero_shot(it["src"], it["src_lang"], it["tgt_lang"])
        for it in items
    ]
    kw_msgs = [
        prompts.build_with_knowledge(it["src"], it["src_lang"], it["tgt_lang"],
                                     "Keyword Pairs", keywords[i])
        for i, it in enumerate(items)
    ]
    topic_msgs = [
        prompts.build_with_knowledge(it["src"], it["src_lang"], it["tgt_lang"],
                                     "Topics", topics[i])
        for i, it in enumerate(items)
    ]
    demo_msgs = [
        prompts.build_with_knowledge(it["src"], it["src_lang"], it["tgt_lang"],
                                     "Related sentence pair", demos[i])
        for i, it in enumerate(items)
    ]

    vanilla = backend.chat(vanilla_msgs, cfg)
    cand_kw = backend.chat(kw_msgs, cfg)
    cand_topic = backend.chat(topic_msgs, cfg)
    cand_demo = backend.chat(demo_msgs, cfg)

    candidates_per_item: List[List[str]] = []
    for i in range(n):
        cands = [
            prompts.clean_translation(vanilla[i]),
            prompts.clean_translation(cand_kw[i]),
            prompts.clean_translation(cand_topic[i]),
            prompts.clean_translation(cand_demo[i]),
        ]
        candidates_per_item.append(cands)

    # --- Stage 3: LLM-as-judge selection -----------------------------------
    judge_msgs = [
        prompts.build_judge_prompt(
            items[i]["src"], items[i]["src_lang"], items[i]["tgt_lang"],
            candidates_per_item[i],
        )
        for i in range(n)
    ]
    judgments = backend.chat(judge_msgs, judge_cfg)
    chosen_idx = [
        prompts.parse_judge_letter(j, len(candidates_per_item[i]))
        for i, j in enumerate(judgments)
    ]

    results = []
    for i in range(n):
        results.append(
            TranslationResult(
                src=items[i]["src"],
                hypothesis=candidates_per_item[i][chosen_idx[i]],
                intermediate={
                    "keywords": keywords[i],
                    "topics": topics[i],
                    "demo": demos[i],
                    "candidates": candidates_per_item[i],
                    "judge_raw": judgments[i],
                    "chosen": chosen_idx[i],
                },
            )
        )
    return results


def rag_translate(
    backend: LLMBackend,
    items: Sequence[dict],
    retrieved_per_item: Sequence[Sequence[tuple[str, str]]],
    cfg: GenConfig | None = None,
) -> List[TranslationResult]:
    """One-shot translation conditioned on retrieved 5-shot exemplars."""
    cfg = cfg or GenConfig()
    if len(retrieved_per_item) != len(items):
        raise ValueError("retrieved_per_item must align with items")
    msgs = [
        prompts.build_rag_few_shot(
            it["src"], it["src_lang"], it["tgt_lang"], retrieved_per_item[i]
        )
        for i, it in enumerate(items)
    ]
    raws = backend.chat(msgs, cfg)
    return [
        TranslationResult(
            src=it["src"],
            hypothesis=prompts.clean_translation(r),
            intermediate={"shots": list(retrieved_per_item[i])},
        )
        for i, (it, r) in enumerate(zip(items, raws))
    ]
