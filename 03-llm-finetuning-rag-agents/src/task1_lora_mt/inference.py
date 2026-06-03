"""Batched generation utility for the fine-tuned (or base) translation model."""

from typing import Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .prompt_format import format_for_inference


def load_model(
    base_model_path: str,
    adapter_path: str | None,
    load_in_4bit: bool = True,
    device_map: str = "auto",
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only LMs require LEFT padding during generation — otherwise the
    # model attends to PAD tokens on the right and produces garbage outputs.
    tokenizer.padding_side = "left"

    kwargs = {"trust_remote_code": True, "device_map": device_map}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def translate_batch(
    model,
    tokenizer,
    sources: List[str],
    direction: str,
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> List[str]:
    """Translate sources in batches and return only the assistant-generated text."""
    outputs: List[str] = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i : i + batch_size]
        prompts = [format_for_inference(tokenizer, s, direction) for s in batch_src]
        encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        gen = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        # With left-padding, generated tokens start at index input_len for all
        # samples in the batch (model.generate appends after the input).
        input_len = encodings["input_ids"].shape[1]
        for j in range(gen.shape[0]):
            new_tokens = gen[j, input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outputs.append(text)
    return outputs
