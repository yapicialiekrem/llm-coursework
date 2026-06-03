"""LoRA / QLoRA training script for EN<->TR machine translation.

Usage:
    python -m src.task1_lora_mt.train --config configs/lora_config.yaml
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from .data_loader import build_training_dataset, load_wmt16_tr_en


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_bnb_config(model_cfg: dict) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=model_cfg["load_in_4bit"],
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=model_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, model_cfg["bnb_4bit_compute_dtype"]),
    )


def build_lora_config(lora_cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[load] base model in 4-bit")
    bnb_config = build_bnb_config(model_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_cfg["gradient_checkpointing"])

    print("[lora] wrapping with PEFT adapters")
    model = get_peft_model(model, build_lora_config(lora_cfg))
    model.print_trainable_parameters()

    print("[data] loading WMT16 tr-en")
    train_raw, val_raw, _ = load_wmt16_tr_en(
        train_samples=data_cfg["train_samples"],
        eval_samples=data_cfg["eval_samples"],
        test_samples=data_cfg["test_samples"],
        seed=train_cfg["seed"],
    )
    train_ds = build_training_dataset(train_raw, tokenizer, direction=data_cfg["direction"])
    val_ds = build_training_dataset(val_raw, tokenizer, direction=data_cfg["direction"])
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    Path(train_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        optim=train_cfg["optim"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        report_to=train_cfg["report_to"],
        seed=train_cfg["seed"],
        max_seq_length=data_cfg["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,   # TRL 0.12+ renamed `tokenizer` to `processing_class`
    )

    print("[train] starting")
    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print(f"[done] adapter saved to {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()
