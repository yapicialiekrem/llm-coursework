"""WMT16 EN-TR loading + preprocessing for LoRA training."""

from datasets import load_dataset, concatenate_datasets, Dataset
from .prompt_format import format_for_training


def load_wmt16_tr_en(train_samples: int, eval_samples: int, test_samples: int, seed: int = 42):
    """Load WMT16 tr-en splits and subsample."""
    ds = load_dataset("wmt16", "tr-en")
    train = ds["train"].shuffle(seed=seed).select(range(min(train_samples, len(ds["train"]))))
    val = ds["validation"].shuffle(seed=seed).select(range(min(eval_samples, len(ds["validation"]))))
    test = ds["test"].shuffle(seed=seed).select(range(min(test_samples, len(ds["test"]))))
    return train, val, test


def build_training_dataset(raw_ds: Dataset, tokenizer, direction: str = "both") -> Dataset:
    """Apply the chat template. If direction='both', duplicate samples in both directions."""
    if direction == "both":
        en2tr = raw_ds.map(
            lambda ex: format_for_training(tokenizer, ex, "en2tr"),
            remove_columns=raw_ds.column_names,
            desc="formatting en->tr",
        )
        tr2en = raw_ds.map(
            lambda ex: format_for_training(tokenizer, ex, "tr2en"),
            remove_columns=raw_ds.column_names,
            desc="formatting tr->en",
        )
        return concatenate_datasets([en2tr, tr2en]).shuffle(seed=42)
    else:
        return raw_ds.map(
            lambda ex: format_for_training(tokenizer, ex, direction),
            remove_columns=raw_ds.column_names,
            desc=f"formatting {direction}",
        )
