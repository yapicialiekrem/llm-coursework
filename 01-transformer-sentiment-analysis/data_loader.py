"""
data_loader.py — IMDb Dataset Loading and Preprocessing

The IMDb Movie Reviews dataset contains 50,000 reviews labeled as
positive (1) or negative (0). The official split provides 25,000
training and 25,000 test samples. We carve out a validation set
from the training split.

Dataset source: https://huggingface.co/datasets/imdb
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


# ─── Dataset Class ────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """
    A generic PyTorch Dataset wrapper around HuggingFace tokenizer encodings.
    Compatible with both BERT and GPT-1 tokenizers.
    """

    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_imdb_data(val_split: float = 0.2, seed: int = 42, max_train_samples: int = None):
    """
    Download the IMDb dataset and create train / validation / test splits.

    Args:
        val_split: Fraction of the training set to use for validation.
        seed:      Random seed for reproducibility.

    Returns:
        train_data:  HuggingFace Dataset (training split)
        val_data:    HuggingFace Dataset (validation split)
        test_data:   HuggingFace Dataset (test split)
        label_map:   Dict mapping integer labels to human-readable strings.
    """
    print("Downloading IMDb dataset...")
    dataset = load_dataset("imdb")

    # Shuffle first to avoid label ordering bias, then optionally subsample
    source = dataset["train"].shuffle(seed=seed)
    if max_train_samples is not None:
        source = source.select(range(max_train_samples))

    # Stratified split — preserves 50/50 class balance in both train and val
    train_val = source.train_test_split(
        test_size=val_split, seed=seed, stratify_by_column="label"
    )
    train_data = train_val["train"]
    val_data   = train_val["test"]
    test_data  = dataset["test"]

    label_map = {0: "negative", 1: "positive"}

    print(f"  Train      : {len(train_data):,} samples")
    print(f"  Validation : {len(val_data):,} samples")
    print(f"  Test       : {len(test_data):,} samples")
    print(f"  Labels     : {label_map}")

    # Print a couple of example records
    print("\nExample records:")
    for i in range(2):
        sample = train_data[i]
        preview = sample["text"][:120].replace("\n", " ")
        print(f"  [{label_map[sample['label']].upper()}] {preview}...")

    return train_data, val_data, test_data, label_map


if __name__ == "__main__":
    train_data, val_data, test_data, label_map = load_imdb_data()
    print("\nDataset loading complete.")
