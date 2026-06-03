"""
train_gpt.py — Fine-tuning GPT-1 for Sentiment Classification

How GPT-1 is adapted for classification
-----------------------------------------
GPT-1 is a decoder-only transformer pretrained with a single objective:
  Autoregressive Language Modeling — predict the next token given all previous tokens.
  Loss: cross-entropy over the token vocabulary at each position.

Unlike BERT, GPT-1 has no bidirectional attention and no special [CLS] token.
We use OpenAIGPTForSequenceClassification, which attaches a linear head on top of the
hidden state of the LAST NON-PADDING token in the sequence. This token acts as a
summary of the entire sequence in the causal (left-to-right) attention stack.

Input prompt structure
-----------------------
We fine-tune directly on the raw review text. No explicit prompt template is needed
because the classification head is applied to the final token embedding.

  tokens:  tok_1  tok_2  ...  tok_n  [PAD] [PAD]
                                  ↑
                   Classification head is applied here
                   (last token before padding)

Padding token
--------------
GPT-1 was not pretrained with a padding token. We add a new [PAD] special token to the
tokenizer and resize the model's token embedding matrix accordingly. The attention mask
ensures the model never attends to these padding positions.

Training objective
-------------------
During fine-tuning, only the cross-entropy classification loss (not the language
modeling loss) is used. This is consistent with the original GPT paper (Radford et al.,
2018) where a linear classification head replaces the language model head for
supervised tasks.
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    OpenAIGPTTokenizer,
    OpenAIGPTForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from data_loader import load_imdb_data, SentimentDataset


# ─── Hyperparameters ──────────────────────────────────────────────────────────

MODEL_NAME    = "openai-gpt"
MAX_LENGTH    = 512         # GPT-1 context window is 512 tokens
BATCH_SIZE    = 16
LEARNING_RATE = 6.25e-5     # Fine-tuning LR from the original GPT-1 paper
NUM_EPOCHS    = 3
WARMUP_RATIO  = 0.002       # 0.2% of total steps used for linear warmup
WEIGHT_DECAY  = 0.01
OUTPUT_DIR    = "outputs/gpt"
SEED          = 42

# ─── Reproducibility ──────────────────────────────────────────────────────────

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def tokenize(tokenizer, texts: list, max_length: int = MAX_LENGTH) -> dict:
    """
    Tokenize text for GPT-1.

    Because GPT-1 has no intrinsic padding token, the tokenizer must have had
    a [PAD] token added before calling this function.
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )


def compute_metrics(labels: list, preds: list) -> dict:
    """Compute Accuracy, Precision, Recall, and F1 for binary classification."""
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {
        "accuracy" : round(float(accuracy),  4),
        "precision": round(float(precision), 4),
        "recall"   : round(float(recall),    4),
        "f1"       : round(float(f1),        4),
    }


def evaluate(model, dataloader: DataLoader) -> dict:
    """Run the model on a DataLoader and return loss + classification metrics."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = round(total_loss / len(dataloader), 4)
    return metrics


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train():
    # 1. Load data
    train_data, val_data, test_data, _ = load_imdb_data(max_train_samples=5000)

    # 2. Tokenizer — add a padding token since GPT-1 has none by default
    tokenizer = OpenAIGPTTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 3. Tokenize
    print("\nTokenizing datasets…")
    train_enc = tokenize(tokenizer, list(train_data["text"]))
    val_enc   = tokenize(tokenizer, list(val_data["text"]))
    test_enc  = tokenize(tokenizer, list(test_data["text"]))

    # 4. Build PyTorch datasets and data loaders
    train_loader = DataLoader(
        SentimentDataset(train_enc, train_data["label"]),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        SentimentDataset(val_enc, val_data["label"]),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        SentimentDataset(test_enc, test_data["label"]),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # 5. Model
    model = OpenAIGPTForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    # Grow the embedding matrix to include the new [PAD] token
    model.resize_token_embeddings(len(tokenizer))
    # Tell the model which token ID is the padding token so it can
    # identify the last *real* token for classification
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # 6. Optimizer — AdamW with weight decay on non-bias, non-LayerNorm parameters
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer = AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": WEIGHT_DECAY},
            {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
    )

    # 7. Learning rate scheduler
    total_steps  = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    history = []

    # 8. Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += outputs.loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        val_metrics    = evaluate(model, val_loader)

        epoch_record = {
            "epoch"      : epoch,
            "train_loss" : round(avg_train_loss, 4),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_record)

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS} — "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        score = val_metrics["f1"] if val_metrics["f1"] > 0 else val_metrics["accuracy"]
        if score > best_val_f1:
            best_val_f1 = score
            best_model_path = os.path.abspath(os.path.join(OUTPUT_DIR, "best_model"))
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  → Best model saved  (score={best_val_f1:.4f})")

    # 9. Final test evaluation using the best saved model
    best_model_path = os.path.abspath(os.path.join(OUTPUT_DIR, "best_model"))
    print(f"\nLoading best model for test evaluation…  ({best_model_path})")
    best_tokenizer = OpenAIGPTTokenizer.from_pretrained(
        best_model_path, local_files_only=True
    )
    best_model = OpenAIGPTForSequenceClassification.from_pretrained(
        best_model_path, local_files_only=True
    ).to(device)
    test_metrics = evaluate(best_model, test_loader)

    print("\n" + "=" * 50)
    print("GPT-1 — TEST RESULTS")
    print("=" * 50)
    for k, v in test_metrics.items():
        print(f"  {k.capitalize():<12}: {v}")
    print("=" * 50)

    # 10. Persist results
    results = {
        "model"       : MODEL_NAME,
        "config"      : {
            "max_length"   : MAX_LENGTH,
            "batch_size"   : BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs"   : NUM_EPOCHS,
            "warmup_ratio" : WARMUP_RATIO,
            "weight_decay" : WEIGHT_DECAY,
        },
        "history"     : history,
        "test_metrics": test_metrics,
    }
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    train()
