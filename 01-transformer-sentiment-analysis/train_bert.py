"""
train_bert.py — Fine-tuning BERT for Sentiment Classification

How BERT is adapted for classification
---------------------------------------
BERT (Bidirectional Encoder Representations from Transformers) is an
encoder-only transformer pretrained with two objectives:
  1. Masked Language Modeling (MLM) — predict randomly masked tokens.
  2. Next Sentence Prediction (NSP) — predict whether two sentences are consecutive.

During fine-tuning for classification we use BertForSequenceClassification,
which adds a single linear layer on top of the [CLS] token's hidden state.

  Input:  [CLS] token_1 token_2 ... token_n [SEP]
                  ↓
           Final hidden state of [CLS]   (shape: [batch, hidden_size])
                  ↓
           Linear(hidden_size → num_labels)
                  ↓
           Cross-entropy loss

The [CLS] token is a special token prepended to every input. BERT learns to
encode the overall meaning of the sequence into this single vector, making it
a natural anchor for sentence-level tasks.
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from data_loader import load_imdb_data, SentimentDataset


# ─── Hyperparameters ──────────────────────────────────────────────────────────

MODEL_NAME    = "bert-base-uncased"
MAX_LENGTH    = 512       # BERT supports up to 512 sub-word tokens
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5      # Recommended in the original BERT paper
NUM_EPOCHS    = 3
WARMUP_STEPS  = 0
WEIGHT_DECAY  = 0.01
OUTPUT_DIR    = "outputs/bert"
SEED          = 42

# ─── Reproducibility ──────────────────────────────────────────────────────────

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def tokenize(tokenizer, texts: list, max_length: int = MAX_LENGTH) -> dict:
    """Tokenize a list of raw strings into BERT-compatible input encodings."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors=None,   # Return plain Python lists; Dataset converts to tensors
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
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
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

    # 2. Tokenize
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    print("\nTokenizing datasets…")
    train_enc = tokenize(tokenizer, list(train_data["text"]))
    val_enc   = tokenize(tokenizer, list(val_data["text"]))
    test_enc  = tokenize(tokenizer, list(test_data["text"]))

    # 3. Build PyTorch datasets and data loaders
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

    # 4. Model
    # num_labels=2 tells the model to add a 2-class linear head on top of [CLS].
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # 5. Optimizer — AdamW with weight decay applied only to non-bias parameters
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

    # 6. Learning rate scheduler — linear warmup then linear decay
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    history = []

    # 7. Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            outputs.loss.backward()
            # Gradient clipping prevents exploding gradients
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

        # Save whenever accuracy improves (F1 used as primary, accuracy as fallback)
        score = val_metrics["f1"] if val_metrics["f1"] > 0 else val_metrics["accuracy"]
        if score > best_val_f1:
            best_val_f1 = score
            best_model_path = os.path.abspath(os.path.join(OUTPUT_DIR, "best_model"))
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  → Best model saved  (score={best_val_f1:.4f})")

    # 8. Final test evaluation using the best saved model
    best_model_path = os.path.abspath(os.path.join(OUTPUT_DIR, "best_model"))
    print(f"\nLoading best model for test evaluation…  ({best_model_path})")
    best_model = BertForSequenceClassification.from_pretrained(
        best_model_path, local_files_only=True
    ).to(device)
    test_metrics = evaluate(best_model, test_loader)

    print("\n" + "=" * 50)
    print("BERT — TEST RESULTS")
    print("=" * 50)
    for k, v in test_metrics.items():
        print(f"  {k.capitalize():<12}: {v}")
    print("=" * 50)

    # 9. Persist results to disk
    results = {
        "model"       : MODEL_NAME,
        "config"      : {
            "max_length"   : MAX_LENGTH,
            "batch_size"   : BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs"   : NUM_EPOCHS,
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
