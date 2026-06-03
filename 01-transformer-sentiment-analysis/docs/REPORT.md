# Technical Report — BERT vs GPT-1 for Sentiment Classification

## 1. Objective

Fine-tune two pre-trained transformers of different architectural families on the
**same** binary sentiment task and compare them under matched conditions:

- **BERT** (`bert-base-uncased`) — a bidirectional **encoder**.
- **GPT-1** (`openai-gpt`) — an early autoregressive **decoder**.

The interesting question is not "which is better at sentiment" (the answer is known) but
*why*, and how each architecture's pre-training objective shows up in classification.

## 2. Data

- **Dataset**: IMDb movie reviews (HuggingFace `imdb`) — 50,000 reviews, perfectly
  balanced positive/negative, official 25k/25k train/test split.
- **Validation**: a stratified 20% slice carved out of the training split (`seed=42`),
  so class balance is preserved in train and val alike.
- **Test**: the untouched official 25k test set is used only for the final numbers.

## 3. Models and classification heads

| | BERT | GPT-1 |
|--|------|-------|
| Class | `BertForSequenceClassification` | `OpenAIGPTForSequenceClassification` |
| Pooling for classification | `[CLS]` token representation | last non-pad token |
| Pad token | native | **none** — added (`[PAD]`) and embeddings resized |
| Max sequence length | 512 | 256 |

GPT-1 has no padding token out of the box (it was trained for plain left-to-right
generation), so one is added to the tokenizer, the embedding matrix is resized
(`resize_token_embeddings`), and `model.config.pad_token_id` is set so the classifier
reads the correct last real token rather than a pad position.

## 4. Training setup

| Hyperparameter | BERT | GPT-1 |
|----------------|:----:|:-----:|
| Learning rate | 2e-5 | 6.25e-5 |
| Batch size | 16 | 8 |
| Epochs | 3 | 3 |
| Weight decay | 0.01 | 0.01 |
| Max length | 512 | 256 |

The learning rates follow each model's conventional fine-tuning range (BERT ~2e-5,
GPT-1's original ~6.25e-5). GPT-1 uses a shorter max length and smaller batch mostly to
fit memory on a single T4.

## 5. Results

### Final test metrics

| Model | Accuracy | Precision | Recall | F1 | Test loss |
|-------|:--------:|:---------:|:------:|:--:|:---------:|
| **BERT** | **0.9237** | 0.9304 | 0.9160 | **0.9231** | 0.2645 |
| GPT-1 | 0.8978 | 0.8865 | 0.9125 | 0.8993 | 0.5923 |

### Validation curves (per epoch)

**BERT** — best val F1 at epoch 2 (0.9045), then mild overfitting (val loss rises 0.31 → 0.43):

| Epoch | Train loss | Val acc | Val F1 | Val loss |
|:-----:|:----------:|:-------:|:------:|:--------:|
| 1 | 0.342 | 0.880 | 0.888 | 0.306 |
| 2 | 0.159 | 0.906 | 0.905 | 0.332 |
| 3 | 0.082 | 0.899 | 0.900 | 0.430 |

**GPT-1** — converges to ~0.89 val F1; val loss climbs sharply (0.34 → 0.72), a clearer
overfitting signal than BERT:

| Epoch | Train loss | Val acc | Val F1 | Val loss |
|:-----:|:----------:|:-------:|:------:|:--------:|
| 1 | 0.442 | 0.873 | 0.869 | 0.338 |
| 2 | 0.200 | 0.873 | 0.879 | 0.547 |
| 3 | 0.043 | 0.885 | 0.887 | 0.716 |

## 6. Discussion

- **Why BERT wins (~2.4 F1).** Bidirectional self-attention lets every token attend to
  the full review in both directions; the `[CLS]` representation aggregates that global
  context, which is exactly what sentiment classification needs. GPT-1's causal mask
  means the final-token representation has only seen the review left-to-right, so it's a
  weaker summary for classification.
- **Overfitting.** Both models overfit by epoch 3 (train loss collapses while val loss
  rises), but GPT-1's test loss (0.59) is more than double BERT's (0.26) at similar
  accuracy — its probability estimates are less calibrated. Early stopping on val F1
  (epoch 2 for BERT) would be the right production choice.
- **Fair framing.** GPT-1 is a 2018 model being used off-label for classification; ~0.90
  F1 is a respectable result and the gap to BERT is exactly what the architectures
  predict. A modern decoder (or simply more capacity) would narrow it.

## 7. Limitations

- IMDb is balanced and stylistically uniform; these numbers would drop on noisier,
  domain-shifted, or code-switched text.
- No hyperparameter search — single configurations per model. The comparison is
  qualitative-architectural, not a tuned benchmark.
