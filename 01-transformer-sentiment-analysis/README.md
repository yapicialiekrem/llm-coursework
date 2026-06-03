# Transformer Sentiment Analysis — BERT vs GPT-1

Fine-tune and compare an **encoder** (BERT) and an early **decoder** (GPT-1) on the
same binary sentiment task, to see how the two architectures behave on classification.

## Problem

Classify IMDb movie reviews as **positive / negative** (50,000 reviews, 25k train /
25k test). Same data, same metrics, two very different pre-trained backbones —
`bert-base-uncased` (encoder, bidirectional) and `openai-gpt` (decoder, left-to-right).

## Approach

- **Data** — IMDb via HuggingFace `datasets`, stratified 80/20 train/val split off the
  training set, untouched test set for the final numbers (`data_loader.py`).
- **BERT** — `[CLS]`-pooled sequence classification head, max length 512 (`train_bert.py`).
- **GPT-1** — sequence classification on the last token; GPT-1 has no pad token, so one
  is added and embeddings resized; max length 256 (`train_gpt.py`).
- **Evaluation** — accuracy, precision, recall, F1 on the held-out test set, plus a
  side-by-side comparison (`compare_models.py`).

## Results (IMDb test set)

| Model | Accuracy | Precision | Recall | F1 |
|-------|:--------:|:---------:|:------:|:--:|
| **BERT** (`bert-base-uncased`) | **0.9237** | 0.9304 | 0.9160 | **0.9231** |
| GPT-1 (`openai-gpt`) | 0.8978 | 0.8865 | 0.9125 | 0.8993 |

**Takeaway.** BERT wins by ~2.4 F1 points. Its bidirectional attention sees the whole
review at once, which suits classification; GPT-1's left-to-right objective was built
for generation, so squeezing a classification head onto it leaves some accuracy on the
table — but it still lands near 0.90, which is a fair showing for a 2018-era model.
Full per-epoch training curves are in [`bert_results.json`](bert_results.json) and
[`gpt_results.json`](gpt_results.json).

## Training configuration

| | BERT | GPT-1 |
|--|------|-------|
| Max length | 512 | 256 |
| Batch size | 16 | 8 |
| Learning rate | 2e-5 | 6.25e-5 |
| Epochs | 3 | 3 |
| Weight decay | 0.01 | 0.01 |

(See [`bert_config.json`](bert_config.json) / [`gpt_config.json`](gpt_config.json).)

## Run it

The full flow is in the notebook ([`sentiment_analysis.ipynb`](sentiment_analysis.ipynb),
~50–60 min on a Colab T4 GPU). To run the scripts directly:

```bash
pip install -r requirements.txt
python train_bert.py        # writes outputs/bert/results.json
python train_gpt.py         # writes outputs/gpt/results.json
python compare_models.py    # prints the side-by-side table
```

## Files

```
sentiment_analysis.ipynb   # end-to-end notebook (Colab T4)
data_loader.py             # IMDb loading + stratified split
train_bert.py              # BERT fine-tuning
train_gpt.py               # GPT-1 fine-tuning (pad-token handling)
compare_models.py          # reads both results.json, prints comparison
bert_results.json /        # measured metrics + per-epoch history
gpt_results.json
docs/REPORT.md             # fuller write-up
```

## Notes & limitations

- IMDb is balanced and relatively "easy" sentiment — these numbers wouldn't transfer
  unchanged to noisier domains (sarcasm, mixed-language reviews).
- GPT-1 is intentionally a *dated* baseline; the point is the architecture comparison,
  not a state-of-the-art result. A modern decoder would close most of the gap.
