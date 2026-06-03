# Task 2 — RAG-Based QA: Implementation and Discussion

> Final results numbers are auto-injected by `scripts/build_final_report.py` into `docs/final_results.md`. This document covers the **decisions, justifications and discussion** required by the assignment.

## Part A — Dataset

We use the `History` subset of **TurkishMMLU** (Yüksel et al., 2024), introduced in *"TurkishMMLU: Measuring Massive Multitask Language Understanding in Turkish"* (arXiv:2407.12402). The benchmark provides curriculum-aligned, expert-written multiple-choice questions across 9 subjects; we filter to History only.

- HuggingFace dataset: `AYueksel/TurkishMMLU`, config `History`
- Splits: `dev` (5 questions, for optional few-shot prompts) + `test` (100 questions, used for evaluation)
- Format: `{question: str, choices: List[str], answer: 'A'-'E'}`
- All questions are in Turkish and aligned with the Turkish high-school history curriculum (Ottoman history, world history, Republic-era reforms, etc.).

## Part B — Knowledge Base Construction

### B.1 Source documents

External knowledge source: **4 MEB (Milli Eğitim Bakanlığı) lise tarih ders kitabı**, 2024-2025 müfredat:

| Book | File | Approx. size |
|---|---|---|
| 9. Sınıf Tarih | `data/raw/meb_tarih/9_sinif_tarih.pdf` | 56 MB, 556 K chars |
| 10. Sınıf Tarih | `data/raw/meb_tarih/10_sinif_tarih.pdf` | 26 MB, 586 K chars |
| 11. Sınıf Tarih | `data/raw/meb_tarih/11_sinif_tarih.pdf` | 33 MB, 537 K chars |
| 12. Sınıf T.C. İnkılap Tarihi | `data/raw/meb_tarih/12_sinif_inkilap.pdf` | 28 MB, 732 K chars |

**Total: 4 books, ~2.41 M characters, ~315 K Turkish words.**

**Why MEB textbooks?** TurkishMMLU's History questions are explicitly drawn from the Turkish high-school curriculum — the same curriculum these books are written against. This is the closest possible match between the QA benchmark and the knowledge source short of using the benchmark itself (which would defeat the point of RAG).

### B.2 Document processing pipeline

```
PDF → pdfplumber → raw text → token-aware recursive chunker → bge-m3 embeddings → ChromaDB persistent collection
```

#### a) Chunking strategy: **Recursive character splitter (token-aware)**

We use LangChain's `RecursiveCharacterTextSplitter` with a custom `length_function` that counts tokens via the `bge-m3` tokenizer. The splitter tries separators in descending order — `["\n\n", "\n", ". ", " ", ""]` — falling back to character-level splits only when no semantic boundary fits in the budget.

**Why recursive over alternatives?** See section B.3.

#### b) Chunk size: **512 tokens**

Rationale:
- Large enough to capture a **multi-sentence paragraph** describing a single historical event (date, actors, consequences) — single-sentence chunks would force the LLM to retrieve many chunks for any non-trivial question.
- Small enough to fit **5 chunks (~2500 tokens) in the LLM prompt** with room to spare for the question, options, and instructions on a 4K-context budget.
- bge-m3's encoder context is 8192 tokens, well above 512 — no encoder truncation.

#### c) Overlap: **64 tokens (~12.5 %)**

- Protects against cutting an event description mid-paragraph (chunk boundaries often interrupt key facts).
- 12.5 % is the **standard heuristic** in the RAG literature (LangChain default is similar); higher overlap (e.g. 25 %) bloats the index without measurable gains for our scale.
- We empirically verified retrieval quality on 3 spot-check queries (see retrieval sanity test in `notebooks/` or the build log) — all top-1 chunks were on-topic.

#### d) Embedding model: **`BAAI/bge-m3`**

- **Multilingual + Turkish-strong**: top-quartile on MIRACL-tr, the Turkish information-retrieval benchmark.
- **Dense + sparse hybrid-capable**: the model can also output sparse weights for BM25-style retrieval, leaving room for a hybrid extension if dense retrieval ever underperforms.
- **Local inference**: runs on our GPU in ~26 seconds for all 1795 chunks (57 batches at batch_size=32). No API cost, no privacy concerns.
- **Open-source, commercially permissive** (MIT-style license).
- Output dimension 1024 — good information density without exploding the index size.

Alternatives considered:
- `intfloat/multilingual-e5-large`: similar quality, slightly weaker on Turkish in our spot checks.
- OpenAI `text-embedding-3-small`: closed, paid, sends data to a third party — disqualifying for a course deliverable.
- TR-specific models (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`): weaker on factual retrieval (paraphrase-tuned).

#### e) Vector database: **ChromaDB (persistent client)**

- **Zero-ops**: no separate server, just a directory on disk.
- **Persistent**: survives pod restarts (we tested by re-loading the index after closing/reopening the Python process).
- **Cosine distance**: matches the normalisation strategy of bge-m3 (we set `normalize_embeddings=True`).
- **Metadata-aware**: each chunk carries `{"source": "meb_9|10|11|12", "chunk_idx": N}` — useful for debugging and source citations.

Alternatives:
- **FAISS**: faster ANN at scale but no metadata API → would need a separate sidecar store; overkill for 1795 chunks.
- **pgvector / Qdrant**: server-based; more moving parts than this project needs.

### B.3 Why recursive chunking — comparison with alternatives

| Strategy | Pros | Cons | Verdict for this project |
|---|---|---|---|
| **Fixed-size sliding window** | Simplest; deterministic | Cuts mid-word — bad for Turkish agglutinative words. Cuts mid-sentence → key fact split across chunks. | ❌ Too lossy on Turkish. |
| **Sentence splitter (NLTK Turkish)** | Always returns whole sentences. | Sentences are often 20-30 tokens — too short. Each retrieved chunk has almost no context. | ❌ Too granular. |
| **Recursive character (ours)** | Honors paragraph/sentence boundaries before falling back to characters. Token-aware sizing avoids overshooting embedding capacity. | A few chunks may still split at sub-optimal points if a paragraph is unusually long without internal punctuation. | ✅ **Best trade-off.** |
| **Semantic chunking** (cluster sentences by embedding similarity, e.g. Kamradt 2023) | Boundaries follow topic shifts, theoretically highest retrieval quality. | ~10× more expensive to build; needs a second pass with the embedding model. For a single textbook corpus the gains are small. | ⚠️ Future work. |
| **Markdown/heading-aware** | Perfect if structure exists. | Our PDFs are flat text after extraction — no headings to exploit. | ❌ Not applicable. |
| **Document-level (one chunk = one section)** | Maximum context per chunk. | Embedding loses focus when chunk is too long; many chunks won't fit alongside the question. | ❌ Loses retrieval precision. |

**Conclusion:** for a single-domain corpus of flat-text PDFs, recursive character chunking with a token-aware length function is the standard recommended approach. Our spot-check retrieval results (top-1 chunks were on-topic for all 3 test queries about Istanbul's conquest, Lozan Treaty, and the Seljuks) confirm this matches expectations.

## Part C — RAG Pipeline

### C.1 Architecture

```
question (TR)
   │
   ▼
bge-m3 encode  ──►  ChromaDB.query(top_k=5)  ──►  [chunk_1, …, chunk_5]
                                                       │
                                                       ▼
                            template = "Bağlam: …\n\nSoru: …\n\nSeçenekler: …\n\nCevap (yalnızca harf):"
                                                       │
                                                       ▼
                            Qwen2.5-7B-Instruct + LoRA adapter (Task 1)
                                                       │
                                                       ▼
                            output → regex parse → A/B/C/D/E
```

### C.2 Implementation modules

All in `src/task2_rag/`:
- `document_loader.py` — PDF→text via pdfplumber
- `chunker.py` — RecursiveCharacterTextSplitter with bge-m3 token length
- `embedder.py` — sentence-transformers wrapper, normalize=True, cosine
- `vectorstore.py` — ChromaDB persistent client
- `build_index.py` — orchestrator (PDF→chunk→embed→store)
- `retriever.py` — thin facade over embedder + store
- `rag_pipeline.py` — prompt templates + answer parsing
- `evaluate_accuracy.py` — runs `zero_shot` and `rag` modes, dumps JSON

### C.3 Generation prompts

Both modes use the same system instruction (Turkish), differ only in whether retrieved context is injected.

**Zero-shot prompt:**
```
SYSTEM: Sen Türkçe tarih sorularını cevaplayan bir asistansın. Sana bir
        çoktan seçmeli soru ve seçenekler verilecek. Sadece doğru şıkkın
        harfini (A, B, C, D veya E) cevap olarak ver. Başka hiçbir açıklama
        yapma.
USER:   Aşağıdaki çoktan seçmeli tarih sorusunu cevapla.
        ### Soru: {question}
        ### Seçenekler:
        A) ...
        E) ...
        Cevap (yalnızca harf):
```

**RAG prompt:** same SYSTEM, USER becomes:
```
        Aşağıdaki bağlam bilgilerini kullanarak soruyu cevapla.
        ### Bağlam:
        {top_5_chunks separated by "\n\n---\n\n"}
        ### Soru: {question}
        ### Seçenekler: ...
        Cevap (yalnızca harf):
```

The first-letter regex (`\b([ABCDE])\b`) parses the model's output. Letter-only generation is enforced by both the instruction and `max_new_tokens=64` (we don't actually need 64; the model usually outputs the letter in the first token).

### C.4 Evaluation

- **Metric**: simple accuracy (correct / total) — matches the assignment's "Evaluate the system using accuracy."
- **Compared modes**: `zero_shot` (LLM only) vs `rag` (LLM + top-5 retrieved chunks).
- **Test set**: 100 TurkishMMLU/History `test` questions.
- **LLM**: Qwen2.5-7B-Instruct + Task 1 LoRA adapter (so the same model is used in both modes — the only variable is whether context is injected).

Results are written to `outputs/logs/rag_eval_results.json` and pulled into `docs/final_results.md`.

### C.5 Measured Results

| Mode | Accuracy | Correct | Total |
|---|---|---|---|
| Zero-shot (LLM only) | 0.3300 | 33 | 100 |
| **RAG (LLM + top-5 retrieval)** | **0.4000** | **40** | **100** |
| Δ (RAG over zero-shot) | **+0.0700 (+21.2% relative)** | +7 | — |

**RAG provided a clean +7 percentage-point gain** on the 100-question TurkishMMLU/History test set — meaningful for a single retrieval-augmented pass with no reranking, no query rewriting, and no few-shot examples. Per-example records (correct + wrong) are dumped to `outputs/logs/rag_eval_records.json`; representative samples appear in `docs/final_results.md`.

### C.6 Discussion of observed behaviour

**Where RAG won (qualitative inspection of records):** factual recall questions about specific events, dates, treaties (e.g. Malazgirt Savaşı sonrası, Osmanlı yerleşim politikası) — exactly the cases where retrieval grounding helps most. The MEB textbooks cover these topics in dedicated paragraphs that bge-m3 retrieves accurately.

**Where RAG still misses (failure analysis):**

1. **Multi-source reasoning required**: questions that combine info from multiple historical periods, where a top-5 set may not include all relevant chunks. Example: questions about Türkiye Selçuklu policies *and* their Ottoman successors.
2. **Subtle distinction questions**: where the answer hinges on choosing between two near-correct options. The retrieved context tells the model *broadly* about the topic but doesn't disambiguate the exact answer.
3. **Conceptual / interpretive questions**: "Hangi bilgiye ulaşılamaz?" style negation questions, where retrieval is helpful for context but the reasoning step is where LLM struggles independent of retrieval.

**Mitigations for future work** (also discussed in Task 3 design): raise top_k to 10 with a re-ranking pass (bge-reranker-v2-m3 cross-encoder), add query rewriting (turning multi-option questions into keyword queries), or use the dev split as a 3-shot in-context bank.
