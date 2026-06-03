# Task 1 — Part A: LoRA Theoretical Background

## 1. The LoRA Approach

**LoRA (Low-Rank Adaptation)** was introduced by Hu et al. (2021, ICLR 2022) as a parameter-efficient method for adapting large pre-trained models to downstream tasks. The core hypothesis is that the *task-specific update* `ΔW` to a pre-trained weight matrix `W ∈ R^{d×k}` has **low intrinsic rank** — i.e. the effective dimensionality of useful task-conditioned changes is far smaller than `d×k`.

LoRA exploits this by freezing the original weights and parameterising the update as a product of two low-rank matrices:

$$
W' = W + \Delta W = W + \frac{\alpha}{r} \cdot B A
$$

where
- `A ∈ R^{r×k}`, `B ∈ R^{d×r}`, and `r ≪ min(d, k)`;
- `A` is initialised with Kaiming-uniform random values;
- `B` is initialised to **zero**, so at training start `BA = 0` and `W' = W` exactly (the adapted model is initially identical to the base model — no destabilisation);
- `α` is a scaling hyperparameter; the `α/r` factor decouples the choice of `r` from the effective learning rate.

During training, only `A` and `B` receive gradients; the much larger `W` is frozen. At inference, one can either keep them separate (small extra cost) or **merge** them via `W ← W + (α/r)BA` (zero extra cost).

For Qwen2.5-7B with our configuration (`r=16`, target = attention + MLP projections), this yields:
- ~40 M trainable parameters out of ~7.66 B total = **~0.53 % trainable**
- Adapter file size on disk: **~154 MB** (saved as fp32 safetensors; ~80 MB if stored in bf16) vs. full checkpoint **~14 GB**

## 2. Advantages over Full Fine-Tuning

### 2.1 Computational efficiency
Full fine-tuning backpropagates through every weight in the model; LoRA backpropagates only through `A` and `B`. Although the forward pass still touches `W` (so the activations have the same shape), the **gradient computation per parameter** is dramatically smaller. In our run, ~99.5 % of parameters are frozen → optimizer state, gradients, and momentum buffers all scale with the trainable count, not the model size.

### 2.2 Memory efficiency
This is the single biggest practical advantage. Adam-style optimizers store two extra fp32 tensors per parameter (first + second moments). For a full 7B fp16 model:
- Weights: 14 GB
- Adam state (fp32): ~56 GB
- Total: ~70 GB → impossible on our 32 GB RTX PRO 4500

With LoRA (40 M trainable):
- Frozen 4-bit base (QLoRA): ~5 GB
- LoRA params (bf16, in memory): ~80 MB
- Adam state (paged 8-bit): ~160 MB
- Activations + grads at seq 256, batch 4: ~8-10 GB
- **Total: ~14-16 GB, comfortably on a single 32 GB GPU.**

### 2.3 Training speed
Less gradient compute → faster steps. In our setup, QLoRA gives ~2.3 sec/step on a single RTX PRO 4500 Blackwell at seq 256 / batch 4. Equivalent full fine-tuning of a 7B model isn't even tractable on this hardware.

### 2.4 Storage requirements
- Full fine-tune: one ~14 GB checkpoint per task variant.
- LoRA: one base model + one ~154 MB adapter per task (fp32 on disk; ~80 MB in bf16). **~90× smaller per task** (fp32), up to **~175× smaller** if the adapter is stored in bf16.

This matters when you fine-tune for many languages / domains: the base model is shared, only the adapter varies.

### 2.5 Reusability of adapters
Adapters are **composable, swappable artifacts**:

- *Swappable*: load the base once, plug in different LoRA adapters at inference for different tasks (translation, summarisation, code) with seconds-level switch time.
- *Composable*: you can sum multiple adapters' deltas (`ΔW_total = ΔW_1 + ΔW_2 + …`) to combine capabilities, with caveats about interference.
- *Merge-friendly*: once a single adapter is finalized, merging `W ← W + BA` produces a standard checkpoint with zero inference overhead.

For us: the same Qwen2.5-7B serves Task 1 (translation) and Task 2 (RAG QA) — we just toggle the adapter on/off.

## 3. Integration into the Transformer

### 3.1 Which layers are modified

The original LoRA paper applied adapters to `W_q` and `W_v` of self-attention. Subsequent practice (and our config) extends to the **full attention block** and the **MLP/FFN block**, since for non-trivial style adaptation (e.g. learning a new translation register) the MLP is critical too. Our target modules:

| Sub-block | Modules | Shape (for Qwen2.5-7B) |
|---|---|---|
| Attention | `q_proj, k_proj, v_proj, o_proj` | each `3584 × 3584` (some have GQA variants) |
| MLP | `gate_proj, up_proj, down_proj` | each `3584 × 18944` |

LayerNorms, embeddings, and the LM head are **not** wrapped — they're either tiny (norms) or shared across tasks (embeddings/LM head).

### 3.2 How low-rank matrices are used

For each targeted projection `h = Wx`, LoRA substitutes:

$$
h = Wx + \frac{\alpha}{r} \cdot B(Ax)
$$

The hidden vector `x` is first projected down by `A` (`R^{r×k}`) into the low-rank space, then back up by `B` (`R^{d×r}`). The dimension reduction is critical: it forces the adapter to learn a *compressed* representation of the task-specific transformation.

Empirically, `r=8` to `r=64` covers most use-cases. For our setup we picked `r=16` as a sweet spot — large enough to capture the style of an MT task, small enough to keep training cheap (40 M trainable parameters).

### 3.3 How frozen and trainable parameters interact

At each forward step:
1. `W` (4-bit quantized via QLoRA) is dequantized **on-the-fly per matmul tile** by `bitsandbytes`. Result lives only in registers — no full fp16 copy ever appears in VRAM.
2. `Wx` and `BAx` are computed in the same bf16 compute dtype and **added**.
3. The combined result feeds the next layer normally.

At each backward step:
1. Gradients flow through the residual stream as normal.
2. The chain rule yields `∂L/∂A` and `∂L/∂B`. `∂L/∂W` is computed by autograd but **dropped** (we don't keep it — the weight is frozen).
3. PEFT explicitly marks `W.requires_grad = False`, so the optimizer skips it entirely.

The **QLoRA additions** (Dettmers et al., 2023, *QLoRA*):
- 4-bit NF4 quantization with double quantization,
- bf16 compute dtype,
- paged 8-bit AdamW for optimizer state,

bring the 7B model from ~14 GB (fp16) down to ~5 GB resident, leaving plenty of headroom for activations on our 32 GB card.

## References

- **Hu et al., 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. <https://arxiv.org/abs/2106.09685>
- **Dettmers et al., 2023.** *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. <https://arxiv.org/abs/2305.14314>
- **Mangrulkar et al., 2023.** *PEFT: Parameter-Efficient Fine-Tuning library.* HuggingFace. <https://github.com/huggingface/peft>
