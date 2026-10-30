# Python(AI-ML) — Building Deep Learning From Scratch

A hands-on curriculum that builds up to a working GPT, one concept per file. This repository is
**Phase 1** of an ML Infrastructure Engineering roadmap: it starts from a single neuron computed
with a `for` loop and ends with a trainable, RoPE-based decoder-only language model with a KV-cache
— every mechanism implemented and understood from first principles before a library is allowed to
do it.

Each idea is built twice: first **from scratch in NumPy (or plain Python)** so the math is visible,
then **re-expressed in PyTorch** to see the library do the same thing. Nothing is a black box.

---

## The learning arc

The whole repository is one continuous thread. Each stage adds exactly one layer of capability over
the last:

```
Stage 1   a single neuron            →  weighted sum + bias + activation
Stage 2   XOR + backpropagation      →  multi-layer network that LEARNS
Stage 3   the PyTorch pipeline       →  Dataset / DataLoader / training loop
Stage 4   the components             →  activations, losses, optimizers, norm, regularization
Stage 5   the transformer            →  attention → block → a working mini-GPT
Stage 6   tokenization & embeddings  →  how real text becomes model input
```

A single toy problem (first a 2-input neuron, then the **XOR** dataset) is reused as a fixed
baseline through Stages 1–4, so each new concept is the *only* thing that changes between files.
Stages 5–6 graduate to the transformer and to real text.

**Roadmap mapping:** Stage 1–2 ≈ Module 1.2 (neural nets & backprop) · Stage 3 ≈ Module 1.3 (PyTorch
fluency) · Stage 4 ≈ the deeper NN components · Stage 5 = Module 1.4 (transformers) · Stage 6 =
Module 1.5 (tokenization). Module 1.1 (math) was pen-and-paper; the Stage 5 mini-GPT is the
Module 1.6 capstone.

---

## How to read this repository

Five conventions repeat across every stage. Once you know them, any file is easy to navigate:

- **NumPy-first, then PyTorch.** A concept is derived by hand (`*_numpy.py`) before the PyTorch
  version (`*_torch.py`) is written. The pairing lets you confirm the library reproduces your math.
- **`*_compare.py` files contrast two things** head-to-head — two activations, two losses, three
  optimizers, hand-built vs library module — usually trained from the **same random seed**
  (`torch.manual_seed(...)`) so the only difference is the one thing being studied.
- **Formula comments** sit above almost every function, stating the math the code implements.
- **Verification is built in.** Files print intermediate tensors, check row-sums, or `assert`
  numeric results (e.g. hand-softmax == `torch.softmax`, `decode(encode(x)) == x`) so you can *see*
  that each piece is correct.
- **Loss curves and plots** (matplotlib) appear throughout Stages 3–5 to make training visible.

---

## Stage 1 — A single neuron

One neuron (weighted sum + bias + threshold), implemented the *same way three times* to show the
raw loop, NumPy, and PyTorch all compute the identical thing. Weights are deliberately matched
across all three files so they produce the same output.

| File | Teaches | How it differs |
|------|---------|----------------|
| `perceptron.py` | A perceptron as an explicit `for`-loop dot product + bias + step | The seed of everything — no libraries at all |
| `numpy_perceptron.py` | The same neuron with the loop replaced by `np.dot` | Adds vectorization; prints the raw pre-activation `z` |
| `torch_perceptron.py` | The same neuron as `nn.Linear(2,1)` | Introduces PyTorch layers, `torch.no_grad()`, `.item()`; weights copied in to match the NumPy result |

---

## Stage 2 — XOR and backpropagation

Solve XOR (`[[0,0],[0,1],[1,0],[1,1]] → [0,1,1,0]`) with a 2→2→1 MLP. This stage walks from a
hand-wired network all the way to a fully trained one, deriving backprop by hand before letting
PyTorch take over.

| File | Teaches | How it differs / adds |
|------|---------|-----------------------|
| `xor_numpy.py` | Why one neuron can't do XOR; a 2-layer forward pass with hand-set weights | Jumps to a multi-layer network and a batch of 4; still no learning |
| `xor_mlp_numpy.py` | Adds an **MSE loss** on top of the forward pass | Editable prediction so you watch the loss number move |
| `xor_backprop_numpy.py` | **Manual backprop** (chain rule) for the output layer; introduces sigmoid | Switches step → sigmoid (step has no usable gradient); computes `dA2, dZ2, dW2, db2` |
| `xor_backprop_update-param.py` | **Full backprop + gradient-descent update in a loop** — the first net that *learns* | Adds hidden-layer gradients (`dW1, db1`) and `w -= lr*dw`; 20 000 epochs |
| `xor_mlp_torch.py` | The same trained MLP as `nn.Module`, with **autograd** | `loss.backward()` replaces hand-derived gradients; update still written manually |
| `xor_mlp-with-optimizer_torch.py` | Replaces the manual update with `torch.optim.SGD` | One line — `optimizer.step()` — replaces the whole hand-written update block |

**Arc:** hand-set weights → + loss → + hand backprop → + update rule (learns) → autograd → optimizer.

---

## Stage 3 — The PyTorch data & training pipeline

The same XOR model, now wrapped in PyTorch's standard machinery. The files are incremental build-ups
of one final training script.

| File | Teaches | How it differs / adds |
|------|---------|-----------------------|
| `stage3_dataset.py` | The `Dataset` abstraction (`__init__/__len__/__getitem__`) | Data wrapper only; no model |
| `stage3_dataloader.py` | Wrapping a Dataset in a `DataLoader` for **batches** | Adds `batch_size=2` and batch iteration |
| `stage3_batches.py` | How `batch_size` changes batching | Same as above with `batch_size=1` — a focused mini-experiment |
| `stage3_training_loop.py` | The **nested loop**: epochs outside, batches inside | First file to combine model + DataLoader into the canonical loop |
| `stage3_loss_tracking.py` | Tracking and **plotting** the loss curve; `model.train()` | Adds matplotlib loss visualization |
| `stage3_final.py` | The complete pipeline: train → `eval()` → `no_grad` → plot | The capstone that assembles every Stage-3 piece; longer run |

---

## Stage 4 — The components of real networks

Each building block of a modern network, derived in NumPy and/or compared head-to-head in PyTorch
with same-seed fairness. Most files share the XOR data; the overfitting pair introduces a new
noisy-sine regression dataset.

| File | Teaches | How it differs / adds |
|------|---------|-----------------------|
| `stage4_activations_numpy.py` | ReLU, Tanh, GELU, Sigmoid **and their derivatives**; the vanishing-gradient picture | Generalizes Stage 2's lone sigmoid into a family; two comparison plots |
| `stage4_activations_compare.py` | Sigmoid vs ReLU on XOR; the "dead ReLU" problem | Same-seed comparison; widens hidden layer 2 → 8 to avoid dead ReLU |
| `stage4_crossentropy_numpy.py` | **BCE vs MSE** and why BCE+sigmoid gives a clean gradient | Reuses Stage 2's gradient framing with a new loss |
| `stage4_crossentropy_torch.py` | `nn.MSELoss` vs `nn.BCELoss` in practice | PyTorch counterpart; warns the two loss *scales* aren't directly comparable |
| `stage4_optimizers_compare.py` | **SGD vs Adam vs AdamW** — momentum & adaptive LR | Extends Stage 2's manual update into modern optimizers; detailed formula block |
| `stage4_lr_schedules.py` | **LR schedules** — StepLR, cosine, warmup+cosine | Sits on top of the optimizer; the LR itself now varies over training |
| `stage4_overfit_demo.py` | **Overfitting vs underfitting**, train/test split | First non-XOR task (noisy sine); tiny vs huge model; tracks train AND test loss |
| `stage4_regularization.py` | **Dropout and weight decay (L2)** as remedies | Directly fixes the overfitting the previous file showed; 3-way comparison |
| `stage4_layernorm_numpy.py` | **LayerNorm by hand** — per-sample feature normalization + γ/β | Derives it before the PyTorch version; contrasts with BatchNorm |
| `stage4_layernorm_torch.py` | `nn.LayerNorm` in the `Linear → LayerNorm → Activation` pattern | The exact pattern reused inside every transformer block in Stage 5 |
| `stage4_save_load.py` | **Saving/loading weights & full checkpoints**; resuming training | The only file that writes to disk (`xor_weights.pth`, `xor_checkpoint.pth`) |

---

## Stage 5 — The transformer, built piece by piece

The heart of the repository. A transformer decoder is assembled one component at a time, each file
adding the thing the previous one omitted, culminating in a **trainable mini-GPT**. Toy tensors are
kept tiny and inspectable throughout.

### The core build-up (each adds to the previous)

| File | Teaches | How it differs / adds |
|------|---------|-----------------------|
| `stage5_self_attention_numpy.py` | Raw Q/K/V and the weighted sum of values | The skeleton — **deliberately no softmax, no scaling** yet |
| `stage5_scaled_dot_product_attention_torch.py` | Full scaled dot-product attention | Adds `1/√d_k` scaling, softmax, optional mask; numeric experiments show *why* scaling is needed |
| `stage5_multihead_attention_torch.py` | Multi-head attention from scratch | Adds learnable W_q/k/v/o projections and head splitting/merging |
| `stage5_multihead_attention_compare.py` | Hand-built MHA vs `nn.MultiheadAttention` | Confirms the hand-rolled version matches PyTorch's shapes/interface |
| `stage5_causal_mask_torch.py` | The **causal (look-ahead) mask** | Turns bidirectional attention into a *decoder*; proves future positions get zero weight |
| `stage5_ffn_torch.py` | The position-wise **feed-forward network** + GELU | The second sublayer of a block; expand → GELU → contract |
| `stage5_residual_preln_torch.py` | **Residual connections, LayerNorm, Pre-LN vs Post-LN** | The "glue" for deep stacks; proves Pre-LN vs Post-LN orderings |
| `stage5_transformer_block_torch.py` | **The complete transformer block** | The convergence point: MHA + causal mask + FFN + Pre-LN in one reusable block |

### The positional track (feeds the block/GPT)

| File | Teaches | How it differs / adds |
|------|---------|-----------------------|
| `stage5_positional_encoding.py` | Classic **sinusoidal** absolute position encoding | The alternative approach; standalone demo with a heatmap |
| `stage5_rope_2d_demo.py` | The 2-D rotation primitive behind RoPE | Prerequisite: position → rotation angle, proven two ways |
| `stage5_rope_torch_final.py` | Full **RoPE** (rotary positional embeddings) | Generalizes rotation to a full vector; proves attention depends on the *relative* offset — this is the positioning used in the mini-GPT |

### The culmination — a working mini-GPT

| File | Teaches | How it differs / adds |
|------|---------|-----------------------|
| `stage5_decoder_only_gpt.py` | **`TinyGPT`** — embeddings → N RoPE blocks → final LN → LM head, trained on next-token prediction | Combines the block + RoPE + embeddings + a real training loop + generation. The first runnable language model |
| `stage5_decoder_KV_cache.py` | The same TinyGPT **plus a KV-cache** for fast generation | Adds `start_pos` RoPE, a rectangular causal mask, cache concatenation, and compute counters; proves identical output at far less compute |

### Side studies (depend on the core, don't extend the main chain)

| File | Teaches | How it differs / adds |
|------|---------|-----------------------|
| `stage5_attention_variants_compare.py` | **MHA vs GQA vs MQA** and their KV-cache cost | Varies only the number of K/V heads; a worked GQA forward pass |
| `stage5_moe_router_demo.py` | **Mixture-of-Experts** routing | An alternative to the plain FFN: a router picks top-k experts per token (sparse activation) |

---

## Stage 6 — Tokenization & embeddings

How raw text becomes the integer-then-vector input a transformer actually consumes. This is the
*input stage* that sits in front of the Stage 5 model.

| File | Teaches | How it differs / adds |
|------|---------|-----------------------|
| `stage6_text_to_vector.py` | The **text → token IDs → embedding vectors** pipeline; `nn.Embedding` as a row lookup | Provides the `token_embedding` used by TinyGPT; shows embeddings are content-based, position-free |
| `stage6_different_tokenization_type.py` | **Word vs character vs subword** tokenization; the OOV problem; the n² attention cost | Motivates *why* subword tokenization exists; hand-splits (not yet learned) |
| `stage6_bpe_from_scratch.py` | **Byte-Pair Encoding trained from scratch** — character-level then byte-level | *Learns* subword units instead of hand-splitting; byte-level guarantees `decode(encode(x)) == x` for any script (English, Hindi, emoji) |

---

## Dependency map

How the files build on one another (→ means "directly builds on / adds to"):

```
Stage 1:  perceptron → numpy_perceptron → torch_perceptron
                                                  │
Stage 2:  xor_numpy → xor_mlp_numpy → xor_backprop_numpy → xor_backprop_update-param
                                                                      │
                            xor_mlp_torch → xor_mlp-with-optimizer_torch
                                                  │
Stage 3:  (dataset → dataloader → batches)  +  training_loop → loss_tracking → final
                                                  │
Stage 4:  swaps ONE component at a time on that pipeline:
          activations · cross-entropy · optimizers · LR schedules ·
          overfitting → regularization · layernorm · save/load
                                                  │
Stage 5:  self_attention → scaled_dot_product → multihead → causal_mask
          → ffn → residual_preln → TRANSFORMER_BLOCK
          (positional: sinusoidal ; rope_2d → rope_final)
                                                  │
                      ┌───────────────────────────┴───────────────┐
            decoder_only_gpt  (TinyGPT)              decoder_KV_cache (same + cache)
                      │
          side studies: attention_variants (GQA/MQA) · moe_router

Stage 6:  text_to_vector → different_tokenization_type → bpe_from_scratch
          (the input pipeline that feeds TinyGPT's embedding layer)
```

The two files that together represent the finished mini-GPT are
**`stage5_decoder_only_gpt.py`** (architecture + training) and
**`stage5_decoder_KV_cache.py`** (the same model with inference-time KV-caching).

---

## Concept index

Where each major idea is taught:

| Concept | File(s) |
|---------|---------|
| Perceptron, dot product, `nn.Linear` | Stage 1 (all three) |
| MLP / hidden layers | `xor_numpy.py`, `xor_mlp_numpy.py` |
| Manual backpropagation (chain rule) | `xor_backprop_numpy.py`, `xor_backprop_update-param.py` |
| Autograd & optimizers | `xor_mlp_torch.py`, `xor_mlp-with-optimizer_torch.py`, `stage4_optimizers_compare.py` |
| Dataset / DataLoader / training loop | Stage 3 (all) |
| `train()` / `eval()` / `no_grad` | `stage3_final.py` |
| Activation functions & vanishing gradient | `stage4_activations_numpy.py`, `stage4_activations_compare.py` |
| Cross-entropy vs MSE | `stage4_crossentropy_numpy.py`, `stage4_crossentropy_torch.py` |
| LR schedules (warmup / cosine) | `stage4_lr_schedules.py` |
| Overfitting, dropout, weight decay | `stage4_overfit_demo.py`, `stage4_regularization.py` |
| LayerNorm | `stage4_layernorm_numpy.py`, `stage4_layernorm_torch.py`, `stage5_residual_preln_torch.py` |
| Save / load / checkpoint | `stage4_save_load.py` |
| Self-attention (Q/K/V) | `stage5_self_attention_numpy.py` |
| Scaled dot-product + softmax + √d_k | `stage5_scaled_dot_product_attention_torch.py` |
| Multi-head attention | `stage5_multihead_attention_torch.py`, `stage5_multihead_attention_compare.py` |
| Positional encoding (sinusoidal) | `stage5_positional_encoding.py` |
| RoPE (rotary embeddings) | `stage5_rope_2d_demo.py`, `stage5_rope_torch_final.py` |
| Causal masking | `stage5_causal_mask_torch.py` |
| Feed-forward network + GELU | `stage5_ffn_torch.py` |
| Residual & Pre-LN vs Post-LN | `stage5_residual_preln_torch.py` |
| Transformer block | `stage5_transformer_block_torch.py` |
| Decoder-only GPT | `stage5_decoder_only_gpt.py` |
| KV-cache | `stage5_decoder_KV_cache.py` |
| GQA / MQA | `stage5_attention_variants_compare.py` |
| Mixture of Experts | `stage5_moe_router_demo.py` |
| Token embeddings (lookup table) | `stage6_text_to_vector.py` |
| Tokenization types & OOV | `stage6_different_tokenization_type.py` |
| Byte-Pair Encoding (char + byte-level) | `stage6_bpe_from_scratch.py` |

---

## Tech stack

Python · NumPy · PyTorch · matplotlib. Everything runs on CPU — models are intentionally tiny so
each concept stays inspectable. No external tokenizer or transformer library is used; BPE and the
transformer are built from scratch.

---

## Notes

- `stage5_decoder_only_gpt.py` and `stage5_decoder_KV_cache.py` share the same `TinyGPT`
  architecture; the KV-cache file is the base model plus the five labelled generation changes.
- `stage6_bpe_from_scratch.py` is organized as Topic 2.1 (character-level BPE) followed by
  Topic 2.2 (byte-level BPE), each section appearing once; running it prints the merge steps and
  passes the merge/encoding verification and the English/Hindi/Emoji round-trip checks.
- Stages 1–4 reuse the XOR dataset and a 2→2→1 (later 2→8→1) model as a fixed baseline so each new
  concept is the only variable.
