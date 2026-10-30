# Stage 5 — Transformer Architecture Deep Dive (Module 1.4)
### A phase-wise / file-wise / topic-wise learning prompt

This is the same kind of plan you used for Stage 4, but built for Module 1.4. It maps every
section of the module PDF to a concrete file you build in the project, in the exact order and
style you've been learning in (NumPy-by-hand first → PyTorch, small concrete numbers, verify
against real PyTorch, formula comments in the file).

**The module's own warning: this is 10–12 days, "do not compress."** It is the keystone of
Phase 1 — everything after it (KV-cache, quantization, serving, distributed training) assumes
you already have this mental model. You did Stage 4 fast *and* kept the depth; the risk here is
rushing attention/RoPE/Pre-LN and paying for it in Module 3. Treat ~2 days per phase as the floor.

---

## How to use this document

You feed it to ChatGPT **one topic block at a time**, not all at once. Each block below is a
ready-to-paste prompt. The workflow per topic is always the same four steps:

1. **Paste the topic's prompt** into ChatGPT. Read its explanation *before* any code.
2. **Build the file yourself** in PyCharm under `Stage 5/`, using the file name given.
3. **Run it, verify the "You've got it when" check** at the bottom of the block.
4. Only then move to the next topic. If a check fails, stay on that file.

A standing instruction to give ChatGPT once, at the very start of each session:

> I am learning the transformer architecture by building one applied Python file per concept.
> For every topic: explain the idea in plain language FIRST with a tiny concrete numeric example,
> THEN give me code to type myself. Use small, inspectable tensors (not random giant ones) so I
> can trace the numbers by hand. Add the formula as a comment above each function. When we use a
> PyTorch built-in, also show the by-hand version so I can confirm they match. Do not move to the
> next concept until I confirm my file runs and the check passes.

File naming follows your convention: `stage5_<topic>_<numpy|torch|compare>.py`.

---

## The whole Stage 5 at a glance

| Phase | Topic (PDF §) | File(s) you build |
|-------|---------------|-------------------|
| 1 | Why attention + Q/K/V + scaled dot-product (§2–5) | `stage5_self_attention_numpy.py`, `stage5_scaled_dot_product_attention_torch.py` |
| 2 | Multi-head attention (§6) | `stage5_multihead_attention_torch.py` |
| 3 | Positional encoding: sinusoidal, learned, RoPE (§7) | `stage5_positional_encoding_numpy.py`, `stage5_rope_torch.py` |
| 4 | FFN, residuals, Pre-LN, causal mask, full block (§8–10, 12, 13) | `stage5_causal_mask_torch.py`, `stage5_transformer_block_torch.py` |
| 5 | Decoder-only GPT + training (§11, 14) | `stage5_decoder_only_gpt.py` |
| 6 | Inference & scaling: KV-cache, GQA/MQA, MoE (§15–17) | `stage5_kv_cache_demo.py`, `stage5_attention_variants_compare.py`, `stage5_moe_router_demo.py` |
| 7 | Read real source: HF GPT-2 (§18) | `stage5_gpt2_source_map.md` (annotation, not a script) |

The arc: by the end of Phase 4 you can build one transformer block from scratch; by the end of
Phase 5 you have a tiny working GPT; Phase 6 is the inference-cost concepts that the rest of the
roadmap obsesses over; Phase 7 is the real payoff — opening GPT-2's actual source and recognizing
every line.

---

## PHASE 1 — Why attention, and the Q/K/V core
**PDF sections 2, 3, 4, 5. Budget ~2 days.** This is the single most important phase. If attention
clicks, the rest is assembly.

### Topic 1.1 — Self-attention by hand (file: `stage5_self_attention_numpy.py`)

> Teach me self-attention (Module 1.4 §4) from scratch in NumPy. Start with the intuition: for
> each token the model makes three vectors — Query ("what am I looking for"), Key ("what do I
> contain"), Value ("what do I actually contribute") — each from the same token embedding via its
> own weight matrix. Then walk me through, with a TINY concrete example I can trace by hand: use a
> sequence of 3 tokens, embedding dimension 4. Make up small integer-ish embeddings and small
> W_Q, W_K, W_V matrices. Compute Q, K, V; then the score matrix Q·Kᵀ; then show me, in actual
> numbers, how one token's new representation becomes a weighted blend of the Values. Put the
> formula as a comment above each step. No softmax or scaling yet — that's the next file.

**You've got it when:** you can point at the score matrix and say *in words* why entry (i, j) is
"how much token i's query matched token j's key," and you can see one output row is a weighted mix
of the three Value vectors.

### Topic 1.2 — Scaled dot-product attention (file: `stage5_scaled_dot_product_attention_torch.py`)

This is **Exercise 1** from the module and the most important formula in the whole module:
`Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V`.

> Now teach me the full scaled dot-product attention (Module 1.4 §5) in PyTorch. Write the function
> `scaled_dot_product_attention(Q, K, V, mask=None)` with the five steps commented: (1) scores =
> Q·Kᵀ, (2) scale by 1/√d_k, (3) optional mask, (4) softmax over the key dimension, (5) weighted
> sum with V. Then — the part I care about — explain WHY we divide by √d_k: what "softmax
> saturation" means, and why a saturated softmax kills the gradient (connect it to the
> vanishing-gradient idea I already know from Stage 4's sigmoid work). Show me with real numbers:
> compute the variance of Q·Kᵀ with and without the √d_k scaling for a larger d_k, so I can SEE
> the scores blow up. Test the function on random Q, K, V of shape (1, 4, 8) and confirm the output
> is (1, 4, 8) and each row of the attention-weight matrix sums to 1.

**You've got it when:** output shape is `(1, 4, 8)`, every attention-weight row sums to 1.0, and
you can explain the √d_k scaling without looking (the module's Self-Check Q1 — write your own
answer, then compare to the PDF's).

---

## PHASE 2 — Multi-head attention
**PDF section 6. Budget ~1.5 days.** This is **Exercise 2**.

### Topic 2.1 — Multi-head attention (file: `stage5_multihead_attention_torch.py`)

> Teach me multi-head attention (Module 1.4 §6). First the intuition: why split d_model into h
> heads instead of one big attention — one attention distribution can only "focus" one way at a
> time; multiple heads let the model track several kinds of relationships at once (syntax,
> coreference, long-range topic). Then build the `MultiHeadAttention(nn.Module)` from the module:
> one Linear each for W_q/W_k/W_v, a split-into-heads reshape, attention per head, concat, final
> W_o projection. Explain the shape at EVERY line — I want to follow (B, T, d_model) →
> (B, h, T, d_k) → back → (B, T, d_model). Build it with d_model=64, n_heads=8, so d_k=8. Verify
> the output shape matches the input, and print the attention weights of two different heads on the
> same input to confirm they genuinely differ.

**You've got it when:** output shape matches input `(B, T, 64)`, `d_k` comes out to 8, and the two
heads' attention weights are visibly different matrices.

---

## PHASE 3 — Positional encoding
**PDF section 7. Budget ~2 days.** Two files: the classic sinusoidal one (with a plot, your style),
and RoPE (what Llama/Mistral/Qwen actually use — worth real time).

### Topic 3.1 — Sinusoidal & learned positional encoding (file: `stage5_positional_encoding_numpy.py`)

> Teach me positional encoding (Module 1.4 §7). First explain WHY it's needed: attention is
> order-agnostic — it treats its input as a set, so permuting tokens permutes outputs identically
> and the attention weights between two tokens don't change. Word order matters, so we must inject
> position. Then build the sinusoidal encoding from the module formula (even dims = sin, odd dims =
> cos, geometrically increasing wavelength) in NumPy, and plot it as a heatmap with matplotlib
> (position on the y-axis, embedding dimension on the x-axis), like Fig 6. Point out in the plot
> where the "fast waves" (low dims) and "slow waves" (high dims) are. Then explain in one paragraph
> how LEARNED positional embeddings (GPT-2/BERT) differ and their one weakness: they can't
> generalize past the max position seen in training.

**You've got it when:** your heatmap looks like Fig 6, and you can state in one sentence why a
learned position table has "no entry for position 4097 if you only trained to 4096."

### Topic 3.2 — RoPE, rotary positional embeddings (file: `stage5_rope_torch.py`)

> Teach me RoPE — rotary positional embeddings (Module 1.4 §7), used by Llama, Mistral, Qwen.
> Explain the key difference from sinusoidal: instead of ADDING a position vector at the input,
> RoPE ROTATES the Q and K vectors by an angle proportional to their position, INSIDE every
> attention layer. The elegant consequence: the dot product between a rotated query and a rotated
> key depends only on their RELATIVE offset, not absolute positions. Build `build_rope_cache` and
> `apply_rope` from the module. Then PROVE the relative-offset property numerically: take two
> vectors, rotate them for positions (2, 5), then for positions (4, 7) — same offset of 3 — and
> show me the resulting Q·K dot product is (nearly) identical in both cases. That's the whole point
> of RoPE, so I want to see it in numbers.

**You've got it when:** your numeric check shows the dot product depends on the *gap* between
positions, not the absolute positions — and you can answer the module's Self-Check Q3 (what RoPE
solves vs sinusoidal) in your own words.

---

## PHASE 4 — The sublayers and assembling a full block
**PDF sections 8, 9, 10, 12, 13. Budget ~2.5 days.** This is where all the pieces come together
into one transformer block. **Exercise 3** lives here.

### Topic 4.1 — Causal mask (file: `stage5_causal_mask_torch.py`)

This is **Exercise 3**.

> Teach me the causal attention mask (Module 1.4 §12). Explain WHY decoder training needs it: for
> next-token prediction to be a valid signal, a token must never attend to tokens that come after
> it, or the model could "cheat" by peeking at the answer. Build `causal_mask(seq_len)` as a
> lower-triangular matrix. Then apply it inside the scaled_dot_product_attention I already wrote:
> feed in a short sequence, print the attention-weight matrix, and verify every entry ABOVE the
> diagonal is exactly 0 (no token attends to the future). Then remove the mask and confirm the
> upper triangle becomes non-zero. Also explain the efficiency insight: within one sequence of
> length n, the mask lets the model learn to predict from every prefix at once — n training signals
> from one forward pass.

**You've got it when:** with the mask, the attention matrix is lower-triangular (upper triangle all
zeros after softmax); without it, the upper triangle is non-zero.

### Topic 4.2 — FFN, residuals, Pre-LN, and the full block (file: `stage5_transformer_block_torch.py`)

> Teach me the three remaining sublayer pieces and then assemble the full transformer block
> (Module 1.4 §8, §9, §10, §13). (1) The feed-forward network (§8): a two-layer MLP that expands
> to ~4× d_model, applies GELU, projects back — it processes each token independently and holds
> ~2/3 of a transformer's parameters. (2) Residual connections (§9): y = x + Sublayer(x), and why
> they let us stack dozens of layers — the derivative of the "+ x" term is exactly 1, so there's
> always an unobstructed gradient path (connect this to what I learned about gradients in Stage 4).
> (3) Pre-LN vs Post-LN (§10): explain that modern LLMs put LayerNorm INSIDE the residual branch,
> BEFORE the sublayer, giving a clean gradient highway — and why Post-LN is finicky to train deep.
> Then build the `TransformerBlock(nn.Module)` from the module's §13 code (Pre-LN decoder block:
> LayerNorm → masked MHA → residual add → LayerNorm → FFN → residual add), reusing my
> MultiHeadAttention. Walk me through the two-line forward pass and confirm input and output shapes
> are identical (so blocks can stack).

**You've got it when:** the block's output shape equals its input shape, and you can recite the
"repeating pattern" (LN → attn → add → LN → FFN → add) from memory. This is the pattern you half-met
in Stage 4's LayerNorm file — note how `Linear → LayerNorm → Activation` grew into a real block.

---

## PHASE 5 — Decoder-only model and training it
**PDF sections 11, 14. Budget ~2 days.** The capstone: stack your blocks into a tiny working GPT
and train it. This is the Karpathy "build GPT" moment the module's Further Reading points to.

### Topic 5.1 — A tiny decoder-only GPT (file: `stage5_decoder_only_gpt.py`)

> Help me assemble a tiny decoder-only (GPT-style) model (Module 1.4 §11) from my transformer
> block, and TRAIN it on a tiny task so I see it actually learn. First explain the decoder-only
> shape (Fig 10): token embeddings + positional info at the bottom → N identical masked
> self-attention + FFN blocks → final LayerNorm → an LM head (a Linear to vocab size) producing a
> probability distribution over the next token. No encoder, no cross-attention. Explain the one
> training objective: predict the next token (given tokens 1..t, output a distribution for t+1),
> trained with cross-entropy (I know BCE/cross-entropy from Stage 4). Then let's keep it tiny and
> inspectable: a small vocabulary and a trivial sequence pattern the model can actually learn (e.g.
> learn to continue a simple repeating or counting pattern), d_model small, 2 blocks, 2 heads.
> Write the training loop (I know the loop: zero_grad → forward → loss → backward → step), plot the
> loss, and then GENERATE from the trained model token by token so I watch it produce the pattern.
> Also explain the §14 scale terms as we go — parameters, layers (depth), d_model (width), heads,
> context window — and roughly why param count scales with depth × width².

**You've got it when:** the loss drops and your trained mini-GPT generates the pattern correctly
when you sample from it token by token. This is the "I built a working transformer" milestone.

---

## PHASE 6 — Inference and scaling concepts
**PDF sections 15, 16, 17. Budget ~1.5 days.** These are concept-level in the module but they drive
every later infra decision, so build small demos rather than just reading.

### Topic 6.1 — KV-cache (file: `stage5_kv_cache_demo.py`)

> Teach me the KV-cache (Module 1.4 §15). Explain the waste it fixes: during autoregressive
> generation the model produces one token, appends it, and runs again — and naively it re-encodes
> the Keys and Values of ALL previous tokens every step, even though those are identical each step.
> The KV-cache stores past K and V so each new step only computes K/V for the single new token,
> turning per-step cost from quadratic to linear. Using my tiny GPT from Phase 5, let me generate a
> sequence TWO ways — once recomputing everything each step, once caching K/V — and (a) confirm the
> outputs are identical and (b) count the K/V computations each way so I SEE the savings. Then
> explain why infra engineers obsess over it: the KV-cache is often the dominant GPU-memory
> consumer at inference and grows linearly with sequence length × batch size.

**You've got it when:** both methods produce identical generations, and your computation count
shows the cached version does far less work as the sequence grows.

### Topic 6.2 — MHA vs GQA vs MQA (file: `stage5_attention_variants_compare.py`)

> Teach me the modern attention variants (Module 1.4 §16): Multi-Head (MHA, one KV per query head),
> Grouped-Query (GQA, query heads share KV heads in groups), and Multi-Query (MQA, all query heads
> share one KV head). Explain the tradeoff: fewer KV heads → smaller KV-cache → less memory and
> memory-bandwidth per token at inference, with GQA being the sweet spot (Llama 2/3, Mistral)
> because it keeps almost all the quality. In the style of my Stage 4 `_compare` files, build a
> small comparison: for a fixed d_model and n_query_heads, compute and print the KV-cache size for
> MHA vs GQA(groups=…) vs MQA so I can see the memory shrink. A tiny GQA attention forward pass
> would be a bonus, but the KV-cache size comparison is the main thing.

**You've got it when:** your table shows KV-cache size dropping MHA → GQA → MQA, and you can say in
one sentence why GQA is the modern default.

### Topic 6.3 — Mixture of Experts router (file: `stage5_moe_router_demo.py`) — lighter/optional

> Teach me Mixture of Experts (Module 1.4 §17) at the concept level. Explain that MoE replaces the
> single FFN in a block with several parallel FFN "experts" plus a lightweight router that, per
> token, picks only the top-k experts to actually run (e.g. 2 of 8 in Mixtral) — "sparse
> activation": huge TOTAL parameter count (capacity) but small ACTIVE count per token (compute).
> Build just the router part as a demo: a Linear that scores experts for each token, a top-k
> selection, and show which experts each token would be routed to. Then explain the infra catch —
> experts must be spread across GPUs (expert parallelism), the router can send uneven token counts
> to experts (load imbalance), and all total params still have to fit in memory.

**You've got it when:** your router sends different tokens to different top-k experts, and you can
explain "sparse activation" (large total params, small active params per token) in one sentence.

---

## PHASE 7 — Read real model source
**PDF section 18. Budget ~1 day.** This is **Exercise 4** and the real goal of the whole module:
open a real model and recognize every piece.

### Topic 7.1 — Map GPT-2's source to concepts (file: `stage5_gpt2_source_map.md`)

This one isn't a script you run — it's an annotation file you write while reading real code.

> I'm going to open Hugging Face's `modeling_gpt2.py` and map its source to the concepts in Module
> 1.4, using the §18 map as my guide. Act as my reading partner: as I find each piece — token
> embedding (wte), learned positional embedding (wpe), the ModuleList of N blocks, the final
> LayerNorm, the per-block Pre-LN placement, the fused QKV projection (Conv1D of 3·d_model), the
> causal mask buffer, the attention math, and the MLP/FFN — confirm I've matched it to the right
> concept number and correct me if I mis-map one. I'll write a one-line comment next to each
> mapping in a markdown file. Quiz me at the end: point at a line and ask which concept it is.

**You've got it when:** you can open the file cold and name what every major block is — the module's
stated goal: "open a real model implementation and map every block of code to a concept, without
confusion." That's the skill you'll use every time you onboard onto a new model codebase.

---

## Finishing the module

When all seven phases are done, close it out the way the module intends:

- **Write your own answers to the three Self-Check Questions (§21)** — √d_k scaling, why decoder-only
  dominates, what RoPE solves — *before* reading the PDF's answers, then compare.
- **Then read the "Attention Is All You Need" paper (§22).** The module is explicit that you should
  read it *after* this module, when you finally have the vocabulary — "it will finally click end to
  end." Jay Alammar's "Illustrated Transformer" and Karpathy's "Let's build GPT" video are the two
  highest-value companions if a topic stays fuzzy.
- **One retention habit** (same advice as after Stage 4): a week after you finish, re-derive two
  things from memory with no notes — the five steps of scaled dot-product attention, and why Pre-LN
  gives a stable gradient highway. If both come back easily, the module stuck.

A realistic self-check on pace: if you can walk into Module 3 (serving, KV-cache management,
quantization) and the KV-cache and GQA from Phase 6 still feel solid without a refresher, Stage 5
worked. If they feel shaky, that's the signal to slow down and re-run Phase 6, not to push ahead.
