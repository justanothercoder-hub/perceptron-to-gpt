# Stage 4 Plan — Filling the Gaps Before Transformers

## Context for the AI Assistant

You are helping a beginner learner named Samikshan build ML/DL understanding from scratch through hands-on coding. He is working in a PyCharm project at `C:\PractiseCoding\Python(AI-ML)` with Python 3.14.7 and has completed Stages 1–3 already.

**His learning style (CRITICAL — follow this exactly):**
- He learns one concept at a time, with concrete numbers and small examples first
- He needs to understand *why* before writing code — no "just trust me" explanations
- He verifies understanding by building it himself — you explain, he writes the code, you check it
- He likes analogies and plain-language explanations (e.g., backpropagation was taught as a "blame relay race")
- He gets stuck on conceptual parts, not syntax — so spend time on the *why* and *intuition*
- He uses NumPy first to see the raw math, then moves to PyTorch to see the framework version
- Every concept should be grounded in his existing XOR project or a natural extension of it

**What he already knows and has built (Stages 1–3):**

### Stage 1 — Single Perceptron
- Built a perceptron from scratch in raw Python, NumPy, and PyTorch
- Understands: z = x·w + b, step activation, decision boundaries as straight lines
- Files: `Stage 1/perceptron.py`, `Stage 1/numpy_perceptron.py`, `Stage 1/torch_perceptron.py`

### Stage 2 — XOR and Multi-Layer Perceptron
- Proved why XOR can't be solved by a single perceptron (mathematical proof + visualization)
- Built a 2-layer MLP (hidden: 2 neurons, output: 1 neuron) that solves XOR
- Implemented full forward pass with vectorized NumPy: `Z1 = X @ W1.T + b1`, `A1 = sigmoid(Z1)`, etc.
- Implemented complete backpropagation from scratch in NumPy with all gradient computations
- Implemented training loop with manual parameter updates over 20k epochs
- Ported to PyTorch: first with manual weight updates, then with `torch.optim.SGD`
- Understands: sigmoid and its derivative, MSE loss, chain rule as multiplication of sensitivities, why `db = Σ dZ`, the difference between "relay batons" (dA1, dZ1) and "keepers" (dW1, db1)
- Files: `Stage 2/xor_numpy.py`, `Stage 2/xor_mlp_numpy.py`, `Stage 2/xor_backprop_numpy.py`, `Stage 2/xor_backprop_update-param.py`, `Stage 2/xor_mlp_torch.py`, `Stage 2/xor_mlp-with-optimizer_torch.py`

### Stage 3 — PyTorch Training Pipeline
- Built Dataset and DataLoader from scratch for XOR data
- Understands `__init__`, `self`, `super().__init__()`, class inheritance with `nn.Module`
- Built proper training loop: epoch → batch → zero_grad → forward → loss → backward → step
- Added loss tracking with matplotlib visualization
- Understands `model.train()` vs `model.eval()` and `torch.no_grad()`
- Final model achieves predictions ≈ [0.011, 0.989, 0.989, 0.014] on XOR
- Files: `Stage 3/stage3_dataset.py`, `Stage 3/stage3_dataloader.py`, `Stage 3/stage3_batches.py`, `Stage 3/stage3_training_loop.py`, `Stage 3/stage3_loss_tracking.py`, `Stage 3/stage3_final.py`

### What he does NOT yet know (and has never seen):
- Any activation function other than sigmoid and step
- Any loss function other than MSE
- Any optimizer other than vanilla SGD (manual and torch.optim.SGD)
- Learning rate schedules
- The concepts of overfitting and underfitting
- Regularization techniques (dropout, L2/weight decay)
- Layer normalization
- Saving and loading models

---

## The Stage 4 Plan

Stage 4 fills the remaining gaps in Module 1 of his ML roadmap before he moves to sequence models and transformers. Each step below is one concept — teach it, let him build it, verify it works, then move on. All new files go in a `Stage 4/` folder inside the project.

---

### Step 1: Activation Functions — ReLU, Tanh, GELU

**Why this is next:** He only knows sigmoid. Sigmoid has a vanishing gradient problem — when z is very large or very small, the derivative is nearly zero, so gradients die during backpropagation. He needs to understand why modern networks use ReLU and its variants instead.

**What to teach:**
1. **The vanishing gradient problem with sigmoid.** Use his existing knowledge: sigmoid_derivative(10) ≈ 0.0000454. If you multiply several of these through layers, gradients become microscopically small. His XOR network only has 2 layers so it works fine, but deeper networks would starve.
2. **ReLU: `max(0, x)`** — derivative is 1 when x > 0, 0 when x ≤ 0. No vanishing gradient on the positive side. Dead neuron problem: if a neuron's output is always negative, it never learns.
3. **Tanh: `(e^x - e^-x) / (e^x + e^-x)`** — outputs between -1 and +1 (centered around zero, unlike sigmoid's 0 to 1). Still has vanishing gradient at extremes, but less severe than sigmoid.
4. **GELU: `x · Φ(x)`** where Φ is the standard normal CDF — smooth approximation of ReLU, used in transformers. Don't need to memorize the formula, just understand it's "ReLU but smooth."

**What to build:**
- `stage4_activations_numpy.py` — Implement all 4 activation functions in NumPy. Plot them side by side (4 subplots). Also plot their derivatives side by side. Visually compare where gradients vanish.
- `stage4_activations_compare.py` — Take his Stage 3 XOR model (`stage3_final.py`) and swap sigmoid for ReLU in the hidden layer. Train both versions, plot their loss curves on the same graph. See which converges faster.

**Suggested filename:** `Stage 4/stage4_activations_numpy.py`, `Stage 4/stage4_activations_compare.py`

---

### Step 2: Cross-Entropy Loss

**Why this is next:** MSE works for regression but is suboptimal for classification. He's been using MSE for XOR classification — it works, but cross-entropy is what real classifiers use, and he needs it before moving forward.

**What to teach:**
1. **Why MSE is slow for classification.** When the prediction is very wrong (say predicted 0.99, actual 0), MSE gives a gradient that's proportional to (0.99 - 0). But the sigmoid output at that point has a tiny derivative, so the update is small despite the prediction being very wrong. Cross-entropy fixes this.
2. **Binary cross-entropy formula:** `L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`. Walk through with concrete numbers:
   - If y=1 and ŷ=0.99: L = -log(0.99) ≈ 0.01 (low loss, good)
   - If y=1 and ŷ=0.01: L = -log(0.01) ≈ 4.6 (high loss, bad)
   - If y=0 and ŷ=0.01: L = -log(0.99) ≈ 0.01 (low loss, good)
3. **The gradient is simpler and stronger:** For sigmoid + cross-entropy, the gradient simplifies to just `(ŷ - y)` — no sigmoid derivative multiplier choking the gradient.
4. **PyTorch:** `nn.BCELoss()` (expects sigmoid output) vs `nn.BCEWithLogitsLoss()` (applies sigmoid internally, more numerically stable).

**What to build:**
- `stage4_crossentropy_numpy.py` — Implement BCE loss in NumPy. Compute it for his XOR predictions. Compare the gradient with MSE gradient side by side (print both).
- `stage4_crossentropy_torch.py` — Retrain his XOR model using `nn.BCELoss()` instead of `nn.MSELoss()`. Plot loss curves of MSE vs BCE training on the same graph. Observe faster convergence with BCE.

**Suggested filename:** `Stage 4/stage4_crossentropy_numpy.py`, `Stage 4/stage4_crossentropy_torch.py`

---

### Step 3: Adam and AdamW Optimizers

**Why this is next:** He's only used vanilla SGD. Adam is the default optimizer in most modern deep learning. He needs to understand why before he encounters it everywhere in transformer code.

**What to teach:**
1. **The problem with vanilla SGD.** Every parameter gets the same learning rate. But some parameters need big updates (they're far from optimal) and others need small ones (they're close). SGD treats them all the same.
2. **Momentum** (build-up concept first): Like a ball rolling downhill — it accumulates velocity. If the gradient keeps pointing the same direction, momentum makes updates bigger. If the gradient oscillates, momentum smooths it out. Formula: `v = β·v + (1-β)·gradient`, then `param -= lr · v`.
3. **Adam = Momentum + Adaptive learning rates.** It tracks two things per parameter:
   - `m` (first moment / mean of gradients) — like momentum
   - `v` (second moment / mean of squared gradients) — measures how noisy/variable the gradient is
   - Parameters with noisy gradients get smaller effective learning rates
4. **AdamW** = Adam but with weight decay done correctly (decoupled from the gradient). This is what transformers actually use. The difference is subtle — just mention it exists and is preferred.
5. **In PyTorch:** `torch.optim.Adam(model.parameters(), lr=0.001)` and `torch.optim.AdamW(model.parameters(), lr=0.001)`.

**What to build:**
- `stage4_optimizers_compare.py` — Train the XOR model 3 times: once with SGD, once with Adam, once with AdamW. Use the SAME initial weights (set `torch.manual_seed(42)` before each). Plot all 3 loss curves on one graph. Observe Adam converges much faster than SGD on this problem.

**Suggested filename:** `Stage 4/stage4_optimizers_compare.py`

---

### Step 4: Learning Rate Schedules

**Why this is next:** He's been using a fixed learning rate. In practice, you want a large learning rate early (explore fast) and a small one later (fine-tune). This is a short step.

**What to teach:**
1. **Why constant LR is suboptimal.** Too high → overshoots. Too low → too slow. Ideal: start high, decrease over time.
2. **Step decay:** Cut the LR by a factor every N epochs. Simple but effective.
3. **Cosine annealing:** LR follows a cosine curve from initial value down to near zero. Smooth and popular.
4. **Warmup:** Start with a tiny LR and increase it for the first few epochs, then decay. Prevents early instability. Used in transformer training.
5. **In PyTorch:** `torch.optim.lr_scheduler.StepLR`, `CosineAnnealingLR`. Show how to add `scheduler.step()` after `optimizer.step()` in the training loop.

**What to build:**
- `stage4_lr_schedules.py` — Plot 3 LR schedules over 1000 epochs (step decay, cosine, warmup+cosine) on one graph so he can visualize how they differ. Then train the XOR model with constant LR vs cosine annealing and compare loss curves.

**Suggested filename:** `Stage 4/stage4_lr_schedules.py`

---

### Step 5: Overfitting and Underfitting

**Why this is next:** Before learning regularization techniques (the fix), he needs to understand the problem they solve. His XOR dataset is too small to demonstrate overfitting, so this step uses a slightly larger synthetic dataset.

**What to teach:**
1. **Training data vs test data.** The model's job isn't to memorize the training data — it's to learn the underlying pattern and generalize to new data it hasn't seen.
2. **Underfitting:** Model is too simple to capture the pattern. High training loss AND high test loss. Like using a straight line to fit a curve.
3. **Overfitting:** Model memorizes the training data but fails on new data. Low training loss BUT high test loss. Like memorizing answers to practice questions but failing the real exam.
4. **The bias-variance tradeoff** — simple explanation: simple model = high bias (misses the pattern), complex model = high variance (memorizes noise).
5. **Train/test split:** Why you always evaluate on data the model hasn't seen during training. `sklearn.model_selection.train_test_split` or just manual splitting.

**What to build:**
- `stage4_overfit_demo.py` — Generate a synthetic dataset (e.g., noisy sine wave, ~200 points). Split into train/test. Train two models:
  - A tiny model (1 hidden layer, 2 neurons) — underfits
  - A huge model (3 hidden layers, 64 neurons each) — overfits
  - Plot training loss AND test loss for both on the same graph. See the classic overfitting pattern: test loss starts increasing while training loss keeps decreasing.

**Suggested filename:** `Stage 4/stage4_overfit_demo.py`

---

### Step 6: Regularization — Dropout and L2 Weight Decay

**Why this is next:** Now that he's seen overfitting, he learns how to fight it.

**What to teach:**
1. **L2 regularization (weight decay):** Add a penalty for large weights to the loss function: `L_total = L_original + λ·Σ(w²)`. This discourages the model from relying too heavily on any single weight. In PyTorch: just set `weight_decay=0.01` in the optimizer.
2. **Dropout:** During training, randomly set some neuron outputs to zero (e.g., 20% of them each forward pass). This forces the network to not rely on any single neuron — it builds redundancy. During evaluation (`model.eval()`), dropout is turned off and outputs are scaled. In PyTorch: `nn.Dropout(p=0.2)`.
3. **Why dropout works** — analogy: it's like a team where random members call in sick each day. The team has to make sure no single person is a bottleneck. Everyone learns to contribute.

**What to build:**
- `stage4_regularization.py` — Take the overfitting model from Step 5. Add dropout (nn.Dropout(0.2)) between hidden layers. Retrain. Plot train/test loss curves WITH and WITHOUT dropout on the same graph. The overfitting gap should shrink with dropout.
- Also try adding `weight_decay=0.01` to the optimizer and see its effect.

**Suggested filename:** `Stage 4/stage4_regularization.py`

---

### Step 7: Layer Normalization

**Why this is next:** LayerNorm is a building block of every transformer. He needs to understand it before he encounters it inside attention blocks.

**What to teach:**
1. **The problem:** As data passes through many layers, the distribution of values can shift wildly (some layers output huge numbers, others tiny). This makes training unstable and slow. Called "internal covariate shift."
2. **What LayerNorm does:** For each sample, across all features in a layer, it normalizes to mean=0, std=1. Then it scales and shifts with learnable parameters γ (gamma) and β (beta).
   - Formula: `LayerNorm(x) = γ · (x - mean) / (std + ε) + β`
   - ε is a tiny number (like 1e-5) to avoid division by zero
3. **LayerNorm vs BatchNorm:** BatchNorm normalizes across the batch (each feature). LayerNorm normalizes across features (each sample). Transformers use LayerNorm because it works with variable sequence lengths and doesn't depend on batch size.
4. **In PyTorch:** `nn.LayerNorm(hidden_size)`

**What to build:**
- `stage4_layernorm_numpy.py` — Implement LayerNorm by hand in NumPy. Take a small matrix (4 samples × 3 features), compute mean and std for each sample, normalize, then apply gamma and beta. Print before and after.
- `stage4_layernorm_torch.py` — Add `nn.LayerNorm(2)` to his XOR model (after the hidden layer, before activation). Train and compare with/without LayerNorm. On XOR the difference will be small, but the exercise teaches him the API so he's ready when he sees it in transformers.

**Suggested filename:** `Stage 4/stage4_layernorm_numpy.py`, `Stage 4/stage4_layernorm_torch.py`

---

### Step 8: Saving and Loading Models (state_dict)

**Why this is last:** This is a practical skill, not a conceptual one. After training a model, you want to save it so you don't have to retrain every time.

**What to teach:**
1. **What is state_dict?** A Python dictionary mapping each layer name to its weight/bias tensors. Print `model.state_dict()` — he'll see keys like `hidden.weight`, `hidden.bias`, `output.weight`, `output.bias` with their actual tensor values.
2. **Saving:** `torch.save(model.state_dict(), 'model_weights.pth')` — saves just the learned parameters.
3. **Loading:** Create a NEW model instance (same architecture), then `model.load_state_dict(torch.load('model_weights.pth'))`. The new model now has the trained weights without retraining.
4. **Saving the full checkpoint** (model + optimizer + epoch): Useful for resuming training. Save a dict: `{'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}`.
5. **Why save state_dict and not the model itself?** `torch.save(model, ...)` uses pickle, which is brittle — it breaks if you rename the class or move the file. state_dict is just numbers, so it's portable.

**What to build:**
- `stage4_save_load.py` — Train his XOR model for 50k epochs. Save the state_dict. Create a brand new model instance. Load the state_dict into it. Run both models on XOR inputs. Verify they produce identical outputs. Also save a full checkpoint and demonstrate resuming training from epoch 50k to 100k.

**Suggested filename:** `Stage 4/stage4_save_load.py`

---

## Summary of Stage 4 Files

After completing Stage 4, the project structure should look like:

```
C:\PractiseCoding\Python(AI-ML)\
├── Stage 1/  (3 files — perceptron basics)
├── Stage 2/  (6 files — XOR, MLP, backprop)
├── Stage 3/  (6 files — PyTorch training pipeline)
├── Stage 4/
│   ├── stage4_activations_numpy.py        (Step 1)
│   ├── stage4_activations_compare.py      (Step 1)
│   ├── stage4_crossentropy_numpy.py       (Step 2)
│   ├── stage4_crossentropy_torch.py       (Step 2)
│   ├── stage4_optimizers_compare.py       (Step 3)
│   ├── stage4_lr_schedules.py             (Step 4)
│   ├── stage4_overfit_demo.py             (Step 5)
│   ├── stage4_regularization.py           (Step 6)
│   ├── stage4_layernorm_numpy.py          (Step 7)
│   ├── stage4_layernorm_torch.py          (Step 7)
│   └── stage4_save_load.py               (Step 8)
├── main.py
└── .venv/
```

---

## Instructions for the AI Assistant

1. **Go step by step.** Do NOT dump all 8 steps at once. Start with Step 1 (Activation Functions). Explain the concept with concrete numbers. Then guide him to build the file. Check his code. Only then move to Step 2.

2. **Always ground explanations in what he already knows.** He understands sigmoid, MSE, SGD, the XOR MLP, backpropagation as "blame relay," and the full PyTorch training pipeline. Build on this foundation — don't re-explain things he already gets.

3. **NumPy first, PyTorch second.** For conceptual steps (activations, cross-entropy, LayerNorm), implement in NumPy first so he sees the raw math, then show the PyTorch equivalent. For practical steps (optimizers, LR schedules, save/load), PyTorch only is fine.

4. **Use concrete numbers.** Don't just show formulas — plug in actual numbers and walk through the computation. For example, when teaching ReLU: "ReLU(3) = 3, ReLU(-2) = 0, ReLU(0) = 0. The derivative: ReLU'(3) = 1, ReLU'(-2) = 0."

5. **Let him write the code.** After explaining a concept, describe what the code should do, and let him write it. Only show him code if he's stuck. His learning happens when he types it himself.

6. **Verify with plots and prints.** Every step should produce either a plot (matplotlib) or printed output that confirms the concept. He's visual — loss curves, activation plots, and gradient comparisons make things click.

7. **Each step builds to one or two .py files** inside `Stage 4/`. Follow the filenames suggested above so the project stays organized.

8. **His existing model for reference.** When you need to reference or modify his model, here's the architecture from `stage3_final.py`:
   ```python
   class XORModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.hidden = nn.Linear(2, 2)
           self.output = nn.Linear(2, 1)

       def forward(self, x):
           x = self.hidden(x)
           x = torch.sigmoid(x)
           x = self.output(x)
           x = torch.sigmoid(x)
           return x
   ```
   Trained with `nn.MSELoss()`, `torch.optim.SGD(lr=0.1)`, `DataLoader(batch_size=2)`, for 100k epochs.

9. **After all 8 steps are done**, give him a summary of what he's learned and tell him he's ready for Module 1.3+ (if gaps remain) or Module 1.4 (Tokenization & Embeddings) of his ML Infrastructure Engineering roadmap.

---

## What Comes After Stage 4

Once Stage 4 is complete, Samikshan will have covered all of Module 1.2 (Neural Networks & Backpropagation) and most of Module 1.3 (Framework Fluency). The next major milestone is:

- **Module 1.4: Tokenization & Embeddings** — BPE tokenization, word embeddings, positional encodings
- **Module 1.5: Attention & Transformers** — self-attention, multi-head attention, the full transformer architecture

Stage 4 gives him every building block he needs to understand transformers when he gets there. ReLU/GELU are the activations used inside transformers. Cross-entropy is the loss function. Adam/AdamW is the optimizer. LayerNorm is applied before/after each attention block. Dropout is used for regularization. And state_dict is how he'll save his trained models.
