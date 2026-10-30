import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =====================================================================
# WHAT AN OPTIMIZER ACTUALLY DOES
# =====================================================================
# The training loop is always: forward pass -> loss -> backward() -> ??? -> repeat.
# backward() computes the gradient (how much each weight is "to blame" for the loss).
# The optimizer is the "???" -- it decides HOW to turn that gradient into an actual
# change to the weight. Three different answers to that question are compared below.
#
# ---- SGD (plain) ----
# Formula:  w_new = w_old - lr * gradient
# That's it. Look at today's gradient, take a step opposite to it, forget it
# immediately. This is the exact manual update you wrote by hand back in Stage 2
# (param_new = param_old - learning_rate * gradient).
# Limitation: every parameter uses the same lr, and a tiny gradient produces a
# tiny step even in a flat region of the loss surface -- it can crawl for a long
# time before escaping (this is exactly what you saw in the plot: SGD sits flat
# at loss=0.25 until ~epoch 4300, because the gradient there is small).
#
# ---- Momentum (a stepping stone to Adam, not used directly in this file) ----
# Formula:   v_t = beta * v_(t-1) + (1 - beta) * g_t      (blend today's gradient
#                                                           into a running memory)
#            w_new = w_old - lr * v_t
# Instead of reacting to only today's gradient, momentum remembers a fraction of
# every past gradient too. If the gradient keeps pointing the same direction
# (like a ball rolling downhill), v_t builds up and the steps get effectively
# bigger over time. PyTorch's SGD can do this too: torch.optim.SGD(params,
# lr=..., momentum=0.9) -- we're using momentum=0 (the default) below, i.e. plain
# SGD with no memory, so the plateau problem above shows up in full force.
#
# ---- Adam (Adaptive Moment Estimation) ----
# Adam keeps TWO running memories per parameter instead of momentum's one:
#   m_t = beta1 * m_(t-1) + (1 - beta1) * g_t        (memory of DIRECTION)
#   v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2       (memory of SIZE, squared)
# Both are corrected for being "cold" early on (bias correction):
#   m_hat = m_t / (1 - beta1^t)
#   v_hat = v_t / (1 - beta2^t)
# Then the update:
#   w_new = w_old - lr * m_hat / (sqrt(v_hat) + eps)
# Dividing by sqrt(v_hat) is the key trick: it cancels out the gradient's own
# typical size, so the step becomes roughly "lr, pointed in the gradient's
# direction" REGARDLESS of whether the raw gradient was 0.001 or 10. That's why
# Adam escapes the flat plateau almost instantly below while SGD crawls for
# ~4000 epochs -- a small gradient still produces a full lr-sized Adam step.
#
# ---- AdamW ----
# Identical to Adam, except weight decay (shrinking weights slightly every step
# to discourage them from growing huge) is applied SEPARATELY instead of being
# mixed into the gradient before the m/v normalization above:
#   w_new = w_old - lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * w_old )
# If you instead added weight_decay*w into the gradient itself (the old "Adam +
# L2" approach), it would get divided by sqrt(v_hat) along with everything else
# and its effect would get watered down. Keeping it separate ("decoupled") means
# it actually does what it's supposed to. Note: torch.optim.AdamW defaults to
# weight_decay=0.01 even though we don't set it explicitly below.
# =====================================================================

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


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


# ---- Why train() takes a FUNCTION that builds an optimizer, instead of just
#      taking an optimizer directly ----
# An optimizer has to be built from a specific model's parameters -- you can't
# create one before the model exists. But train() creates a brand-new XORModel
# every time it's called (so each of the 3 runs below starts from the exact
# same random weights, for a fair comparison). That means the optimizer can't
# be built until AFTER train() has already created its own fresh model.
# So instead of passing an already-built optimizer, we pass a small function
# that knows HOW to build one once it's given a model's parameters. train()
# calls that function itself, at the right moment, using its own fresh model.
def train(optimizer_fn, epochs=5000):
    torch.manual_seed(1)  # SAME starting weights every time — fair comparison
    model = XORModel()
    loss_fn = nn.MSELoss()
    optimizer = optimizer_fn(model.parameters())  # optimizer_fn is called HERE,
                                                    # only now that "model" exists

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()          # computes gradient = dL/dw for every parameter
        optimizer.step()         # THIS is where the formula above actually runs --
                                  # SGD/Adam/AdamW each implement their own version
                                  # of "w_new = w_old - (something using the gradient)"
                                  # internally; we never see the formula, but it's
                                  # exactly what's written in the comment block above.
        losses.append(loss.item())

    return losses


# These are the three "optimizer builder" functions mentioned above. Each one
# takes params (the model's weights and biases) and returns a ready-to-use
# optimizer wrapping them. This does exactly what the lambda one-liners used
# to do -- it's just spelled out as a normal function instead, so it's easier
# to read step by step.

# w_new = w_old - lr * gradient  (no memory, no per-parameter adaptation)
def build_sgd(params):
    return torch.optim.SGD(params, lr=0.5)

# w_new = w_old - lr * m_hat / (sqrt(v_hat) + eps)  (direction memory m,
# size memory v, both bias-corrected -- see the big comment block up top)
def build_adam(params):
    return torch.optim.Adam(params, lr=0.01)

# same as Adam, but weight_decay (default 0.01 here) is applied as its own
# separate shrink step instead of being folded into the gradient first
def build_adamw(params):
    return torch.optim.AdamW(params, lr=0.01)


sgd_losses = train(build_sgd)
adam_losses = train(build_adam)
adamw_losses = train(build_adamw)

print("Final loss after 5000 epochs:")
print("SGD:  ", sgd_losses[-1])
print("Adam: ", adam_losses[-1])
print("AdamW:", adamw_losses[-1])
print()
print("SGD is still visibly making progress but hasn't fully converged in 5000 steps.")
print("Adam and AdamW reach near-zero loss much faster -- that's the adaptive")
print("per-parameter learning rate + momentum combo doing its job.")

plt.plot(sgd_losses, label="SGD (lr=0.5)")
plt.plot(adam_losses, label="Adam (lr=0.01)")
plt.plot(adamw_losses, label="AdamW (lr=0.01)", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SGD vs Adam vs AdamW on XOR (identical starting weights)")
plt.legend()
plt.show()

# Watch how much faster Adam/AdamW drop in the first few hundred epochs compared to SGD.
# That's the adaptive per-parameter learning rate + momentum doing their job.
