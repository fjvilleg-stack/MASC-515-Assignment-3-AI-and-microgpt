"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.
 
@karpathy
Modified: Replaced the single MLP block with a Mixture of Experts (MoE) layer.
 
MoE key ideas:
  - Instead of one MLP, there are n_experts independent MLPs ("experts")
  - A small router/gating network scores each expert for the current token
  - Only the top-K experts are activated per token (all others skipped)
  - Each selected expert output is weighted by its router probability and summed
  - An auxiliary load-balancing loss encourages all experts to be used equally
"""
 
import os
import math
import random
random.seed(42)
 
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
 
# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")
 
# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
 
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
 
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
 
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
 
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self):  return Value(math.log(self.data),  (self,), (1 / self.data,))
    def exp(self):  return Value(math.exp(self.data),  (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data),    (self,), (float(self.data > 0),))
    def __neg__(self):             return self * -1
    def __radd__(self, other):     return self + other
    def __sub__(self, other):      return self + (-other)
    def __rsub__(self, other):     return other + (-self)
    def __rmul__(self, other):     return self * other
    def __truediv__(self, other):  return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
 
    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
 
# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
n_layer    = 1
n_embd     = 16
block_size = 16
n_head     = 4
head_dim   = n_embd // n_head
 
# MoE hyperparameters
n_experts           = 4     # total number of expert MLPs available per layer
top_k               = 2     # how many experts are activated per token
moe_aux_loss_weight = 0.01  # weight for the load-balancing auxiliary loss
 
# ---------------------------------------------------------------------------
# Parameter initialization
# ---------------------------------------------------------------------------
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
 
state_dict = {
    'wte':     matrix(vocab_size, n_embd),
    'wpe':     matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}
 
for i in range(n_layer):
    # Attention weights unchanged
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    # Router: projects x -> n_experts logits
    state_dict[f'layer{i}.moe_router'] = matrix(n_experts, n_embd)
    # n_experts independent MLPs
    for e in range(n_experts):
        state_dict[f'layer{i}.expert{e}.fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.expert{e}.fc2'] = matrix(n_embd, 4 * n_embd)
 
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params : {len(params)}")
print(f"num experts: {n_experts}  |  top-k activated: {top_k}")
 
# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
 
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps    = [(val - max_val).exp() for val in logits]
    total   = sum(exps)
    return [e / total for e in exps]
 
def rmsnorm(x):
    ms    = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
 
# ---------------------------------------------------------------------------
# MoE block
# Steps:
#   1. Router logits  = router_W @ x              (n_experts,)
#   2. Gates          = softmax(router_logits)     (n_experts,)
#   3. Pick top-K experts by gate value
#   4. Run only those K expert MLPs on x
#   5. output = sum_k( gate_k * expert_k(x) )
#   6. Aux loss penalises unequal expert usage
# ---------------------------------------------------------------------------
def moe(x, layer_idx):
    # Steps 1 & 2: router scores
    router_logits = linear(x, state_dict[f'layer{layer_idx}.moe_router'])
    gates         = softmax(router_logits)
 
    # Step 3: top-K selection (use .data for comparison, no gradient here)
    top_k_indices = sorted(range(n_experts), key=lambda e: gates[e].data, reverse=True)[:top_k]
 
    # Steps 4 & 5: weighted sum of selected expert outputs
    out = [Value(0.0)] * n_embd
    for e in top_k_indices:
        h = linear(x, state_dict[f'layer{layer_idx}.expert{e}.fc1'])
        h = [hi.relu() for hi in h]
        h = linear(h, state_dict[f'layer{layer_idx}.expert{e}.fc2'])
        out = [oi + gates[e] * hi for oi, hi in zip(out, h)]
 
    # Step 6: auxiliary load-balancing loss
    # Encourages uniform routing: penalises gate mass concentrated on few experts
    aux = Value(0.0)
    for e in range(n_experts):
        indicator = 1.0 if e in top_k_indices else 0.0
        aux = aux + gates[e] * indicator
    aux_loss = aux * n_experts
 
    return out, aux_loss
 
# ---------------------------------------------------------------------------
# GPT forward pass
# ---------------------------------------------------------------------------
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
 
    total_aux_loss = Value(0.0)
 
    for li in range(n_layer):
        # 1) Multi-head Attention (unchanged)
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs  = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h = [ki[hs:hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs + head_dim] for vi in values[li]]
            attn_logits  = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                            for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out     = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                            for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
 
        # 2) MoE block replaces single MLP
        x_residual = x
        x = rmsnorm(x)
        x, aux_loss = moe(x, li)
        x = [a + b for a, b in zip(x, x_residual)]
        total_aux_loss = total_aux_loss + aux_loss
 
    logits = linear(x, state_dict['lm_head'])
    return logits, total_aux_loss
 
# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)
v_buf = [0.0] * len(params)
 
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
num_steps = 1000
for step in range(num_steps):
 
    doc    = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n      = min(block_size, len(tokens) - 1)
 
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses, aux_losses = [], []
 
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits, aux_loss = gpt(token_id, pos_id, keys, values)
        probs  = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
        aux_losses.append(aux_loss)
 
    main_loss     = (1 / n) * sum(losses)
    aux_loss_mean = (1 / n) * sum(aux_losses)
    loss          = main_loss + moe_aux_loss_weight * aux_loss_mean
 
    loss.backward()
 
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat    = m_buf[i] / (1 - beta1 ** (step + 1))
        v_hat    = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data  -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad   = 0
 
    print(f"step {step+1:4d} / {num_steps:4d} | loss {main_loss.data:.4f} | aux {aux_loss_mean.data:.4f}", end='\r')
 
# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample   = []
    for pos_id in range(block_size):
        logits, _ = gpt(token_id, pos_id, keys, values)
        probs     = softmax([l / temperature for l in logits])
        token_id  = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
