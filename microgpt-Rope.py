"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.
 
@karpathy
Modified: Replaced absolute positional embeddings (wpe) with
          Rotary Position Embeddings (RoPE).
 
RoPE key ideas:
  - No learned position table (wpe removed, fewer parameters)
  - Q and K vectors are rotated by a position-dependent angle before dot-product attention
  - Rotation is applied pair-wise on (x_{2i}, x_{2i+1}) dimensions using sine/cosine
  - Encodes RELATIVE position: the dot product q_m · k_n depends only on (m - n)
  - Generalizes better to sequence lengths not seen during training
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
    def __neg__(self):          return self * -1
    def __radd__(self, other):  return self + other
    def __sub__(self, other):   return self + (-other)
    def __rsub__(self, other):  return other + (-self)
    def __rmul__(self, other):  return self * other
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
# RoPE: precompute cosine/sine tables, then rotate Q or K in-place
# ---------------------------------------------------------------------------
# For each position m and each pair of dimensions (2i, 2i+1), the rotation angle is:
#   theta_i = m / (10000 ^ (2i / head_dim))
# The rotation matrix applied to a 2D pair (x0, x1) is:
#   [ cos(theta)  -sin(theta) ] [ x0 ]   =   [ x0*cos - x1*sin ]
#   [ sin(theta)   cos(theta) ] [ x1 ]       [ x0*sin + x1*cos ]
 
n_layer    = 1
n_embd     = 16
block_size = 16
n_head     = 4
head_dim   = n_embd // n_head   # must be even for RoPE pairs
 
# Precompute cos/sin tables: shape [block_size][head_dim]
# Each position gets head_dim angles (one per dimension), repeated as pairs
rope_cos = []
rope_sin = []
for pos in range(block_size):
    cos_row, sin_row = [], []
    for i in range(head_dim // 2):           # iterate over pairs
        theta = pos / (10000 ** (2 * i / head_dim))
        cos_row += [math.cos(theta), math.cos(theta)]   # same angle for both dims in pair
        sin_row += [math.sin(theta), math.sin(theta)]
    rope_cos.append(cos_row)   # [head_dim] floats
    rope_sin.append(sin_row)
 
def rope(x, pos):
    """
    Apply RoPE rotation to a single vector x of length head_dim at position pos.
    x    : list[Value]  — a single Q or K head vector
    pos  : int          — the absolute position index in the sequence
    Returns a new list[Value] of the same length, rotated.
    """
    out = []
    cos_vals = rope_cos[pos]   # precomputed floats, no gradient needed
    sin_vals = rope_sin[pos]
    for i in range(0, head_dim, 2):        # step through pairs (x0, x1)
        x0, x1 = x[i], x[i + 1]
        c0, s0 = cos_vals[i], sin_vals[i]  # same angle shared by the pair
        # Rotated pair:  x0' = x0*cos - x1*sin
        #                x1' = x0*sin + x1*cos
        out.append(x0 * c0 + x1 * (-s0))
        out.append(x0 * s0 + x1 *   c0 )
    return out
 
# ---------------------------------------------------------------------------
# Model parameters  (wpe removed — RoPE needs no learned position table)
# ---------------------------------------------------------------------------
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
 
state_dict = {
    'wte':     matrix(vocab_size, n_embd),   # token embeddings only (no wpe)
    'lm_head': matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
 
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}  (wpe removed, saved {block_size * n_embd} params)")
 
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
# GPT forward pass with RoPE
# ---------------------------------------------------------------------------
def gpt(token_id, pos_id, keys, values):
    # Token embedding only — no additive positional embedding
    x = list(state_dict['wte'][token_id])   # copy so we don't mutate the table
    x = rmsnorm(x)
 
    for li in range(n_layer):
        # 1) Multi-head Attention
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
            # Slice out this head's Q and K, then apply RoPE rotation
            q_h = rope(q[hs:hs + head_dim], pos_id)                          # rotated Q
            k_h = [rope(ki[hs:hs + head_dim], t) for t, ki in enumerate(keys[li])]  # rotated K cache
            v_h = [vi[hs:hs + head_dim] for vi in values[li]]                # V unchanged
 
            attn_logits  = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                            for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out     = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                            for j in range(head_dim)]
            x_attn.extend(head_out)
 
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
 
        # 2) MLP block (unchanged)
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
 
    logits = linear(x, state_dict['lm_head'])
    return logits
 
# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)
v_buf = [0.0] * len(params)
 
num_steps = 1000
for step in range(num_steps):
 
    doc    = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n      = min(block_size, len(tokens) - 1)
 
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits  = gpt(token_id, pos_id, keys, values)
        probs   = softmax(logits)
        loss_t  = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)
 
    loss.backward()
 
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat    = m_buf[i] / (1 - beta1 ** (step + 1))
        v_hat    = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data  -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad   = 0
 
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
 
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
        logits   = gpt(token_id, pos_id, keys, values)
        probs    = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
 
