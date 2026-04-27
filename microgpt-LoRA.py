"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.
 
@karpathy
Modified: Added LoRA (Low-Rank Adaptation) for attention weight matrices.
  - Original weights (wq, wk, wv, wo) are FROZEN
  - LoRA adds two small trainable matrices A (n_embd x r) and B (r x n_embd) per attention weight
  - Effective weight = W + B @ A, with rank r << n_embd
  - Only LoRA params are updated during training, drastically reducing trainable parameters
"""
 
import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos
 
# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
 
# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")
 
# Let there be Autograd to recursively apply the chain rule through a computation graph
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
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
 
    def backward(self):
        topo = []
        visited = set()
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
# LoRA hyperparameters
# ---------------------------------------------------------------------------
lora_rank = 4       # rank r: the bottleneck dimension of LoRA (smaller = fewer params)
lora_alpha = 8      # scaling factor: effective update = (lora_alpha / lora_rank) * B @ A
lora_scale = lora_alpha / lora_rank
 
# ---------------------------------------------------------------------------
# Initialize the parameters, to store the knowledge of the model
# ---------------------------------------------------------------------------
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head
 
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
zeros  = lambda nout, nin:           [[Value(0.0)                   for _ in range(nin)] for _ in range(nout)]
 
# Build the base (frozen) state dict — same as original
state_dict = {
    'wte':     matrix(vocab_size, n_embd),
    'wpe':     matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
 
# LoRA matrices — only for the four attention projections per layer
# A is (r x n_embd): initialized with small random values  → captures input directions
# B is (n_embd x r): initialized to ZERO                   → ensures delta-W = 0 at start
lora_dict = {}
for i in range(n_layer):
    for proj in ('wq', 'wk', 'wv', 'wo'):
        lora_dict[f'layer{i}.{proj}.lora_A'] = matrix(lora_rank, n_embd, std=0.01)  # (r x d)
        lora_dict[f'layer{i}.{proj}.lora_B'] = zeros(n_embd, lora_rank)              # (d x r) — zero init
 
# Only LoRA params are trained; base weights stay frozen
lora_params = [p for mat in lora_dict.values() for row in mat for p in row]
 
# Count for reference
base_params = sum(len(row) for mat in state_dict.values() for row in mat)
print(f"base (frozen) params : {base_params}")
print(f"LoRA (trainable) params: {len(lora_params)}  (rank={lora_rank})")
 
# ---------------------------------------------------------------------------
# Helper: linear projection with optional LoRA delta
# lora_A shape: (r x in_dim),  lora_B shape: (out_dim x r)
# delta_W = lora_scale * (lora_B @ lora_A),  applied as delta_W @ x
# ---------------------------------------------------------------------------
def linear(x, w, lora_A=None, lora_B=None):
    # Base projection: W @ x
    out = [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
    if lora_A is not None and lora_B is not None:
        # LoRA path: (lora_B @ (lora_A @ x)) * scale
        ax = [sum(ai * xi for ai, xi in zip(a_row, x)) for a_row in lora_A]  # (r,)
        bax = [sum(bi * axi for bi, axi in zip(b_row, ax)) for b_row in lora_B]  # (out_dim,)
        out = [oi + lora_scale * di for oi, di in zip(out, bax)]
    return out
 
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
 
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
 
# ---------------------------------------------------------------------------
# GPT forward pass — attention projections now use LoRA
# ---------------------------------------------------------------------------
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
 
    for li in range(n_layer):
        # 1) Multi-head Attention block — with LoRA on all four projections
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'],
                   lora_dict[f'layer{li}.wq.lora_A'], lora_dict[f'layer{li}.wq.lora_B'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'],
                   lora_dict[f'layer{li}.wk.lora_A'], lora_dict[f'layer{li}.wk.lora_B'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'],
                   lora_dict[f'layer{li}.wv.lora_A'], lora_dict[f'layer{li}.wv.lora_B'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'],
                   lora_dict[f'layer{li}.wo.lora_A'], lora_dict[f'layer{li}.wo.lora_B'])
        x = [a + b for a, b in zip(x, x_residual)]
 
        # 2) MLP block — no LoRA (same as original)
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
 
    logits = linear(x, state_dict['lm_head'])
    return logits
 
# ---------------------------------------------------------------------------
# Adam optimizer — operates ONLY on lora_params (base weights are frozen)
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(lora_params)
v = [0.0] * len(lora_params)
 
num_steps = 1000
for step in range(num_steps):
 
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
 
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)
 
    loss.backward()
 
    # Update only LoRA params — base weights accumulate no grad and are never stepped
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(lora_params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
 
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
 
# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
