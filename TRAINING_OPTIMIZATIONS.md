# Training Optimizations Notes

How we achieved **16x speedup** (1.6s/step → 0.1s/step) on GPT-2 Small training.

---

## Overview

| Optimization | Speed Gain | Memory Impact |
|--------------|------------|---------------|
| Pre-tokenization | ~2x | Saves CPU |
| RAM preload | ~2-3x | +5.6 GB RAM |
| torch.compile | ~1.5-2x | +slight (kernels) |
| AMP (Mixed Precision) | ~1.5-2x | -50% activations |
| Vectorized batching | ~1.2x | Negligible |
| Flash Attention | ~2x | -90% attention memory |
| **Combined** | **~7x** | Net savings |

---

## Batch Fundamentals

Before diving into optimizations, let's understand what a batch is — this concept is central to all the optimizations below.

---

### What is a Batch?

A **batch** is a group of training samples processed together in one forward/backward pass.

```
Your dataset: 5.5 million sequences (each 512 tokens)

Without batching (batch_size=1):
┌─────────┐     ┌─────────┐     ┌─────────┐
│ seq 1   │ →   │ seq 2   │ →   │ seq 3   │ → ... → 5.5M updates
│ forward │     │ forward │     │ forward │
│ backward│     │ backward│     │ backward│
│ update  │     │ update  │     │ update  │
└─────────┘     └─────────┘     └─────────┘
   slow            slow            slow

With batching (batch_size=64):
┌───────────────────────────────┐     ┌───────────────────────────────┐
│ seq 1, seq 2, ... seq 64      │ →   │ seq 65, seq 66, ... seq 128   │ → ...
│ forward (all 64 at once)      │     │ forward (all 64 at once)      │
│ backward (average gradient)   │     │ backward (average gradient)   │
│ update (once)                 │     │ update (once)                 │
└───────────────────────────────┘     └───────────────────────────────┘
        ~86,000 updates total (5.5M / 64)
```

---

### Why Not Process One Sample at a Time?

**Problem 1: GPU underutilization**
```
GPU with batch_size=1:
┌──────────────────────────────────────────────────────┐
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│  5%                                                  │
│  Used                        95% Idle                │
└──────────────────────────────────────────────────────┘

GPU with batch_size=64:
┌──────────────────────────────────────────────────────┐
│ ████████████████████████████████████████████████░░░ │
│                    90% Used              10% Idle    │
└──────────────────────────────────────────────────────┘
```

GPUs have thousands of cores. Processing 1 sample leaves most cores idle.

**Problem 2: Noisy gradients**
```
Single sample gradient:
  "This sample says: move weights LEFT a lot!"

Next sample gradient:
  "This sample says: move weights RIGHT a lot!"

Result: Weights bounce around chaotically
```

---

### Why Not Process Entire Dataset at Once?

**Problem: Memory**
```
Your dataset: 5.5M sequences × 512 tokens × 2 bytes = ~5.6 GB just for input
Plus activations: ~50-100 GB
Plus gradients: another 50-100 GB

Total needed: Hundreds of GB → Doesn't fit in any GPU!
```

---

### The Batch Tradeoff

| Batch Size | Memory | Speed | Gradient Quality |
|------------|--------|-------|------------------|
| 1 | Minimal | Very slow | Very noisy |
| 16 | Low | Slow | Noisy |
| 64 | Medium | Fast | Good |
| 256 | High | Faster | Smoother |
| 1024+ | Very high | Fastest | Very smooth (may need LR tuning) |

---

### What Happens Inside a Batch

```python
batch_size = 64
seq_len = 512

# Your batch shape:
batch = tensor of shape [64, 512]
#                        │    │
#                        │    └── 512 tokens per sequence
#                        └── 64 sequences

# Visual:
┌─────────────────────────────────────────────────────┐
│ Sequence 0:  [token₀, token₁, token₂, ... token₅₁₁] │
│ Sequence 1:  [token₀, token₁, token₂, ... token₅₁₁] │
│ Sequence 2:  [token₀, token₁, token₂, ... token₅₁₁] │
│ ...                                                  │
│ Sequence 63: [token₀, token₁, token₂, ... token₅₁₁] │
└─────────────────────────────────────────────────────┘
```

Forward pass processes all 64 sequences in parallel. Loss is averaged across all 64. Gradient is the average direction to improve on all 64.

---

### Gradient Averaging (The Key Insight)

```
Batch of 4 samples:

Sample 1 gradient: "move weight +0.1"
Sample 2 gradient: "move weight -0.3"
Sample 3 gradient: "move weight +0.2"
Sample 4 gradient: "move weight +0.4"
─────────────────────────────────────
Average gradient:  "move weight +0.1"  ← More stable!

This averaged gradient represents the "consensus" direction
that improves performance across all 4 samples.
```

---

### Memory Usage Per Batch Size

For our 134M parameter GPT-2 Small:

| Batch Size | Input Tokens | Activations | Total GPU RAM |
|------------|--------------|-------------|---------------|
| 16 | 8K | ~2 GB | ~4 GB |
| 32 | 16K | ~4 GB | ~6 GB |
| 64 | 32K | ~8 GB | ~10 GB |
| 128 | 64K | ~16 GB | ~18 GB |

**Why activations grow:** Each layer stores intermediate values for the backward pass.

```
Forward pass stores (for backward):
┌─────────────────────────────────────────────┐
│ Layer 1 output: [64, 512, 768] = 25 MB      │
│ Layer 2 output: [64, 512, 768] = 25 MB      │
│ ...                                          │
│ Layer 12 output: [64, 512, 768] = 25 MB     │
│ Attention scores: 12 layers × ~50 MB each   │
│ ────────────────────────────────────────    │
│ Total activations: ~1 GB per batch          │
└─────────────────────────────────────────────┘

Double the batch size → Double the activations
```

---

### Our Training Configuration

```
batch_size = 64
sequences per epoch = 5.5M
steps per epoch = 5.5M / 64 ≈ 86,000 steps

Each step:
- Processes 64 × 512 = 32,768 tokens
- Computes 1 averaged gradient
- Updates weights once
```

---

## 1. Pre-Tokenization (Before Training)

**What it is:** Convert text → token IDs once, save as binary file

### Before (Slow)
```python
# Tokenize every batch during training
for text in dataset:
    tokens = tokenizer.encode(text)  # SLOW - runs every epoch!
```

### After (Fast)
```python
# Pre-tokenize once locally, then load binary
tokens = np.memmap('tokens.bin', dtype=np.int16)  # Already done!
```

### Impact

| Metric | Before | After |
|--------|--------|-------|
| CPU overhead | High (tokenization) | Near zero |
| Disk format | Raw text | Binary int16/int32 |
| File size | 12 GB text | ~5.6 GB binary |

---

## 2. RAM Preload vs Memory-Mapping (mmap)

### Memory-mapping (slow for random access)
```python
data = np.memmap('tokens.bin', dtype=np.int16, mode='r')
# OS fetches pages from disk on demand - slow for random batch sampling
```

### RAM Preload (fast)
```python
data = np.fromfile('tokens.bin', dtype=np.int16)
# Everything in RAM - instant random access
```

### Comparison

| Metric | mmap | RAM Preload |
|--------|------|-------------|
| RAM usage | ~0 (OS page cache) | Full dataset size (~5.6 GB) |
| Random access speed | Slow (disk I/O) | Fast (memory) |
| Best for | Huge datasets (>RAM) | Fits in RAM |

**Our case:** 12GB raw → ~5.6GB tokens fits in Colab's 80GB RAM (A100). RAM preload wins.

---

## 3. torch.compile (PyTorch 2.0+)

**What it is:** JIT-compiles your model into optimized CUDA kernels

```python
model = TransformerLanguageModel(...)
model = torch.compile(model)  # One line = 1.5-2x speedup
```

### What is a CUDA Kernel?

A "kernel" is a function that runs on the GPU. Each PyTorch operation launches a kernel:

```python
# Each line = separate kernel launch = GPU overhead
x = a + b         # Kernel 1: addition
x = x * c         # Kernel 2: multiplication
x = torch.relu(x) # Kernel 3: ReLU
x = x / d         # Kernel 4: division
```

**Problem:** Each kernel launch has overhead (~5-20 microseconds). With thousands of operations per forward pass, this adds up.

### How torch.compile Fixes This

**Before (Eager Mode):**
```
Python → Op 1 → Launch Kernel 1 → Wait
       → Op 2 → Launch Kernel 2 → Wait
       → Op 3 → Launch Kernel 3 → Wait
       → Op 4 → Launch Kernel 4 → Wait
```

**After (Compiled):**
```
Python → Compiler analyzes graph → Generates FUSED kernel
       → Launch Single Kernel → Done
```

### Concrete Example: Layer Normalization

**Eager mode (4 kernels):**
```python
def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)    # Kernel 1
    var = x.var(dim=-1, keepdim=True)      # Kernel 2
    x = (x - mean) / torch.sqrt(var + eps) # Kernel 3
    return x * weight + bias               # Kernel 4
```

**Compiled (1 fused kernel):**
```python
@torch.compile
def layer_norm(x, weight, bias, eps=1e-5):
    # Compiler fuses ALL of this into ONE kernel
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    return x * weight + bias
```

### Visual Comparison

```
BEFORE (Eager):
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Kernel 1│ → │ Kernel 2│ → │ Kernel 3│ → │ Kernel 4│
│  mean   │   │   var   │   │normalize│   │ scale   │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
     ↓             ↓             ↓             ↓
   [RAM]    →   [RAM]    →   [RAM]    →   [RAM]

   4 kernel launches + 4 memory round-trips

AFTER (Compiled):
┌─────────────────────────────────────────────────┐
│            Single Fused Kernel                  │
│  mean → var → normalize → scale (all in GPU)   │
└─────────────────────────────────────────────────┘
     ↓
   [RAM]

   1 kernel launch + 1 memory round-trip
```

### Why Memory Round-Trips Matter

GPU memory bandwidth is the bottleneck, not compute:

```
Operation: x = a + b + c + d

Eager (naive):
  Load a, b → Add → Store temp1     (2 reads, 1 write)
  Load temp1, c → Add → Store temp2 (2 reads, 1 write)
  Load temp2, d → Add → Store x     (2 reads, 1 write)
  Total: 6 reads + 3 writes = 9 memory ops

Fused:
  Load a, b, c, d → Add all → Store x
  Total: 4 reads + 1 write = 5 memory ops

  ~45% less memory traffic!
```

### What the Compiler Generates

torch.compile uses **Triton** to generate optimized GPU code:

```python
# Your Python code
@torch.compile
def gelu(x):
    return x * 0.5 * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))
```

**Compiler generates (simplified):**
```c
// Auto-generated Triton kernel
@triton.jit
def fused_gelu_kernel(x_ptr, out_ptr, n):
    idx = triton.program_id(0) * BLOCK + triton.arange(BLOCK)
    x = triton.load(x_ptr + idx)

    # All math fused into single pass
    x3 = x * x * x
    inner = 0.7978845608 * (x + 0.044715 * x3)
    tanh_val = triton.tanh(inner)
    result = x * 0.5 * (1 + tanh_val)

    triton.store(out_ptr + idx, result)
```

### Impact on Transformer Forward Pass

| Component | Eager Kernels | Compiled Kernels |
|-----------|---------------|------------------|
| Embedding lookup | 2 | 1 |
| Layer Norm (×24) | 4 × 24 = 96 | 1 × 24 = 24 |
| Attention (×12) | ~20 × 12 = 240 | ~5 × 12 = 60 |
| FFN (×12) | ~8 × 12 = 96 | ~2 × 12 = 24 |
| **Total** | **~434 kernels** | **~109 kernels** |

**75% fewer kernel launches = significant speedup**

### The Compilation Process

```python
model = torch.compile(model)

# First forward pass:
output = model(input)  # SLOW (~30-60 seconds)
# ↳ Traces your model
# ↳ Builds computation graph
# ↳ Optimizes graph (fusion, etc.)
# ↳ Generates CUDA/Triton kernels
# ↳ Compiles kernels to GPU machine code

# Subsequent passes:
output = model(input)  # FAST (uses cached kernels)
```

### When torch.compile Helps Most

| Scenario | Speedup |
|----------|---------|
| Many small ops (LayerNorm, GELU, etc.) | 2-3x |
| Memory-bound operations | 1.5-2x |
| Large batch sizes | 1.5-2x |
| Small models on fast GPUs | 1.2-1.5x |
| Already using Flash Attention | 1.1-1.3x |

---

## 4. AMP (Automatic Mixed Precision)

**What it is:** Use FP16 for most operations, FP32 only where needed

---

### What is FP16 and FP32?

**FP = Floating Point** - how computers represent decimal numbers.

```
FP32 (32-bit float, "single precision"):
┌─────┬──────────┬───────────────────────┐
│sign │ exponent │       mantissa        │
│1 bit│  8 bits  │       23 bits         │
└─────┴──────────┴───────────────────────┘
Total: 32 bits = 4 bytes per number

FP16 (16-bit float, "half precision"):
┌─────┬──────────┬───────────┐
│sign │ exponent │ mantissa  │
│1 bit│  5 bits  │  10 bits  │
└─────┴──────────┴───────────┘
Total: 16 bits = 2 bytes per number
```

### FP16 vs FP32 Comparison

| Property | FP32 | FP16 |
|----------|------|------|
| Bytes per number | 4 | 2 |
| Max value | ~3.4 × 10³⁸ | ~65,504 |
| Min positive value | ~1.2 × 10⁻³⁸ | ~6.0 × 10⁻⁸ |
| Precision (decimal digits) | ~7 | ~3 |
| Memory for 134M params | ~536 MB | ~268 MB |

**Key insight:** FP16 uses half the memory and is 2x faster on modern GPUs (Tensor Cores), but has less precision and a smaller range.

---

### The Underflow Problem

**Underflow** = when a number is too small to represent, it becomes zero.

```
FP16 minimum positive value: 0.00006 (approximately)

Gradient example:
  Actual gradient:     0.00001  (valid, important for learning)
  FP16 representation: 0.00000  (UNDERFLOW - becomes zero!)

Result: Model stops learning because gradients are "lost"
```

**Visual:**
```
FP32 number line (zoomed in near zero):
... -0.0001 ... -0.00001 ... 0 ... 0.00001 ... 0.0001 ...
                              ↑
                    All these values exist

FP16 number line (zoomed in near zero):
... -0.0001 ...    [DEAD ZONE]    0    [DEAD ZONE]    ... 0.0001 ...
                        ↑                   ↑
              Numbers here become 0    Numbers here become 0
```

Small gradients are common in deep networks (especially early layers), so underflow is a real problem.

---

### How GradScaler Solves This

**GradScaler** multiplies the loss by a large number before backprop, then divides afterward.

```
Without scaling:
  loss = 0.5
  gradient = 0.00001  → FP16: 0 (UNDERFLOW!)

With scaling (scale = 1024):
  scaled_loss = 0.5 × 1024 = 512
  scaled_gradient = 0.00001 × 1024 = 0.01024  → FP16: 0.01024 ✓

  After backward:
  actual_gradient = 0.01024 / 1024 = 0.00001  ✓ (recovered!)
```

**Visual flow:**
```
                    FORWARD PASS (FP16)
                           │
                           ▼
                    ┌─────────────┐
                    │ loss = 0.5  │
                    └─────────────┘
                           │
                           ▼ × scale (1024)
                    ┌─────────────────┐
                    │ scaled_loss=512 │
                    └─────────────────┘
                           │
                    BACKWARD PASS (FP16)
                           │
                           ▼
                    ┌─────────────────────┐
                    │ scaled_grad = 0.01  │  (survives in FP16!)
                    └─────────────────────┘
                           │
                           ▼ ÷ scale (1024)
                    ┌─────────────────────┐
                    │ actual_grad=0.00001 │  (recovered!)
                    └─────────────────────┘
                           │
                    WEIGHT UPDATE (FP32)
                           │
                           ▼
                    weights -= lr × actual_grad
```

---

### GradScaler: Dynamic Scale Adjustment

The scaler automatically adjusts the scale factor during training:

```python
scaler = GradScaler(init_scale=65536)  # Start with large scale

# During training, scaler monitors for:
# 1. INF/NaN gradients → scale was too high → reduce scale
# 2. No INF/NaN for N steps → scale might be too low → increase scale
```

**How it adapts:**
```
Step 1-100:   scale = 65536  (no issues)
Step 101:     scale = 65536  → INF detected! Skip update, scale = 32768
Step 102-200: scale = 32768  (stable)
Step 201:     scale = 32768  → No issues for 100 steps → scale = 65536
...and so on
```

**Visual:**
```
Scale factor over training:

65536 |    ████                    ████████
      |    █  █                    █
32768 |████   ████████████████████
      |
16384 |
      +----------------------------------------→ Steps
           ↑        ↑
        INF found   Stable, increase scale
        (reduce)
```

---

### Complete AMP Code Explained

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler with starting scale
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # ══════════════════════════════════════════════
    # STEP 1: Forward pass in FP16
    # ══════════════════════════════════════════════
    with autocast():
        # autocast() automatically:
        # - Converts inputs to FP16
        # - Runs matmuls, convs in FP16 (fast!)
        # - Keeps some ops in FP32 (softmax, loss, layer norm)
        output = model(batch)
        loss = criterion(output, target)

    # ══════════════════════════════════════════════
    # STEP 2: Scale loss and backward pass
    # ══════════════════════════════════════════════
    scaler.scale(loss).backward()
    # Equivalent to: (loss * scale).backward()
    # Gradients are now scaled up → no underflow!

    # ══════════════════════════════════════════════
    # STEP 3: Unscale gradients and update weights
    # ══════════════════════════════════════════════
    scaler.step(optimizer)
    # This does:
    # 1. Unscale gradients (divide by scale)
    # 2. Check for INF/NaN
    # 3. If OK: optimizer.step() (update weights in FP32)
    # 4. If INF/NaN: skip this update

    # ══════════════════════════════════════════════
    # STEP 4: Adjust scale for next iteration
    # ══════════════════════════════════════════════
    scaler.update()
    # Increase scale if no INF/NaN for a while
    # Decrease scale if INF/NaN detected
```

---

### What autocast() Does Automatically

| Operation | Precision Used | Why |
|-----------|---------------|-----|
| Matrix multiply (Linear, Attention) | FP16 | Fast on Tensor Cores |
| Convolution | FP16 | Fast on Tensor Cores |
| GELU, ReLU | FP16 | Simple ops, FP16 is fine |
| Softmax | FP32 | Needs precision for exp() |
| Layer Norm | FP32 | Needs precision for variance |
| Loss calculation | FP32 | Needs precision |
| Weight updates | FP32 | Accumulation needs precision |

---

### Memory Comparison

```
Without AMP (all FP32):
┌────────────────────────────────────────────┐
│ Model weights:     134M × 4 bytes = 536 MB │
│ Gradients:         134M × 4 bytes = 536 MB │
│ Optimizer states:  134M × 8 bytes = 1.07 GB│ (Adam has 2 states)
│ Activations:       ~2 GB (batch dependent) │
│ ─────────────────────────────────────────  │
│ TOTAL:             ~4.1 GB                 │
└────────────────────────────────────────────┘

With AMP (mixed FP16/FP32):
┌────────────────────────────────────────────┐
│ Model weights:     134M × 4 bytes = 536 MB │ (FP32 master copy)
│ FP16 weights:      134M × 2 bytes = 268 MB │ (for forward pass)
│ Gradients:         134M × 2 bytes = 268 MB │ (FP16)
│ Optimizer states:  134M × 8 bytes = 1.07 GB│ (still FP32)
│ Activations:       ~1 GB (half!)           │
│ ─────────────────────────────────────────  │
│ TOTAL:             ~3.1 GB                 │
│ SAVINGS:           ~1 GB (24% less)        │
└────────────────────────────────────────────┘
```

**Bigger win:** Activations scale with batch size. With AMP, you can use 2x larger batches!

---

### Summary Table

| Metric | FP32 | AMP (FP16/FP32) |
|--------|------|-----------------|
| Memory per activation | 4 bytes | 2 bytes |
| Tensor Core utilization | Partial | Full (2x throughput) |
| Training speed | Baseline | 1.5-2x faster |
| Model accuracy | Baseline | Same (with GradScaler) |
| Gradient underflow risk | None | Handled by GradScaler |

---

### Memory Savings Example (Our Model)

- Activations for 134M model: ~2GB FP32 → ~1GB with AMP
- Allows larger batch sizes!

---

## 5. Vectorized Batch Creation

**What it is:** Replace Python loops with array operations that process many elements at once

---

### What is Vectorization?

**Vectorization** = performing operations on entire arrays/tensors at once, instead of looping through elements one by one.

```
Loop approach (scalar):
┌─────────────────────────────────────────────────────┐
│  for i in range(1000):                              │
│      result[i] = a[i] + b[i]   ← 1000 Python calls  │
└─────────────────────────────────────────────────────┘

Vectorized approach:
┌─────────────────────────────────────────────────────┐
│  result = a + b                ← 1 NumPy/PyTorch call│
└─────────────────────────────────────────────────────┘
```

**Key insight:** The vectorized version does the same 1000 additions, but in optimized C/CUDA code instead of Python.

---

### Why Are Python Loops Slow?

Every Python loop iteration has overhead:

```
What happens in a Python loop:

for i in range(1000):        ←── Loop iteration overhead
    result[i] = a[i] + b[i]
         │        │     │
         │        │     └── 3. Index into b (type check, bounds check)
         │        └──────── 2. Index into a (type check, bounds check)
         └───────────────── 1. Index into result (type check, bounds check)
                            4. Add operation (type check, create new object)
                            5. Store result (type check)

Per iteration: ~5-10 Python operations just for bookkeeping!
```

**Visual: Python vs NumPy execution**
```
Python loop (1000 iterations):
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     ┌─────┐
│ i=0 │→│ i=1 │→│ i=2 │→│ i=3 │→...→│i=999│  ← 1000 Python calls
└─────┘ └─────┘ └─────┘ └─────┘     └─────┘
  ↓       ↓       ↓       ↓           ↓
 slow    slow    slow    slow        slow

NumPy vectorized (same 1000 elements):
┌─────────────────────────────────────────┐
│  Single C function processes all 1000   │  ← 1 call to optimized C
│  elements using CPU SIMD instructions   │
└─────────────────────────────────────────┘
                    ↓
               blazing fast
```

---

### SIMD: The Hardware Secret

Modern CPUs have **SIMD** (Single Instruction, Multiple Data) units that process multiple values simultaneously:

```
Without SIMD (scalar):
┌─────────────────────────────────────────────┐
│ Clock 1: a[0] + b[0] = result[0]            │
│ Clock 2: a[1] + b[1] = result[1]            │
│ Clock 3: a[2] + b[2] = result[2]            │
│ Clock 4: a[3] + b[3] = result[3]            │
│ → 4 clock cycles for 4 additions            │
└─────────────────────────────────────────────┘

With SIMD (vectorized):
┌─────────────────────────────────────────────┐
│ Clock 1: [a[0], a[1], a[2], a[3]]           │
│        + [b[0], b[1], b[2], b[3]]           │
│        = [result[0], result[1], result[2],  │
│           result[3]]                        │
│ → 1 clock cycle for 4 additions!            │
└─────────────────────────────────────────────┘
```

NumPy/PyTorch use SIMD automatically. Python loops cannot.

---

### Batch Creation: Before vs After

**The Problem:** We need to create random batches from our token data.

**Before (Python loops - slow)**
```python
batch = []
for i in range(batch_size):           # 64 iterations
    start = random.randint(0, len(data) - seq_len)
    batch.append(data[start:start+seq_len])  # Python list append
batch = torch.tensor(batch)           # Convert at the end
```

**What's happening:**
```
Iteration 1: random() → slice → append → list grows
Iteration 2: random() → slice → append → list grows
Iteration 3: random() → slice → append → list grows
...
Iteration 64: random() → slice → append → list grows
Finally: Convert entire list to tensor

Total: 64 Python function calls + 64 list operations + 1 tensor conversion
```

**After (Vectorized - fast)**
```python
# Generate all 64 random starts at once
starts = np.random.randint(0, len(data) - seq_len, size=batch_size)

# Create all sequences at once
batch = np.stack([data[s:s+seq_len] for s in starts])

# Single conversion to PyTorch
batch = torch.from_numpy(batch)
```

**What's happening:**
```
Step 1: np.random.randint generates 64 numbers in C (one call)
Step 2: np.stack creates the batch array (optimized memory allocation)
Step 3: torch.from_numpy shares memory (zero-copy!)

Total: 3 optimized operations
```

---

### Even Faster: Strided Access

With pre-computed strides, we can make batch creation a single memory operation:

```python
# Precompute all possible sequence start positions
all_starts = np.arange(0, len(data) - seq_len)

# During training: just pick random indices
batch_indices = np.random.choice(len(all_starts), size=batch_size, replace=False)
starts = all_starts[batch_indices]

# Use as_strided for zero-copy view (advanced)
# batch = np.lib.stride_tricks.as_strided(...)
```

---

### Concrete Example: Adding Numbers

```python
import numpy as np
import time

size = 1_000_000

# Setup
a = np.random.rand(size)
b = np.random.rand(size)

# ═══════════════════════════════════════════════════════
# Method 1: Python loop
# ═══════════════════════════════════════════════════════
result = np.zeros(size)
start = time.time()
for i in range(size):
    result[i] = a[i] + b[i]
loop_time = time.time() - start
# → ~500ms on typical CPU

# ═══════════════════════════════════════════════════════
# Method 2: Vectorized
# ═══════════════════════════════════════════════════════
start = time.time()
result = a + b
vectorized_time = time.time() - start
# → ~2ms on typical CPU

# Speedup: ~250x faster!
```

---

### Comparison Table

| Aspect | Python Loops | Vectorized (NumPy/PyTorch) |
|--------|--------------|---------------------------|
| **Execution** | Python interpreter | Compiled C/C++ |
| **Per-element overhead** | ~100-500 ns | ~1-5 ns |
| **SIMD utilization** | None | Full |
| **Memory access** | Random, cache-unfriendly | Sequential, cache-friendly |
| **GPU compatible** | No | Yes (PyTorch) |
| **Code readability** | More verbose | Concise |

---

### Performance Comparison (Batch Creation)

| Batch Size | Python Loop | Vectorized | Speedup |
|------------|------------|------------|---------|
| 16 | ~1.2 ms | ~0.02 ms | 60x |
| 32 | ~2.4 ms | ~0.03 ms | 80x |
| 64 | ~4.8 ms | ~0.05 ms | 96x |
| 128 | ~9.6 ms | ~0.08 ms | 120x |

**Why speedup increases with batch size:** Vectorization amortizes the fixed overhead across more elements.

---

### Impact on Training

```
Per-step breakdown (batch_size=64, seq_len=512):

With Python loops:
┌────────────────────────────────────────────────┐
│ Batch creation:    5 ms   ████████░░░░░░░░░░░░ │
│ Forward pass:     15 ms   ░░░░░░░░████████████ │
│ Backward pass:    25 ms   ░░░░░░░░░░░░░░░░░░░░ │
│ TOTAL:            45 ms                        │
└────────────────────────────────────────────────┘
Batch creation = 11% of step time!

With vectorization:
┌────────────────────────────────────────────────┐
│ Batch creation:  0.05 ms  ░░░░░░░░░░░░░░░░░░░░ │
│ Forward pass:    15 ms    ███████████████░░░░░ │
│ Backward pass:   25 ms    ░░░░░░░░░░░░░░░░░░░░ │
│ TOTAL:          40.05 ms                       │
└────────────────────────────────────────────────┘
Batch creation = 0.1% of step time!
```

---

### Summary: When to Vectorize

| Situation | Recommendation |
|-----------|----------------|
| Loop over array elements | Always vectorize |
| Batch data creation | Always vectorize |
| Random sampling | Use np.random functions |
| Conditional logic on arrays | Use np.where() |
| Custom element-wise operations | Use np.vectorize() or write in NumPy ops |

---

### Quick Vectorization Patterns

```python
# ❌ Loop pattern          →  ✅ Vectorized pattern

# Sum elements
total = 0
for x in arr:              →  total = np.sum(arr)
    total += x

# Conditional count
count = 0
for x in arr:              →  count = np.sum(arr > 0)
    if x > 0:
        count += 1

# Element-wise operation
for i in range(len(arr)):  →  result = arr * 2 + 1
    result[i] = arr[i] * 2 + 1

# Random selection
samples = []
for _ in range(n):         →  samples = np.random.choice(arr, n)
    samples.append(random.choice(arr))
```

---

## 6. Flash Attention (PyTorch 2.0+)

**What it is:** A memory-efficient attention algorithm that computes attention in chunks instead of materializing the full attention matrix.

**Impact:** 0.2s/step → 0.1s/step (2x speedup!)

---

### The Standard Attention Problem

Standard attention computes a massive matrix that grows quadratically with sequence length:

```
Standard Attention:

Input: Q, K, V each of shape [batch, heads, seq_len, head_dim]
       [64, 12, 512, 64]

Step 1: Compute attention scores
        scores = Q @ K.T  →  shape: [64, 12, 512, 512]
                                              ↑    ↑
                                         seq × seq = O(n²)

Step 2: Store this entire matrix in GPU memory
        512 × 512 × 64 batches × 12 heads × 4 bytes = ~800 MB!

Step 3: Apply softmax
Step 4: Multiply by V
```

**The memory problem:**
```
Sequence length vs Attention Memory (per batch, 12 heads):

seq_len=512:   512 × 512 × 12 × 64 × 4 bytes = ~800 MB
seq_len=1024:  1024 × 1024 × 12 × 64 × 4 bytes = ~3.2 GB
seq_len=2048:  2048 × 2048 × 12 × 64 × 4 bytes = ~12.8 GB
seq_len=4096:  4096 × 4096 × 12 × 64 × 4 bytes = ~51 GB  ← Doesn't fit!
```

---

### How Flash Attention Works

Instead of computing the full attention matrix, Flash Attention processes it in **tiles**:

```
Standard Attention (materializes full matrix):
┌─────────────────────────────────────────────┐
│                                             │
│          Full 512×512 attention             │
│              matrix in VRAM                 │
│                                             │
│                 ~800 MB                     │
│                                             │
└─────────────────────────────────────────────┘

Flash Attention (processes in tiles):
┌───────┐
│ tile  │  Process tile 1 → output chunk 1
│  1    │  (discard tile from memory)
└───────┘
         ┌───────┐
         │ tile  │  Process tile 2 → output chunk 2
         │  2    │  (discard tile from memory)
         └───────┘
                  ┌───────┐
                  │ tile  │  ...
                  │  3    │
                  └───────┘

Each tile: ~few MB instead of ~800 MB total
```

---

### The Key Insight: Tiling + Recomputation

Flash Attention uses two tricks:

**1. Tiling:** Process attention in small chunks that fit in fast SRAM (not slow VRAM)

```
GPU Memory Hierarchy:

┌─────────────────────────────────────────────────────────┐
│ SRAM (on-chip)     │ ~20 MB   │ 19 TB/s  │ VERY FAST   │
├────────────────────┼──────────┼──────────┼─────────────┤
│ HBM (VRAM)         │ 40-80 GB │ 2 TB/s   │ 10x slower  │
├────────────────────┼──────────┼──────────┼─────────────┤
│ System RAM         │ 64+ GB   │ 50 GB/s  │ 40x slower  │
└─────────────────────────────────────────────────────────┘

Flash Attention keeps tiles in SRAM → massive bandwidth win!
```

**Understanding GPU Memory Types:**

| Memory Type | Full Name | What It Is |
|-------------|-----------|------------|
| **SRAM** | Static Random Access Memory | On-chip cache built into GPU cores. Tiny (~20MB) but incredibly fast (19 TB/s). Data here can be accessed almost instantly. |
| **HBM** | High Bandwidth Memory | The GPU's main memory (what you see in `nvidia-smi`). Stacked memory chips beside the GPU die. |
| **VRAM** | Video RAM | Marketing term for HBM. When you hear "40GB VRAM" on an A100, that's HBM. Same thing, different name. |

```
Physical Layout on GPU:

┌─────────────────────────────────────────────────────────────┐
│                        GPU Package                          │
│                                                             │
│  ┌─────────────────────────────┐  ┌─────────┐ ┌─────────┐  │
│  │         GPU Die             │  │   HBM   │ │   HBM   │  │
│  │                             │  │  Stack  │ │  Stack  │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐│  │         │ │         │  │
│  │  │ SRAM │ │ SRAM │ │ SRAM ││  │  10GB   │ │  10GB   │  │
│  │  │ Core │ │ Core │ │ Core ││  │         │ │         │  │
│  │  └──────┘ └──────┘ └──────┘│  └─────────┘ └─────────┘  │
│  │                             │        ↑                  │
│  │        ↑                    │        │                  │
│  │        │                    │   Off-chip memory         │
│  │   On-chip cache             │   (fast) ~2 TB/s          │
│  │   (instant) ~19 TB/s        │                           │
│  └─────────────────────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ PCIe bus (~32 GB/s)
                              ▼
                    ┌─────────────────┐
                    │   System RAM    │
                    │  (CPU memory)   │
                    │   64-128 GB     │
                    └─────────────────┘
```

**Why This Matters for Flash Attention:**
```
Standard Attention:
  Q, K, V in HBM → Compute attention → Store 512×512 matrix in HBM
                                       ↑
                              Slow write (2 TB/s)

Flash Attention:
  Q, K, V in HBM → Load tile to SRAM → Compute in SRAM → Next tile
                           ↑                    ↑
                   Fast access (19 TB/s)    Never writes full matrix to HBM!
```

The 10x bandwidth difference between SRAM and HBM is why Flash Attention achieves 2x speedup — it minimizes slow HBM accesses by keeping intermediate computations in fast SRAM.

**2. Recomputation:** Don't store attention weights for backward pass — recompute them
```
Standard: Store attention matrix (800 MB) for backward
Flash:    Recompute attention in backward pass (uses 0 MB storage)

Trade computation for memory → Net win because memory bandwidth is the bottleneck!
```

---

### Enabling Flash Attention

In PyTorch 2.0+, Flash Attention is automatic when using `scaled_dot_product_attention`:

```python
import torch.nn.functional as F

# Old way (materializes full attention matrix):
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
scores = scores.masked_fill(mask == 0, float('-inf'))
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, V)

# New way (uses Flash Attention automatically):
output = F.scaled_dot_product_attention(
    Q, K, V,
    attn_mask=mask,
    is_causal=True  # For decoder-only models like GPT
)
```

PyTorch automatically selects the best backend:
- **Flash Attention** (if available and beneficial)
- **Memory-Efficient Attention** (xFormers-style)
- **Standard math** (fallback)

---

### Requirements for Flash Attention

| Requirement | Details |
|-------------|---------|
| PyTorch version | 2.0+ |
| GPU | NVIDIA (Ampere or newer recommended: A100, 3090, 4090) |
| Head dimension | Must be ≤ 128 and divisible by 8 |
| Dtype | FP16 or BF16 (not FP32) |

```python
# Check if Flash Attention is being used:
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    # Will error if Flash Attention isn't available
```

---

### Performance Comparison

| Metric | Standard Attention | Flash Attention |
|--------|-------------------|-----------------|
| Memory for attention | O(n²) | O(n) |
| Speed (512 seq) | Baseline | 1.5-2x faster |
| Speed (2048 seq) | Baseline | 2-4x faster |
| Speed (8192 seq) | OOM (Out of Memory) | Works! |
| Numerical accuracy | Exact | Mathematically equivalent |

---

### Memory Savings

```
Our model (batch=64, seq=512, heads=12):

Standard Attention:
┌────────────────────────────────────────────┐
│ Attention matrix: 64 × 12 × 512 × 512 × 2  │
│                 = 402 MB (FP16)            │
│ Stored for backward: another 402 MB        │
│ TOTAL: ~800 MB just for attention!         │
└────────────────────────────────────────────┘

Flash Attention:
┌────────────────────────────────────────────┐
│ Tile buffer: ~few MB                       │
│ Stored for backward: O(seq_len) = ~few MB  │
│ TOTAL: ~10-20 MB for attention             │
└────────────────────────────────────────────┘

Savings: ~780 MB → Can use larger batch sizes!
```

---

### Why It's 2x Faster (Not Just Memory)

The speed comes from **memory bandwidth**, not compute:

```
GPU bottleneck analysis:

A100 specs:
- Compute: 312 TFLOPS (FP16)
- Memory bandwidth: 2 TB/s

Standard attention:
- Must read/write 800 MB attention matrix from slow HBM
- Time = 800 MB ÷ 2 TB/s = 0.4 ms per layer
- 12 layers = 4.8 ms just moving attention data!

Flash attention:
- Keeps data in fast SRAM
- Bandwidth: 19 TB/s (not 2 TB/s)
- Time reduced by ~10x for memory-bound operations
```

---

### Summary Table

| Aspect | Standard | Flash Attention |
|--------|----------|-----------------|
| Memory complexity | O(n²) | O(n) |
| Stores attention matrix | Yes | No (recomputes) |
| Uses SRAM tiling | No | Yes |
| Works with long sequences | Limited | Yes |
| Requires PyTorch 2.0+ | No | Yes |
| Speed improvement | Baseline | 1.5-4x |
| Our training impact | 0.2s/step | 0.1s/step |

---

## 7. Training Stability Techniques

These don't speed up individual steps, but they improve **convergence** — getting better results in fewer epochs.

---

### Learning Rate Fundamentals

**What is Learning Rate?**

The **learning rate (LR)** controls how much to adjust the model's weights based on the gradient. It's the single most important hyperparameter in training.

```
Weight update formula:
  new_weight = old_weight - learning_rate × gradient

Example:
  old_weight = 0.5
  gradient = 0.1 (says "increase weight")

  With lr = 0.01:  new_weight = 0.5 - 0.01 × 0.1 = 0.499  (tiny step)
  With lr = 1.0:   new_weight = 0.5 - 1.0 × 0.1  = 0.4    (big step)
```

---

**Visual: Learning Rate as Step Size**

Imagine you're blindfolded on a hilly landscape, trying to find the lowest valley (minimum loss):

**🔴 LR Too High (lr=1.0) — Overshoots and bounces forever**
```
Loss
  │
  │    ·                         ·
  │   · ·                       · ·
  │  ·   ·       ·   ·         ·   ·
  │ ·     ·     · · · ·       ·     ·
  │·  ①    ·   ·       ·     ·
  │         · ·    ③    ·   ·
  │          ②           · ·
  │                       ④        ← Keeps bouncing, never settles!
  └─────────────────────────────────
        ① → ② → ③ → ④ → ...
```

**🟢 LR Just Right (lr=3e-4) — Smooth descent to minimum**
```
Loss
  │
  │    ·                         ·
  │   · ·                       · ·
  │  ·   ·                     ·   ·
  │ ·     ·                   ·     ·
  │·  ①    ·                 ·
  │    ↘    ·               ·
  │     ②   ·             ·
  │      ↘    ·           ·
  │       ③    ·         ·
  │        ↘    ·       ·
  │         ④   ·     ·
  │          ↘   ·   ·
  │           ⑤  · ·
  │            ↘  ★              ★ = Minimum reached!
  └─────────────────────────────────
        ① → ② → ③ → ④ → ⑤ → ★
```

**🟡 LR Too Low (lr=1e-7) — Crawls painfully slow**
```
Loss
  │
  │    ·                         ·
  │   · ·                       · ·
  │  ·   ·                     ·   ·
  │ ·     ·                   ·     ·
  │·  ①    ·                 ·
  │    ②   ·                ·
  │    ③    ·              ·           ← After 1000 steps, barely moved!
  │    ④     ·            ·
  │    ⑤      ·          ·
  │    ...     ·   ★    ·              ★ = Minimum (will take forever)
  └─────────────────────────────────
        ① → ② → ③ → ④ → ⑤ → ... → ★ (10,000+ steps)
```

---

**How Learning Rate Affects Training**

| LR Value | Effect on Training | Typical Outcome |
|----------|-------------------|-----------------|
| **Too High** (>1e-3) | Weights change too much each step | Loss explodes, NaN values, divergence |
| **High** (1e-3 to 1e-4) | Fast learning, some instability | Quick progress but may overshoot |
| **Medium** (1e-4 to 1e-5) | Balanced learning | Good convergence, standard choice |
| **Low** (1e-5 to 1e-6) | Very cautious updates | Slow but stable, good for fine-tuning |
| **Too Low** (<1e-7) | Barely any learning | Wastes compute, may never converge |

---

**LR Does NOT Affect Memory**

Learning rate has zero impact on GPU memory usage:

```
Memory usage is determined by:
✓ Model size (parameters)
✓ Batch size (activations)
✓ Sequence length
✓ Precision (FP16 vs FP32)

Learning rate is just a number:
  lr = 0.001  ← This is a single float, uses ~4 bytes
```

---

**LR Affects Training Speed (Convergence)**

```
High LR (3e-4):                      Low LR (3e-6):

Loss                                 Loss
  │█                                   │█
  │ █                                  │ █
  │  █                                 │  █
  │   █                                │   █
  │    ██                              │    █
  │      ███                           │     █
  │         ████████                   │      █
  │                 ████████           │       █
  └──────────────────────────→         │        █
                         Steps         │         █
                                       │          █
  Reaches low loss quickly!            │           █ ← Still decreasing...
                                       └──────────────────────────────→
                                                                 Steps
                                       Same loss takes 10-100x more steps!
```

---

**Common Learning Rates by Model Size**

| Model Size | Typical LR | Notes |
|------------|-----------|-------|
| GPT-2 Small (124M) | 3e-4 to 6e-4 | Can handle higher LR |
| GPT-2 Medium (355M) | 1e-4 to 3e-4 | |
| GPT-2 Large (774M) | 1e-4 to 2e-4 | |
| GPT-2 XL (1.5B) | 5e-5 to 1e-4 | Larger models need lower LR |
| GPT-3 (175B) | 0.6e-4 | Very low for stability |

**Rule of thumb:** Larger models need smaller learning rates because:
- More parameters = more interactions = more instability
- Gradients can be noisier in deep networks
- Small LR changes have bigger cumulative effects

---

**Diagnosing LR Problems**

```
Loss curve tells you if LR is wrong:

LR too high:                    LR too low:                   LR just right:
Loss                            Loss                          Loss
  │  ╱╲  ╱╲                       │█                            │█
  │ ╱  ╲╱  ╲                      │ █                           │ █
  │╱        ╲  ╱                  │  █                          │  ██
  │          ╲╱                   │   █                         │    ███
  │              ╱╲               │    █                        │       █████
  │             ╱  ╲              │     █████████████████       │            ████████
  └─────────────────→             └─────────────────────→        └─────────────────────→
  Oscillating = reduce LR        Flat too early = increase LR  Smooth descent = good!

Loss = NaN or Inf:
  → LR WAY too high, reduce by 10x
```

---

**Our Training Configuration**

```python
# GPT-2 Small (134M parameters)
learning_rate = 3e-4  # 0.0003

# This means:
# - Each weight changes by at most 0.0003 × gradient per step
# - With 134M parameters, that's 134M small adjustments per step
# - Over 86,000 steps, each weight gets adjusted 86,000 times
```

---

### Learning Rate Scheduler (Cosine Decay)

**What it is:** Gradually decrease learning rate during training following a cosine curve.

```
Learning Rate over Training:

Constant LR (bad):
lr |████████████████████████████████████████
   +----------------------------------------→ Steps
   Same LR entire time → may overshoot optimal weights

Cosine Decay (good):
lr |████████████
   |        ████████
   |                ████████
   |                        ████████
   |                                ████
   +----------------------------------------→ Steps
   High LR early (explore) → Low LR late (fine-tune)
```

**Why it works:**
```
Training phases:

Early training (high LR):
- Weights are random, far from optimal
- Need big steps to make progress
- High LR = fast exploration

Late training (low LR):
- Weights are close to optimal
- Big steps would overshoot
- Low LR = careful fine-tuning
```

**Implementation:**
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps,  # Total training steps
    eta_min=1e-5        # Minimum LR at end
)

# In training loop:
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update LR after each step
```

**Cosine formula:**
```
lr(t) = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(π * t / T_max))

Where:
- t = current step
- T_max = total steps
- lr_max = initial learning rate (3e-4)
- eta_min = minimum learning rate (1e-5)
```

---

### Learning Rate Warmup

**What it is:** Start with a tiny learning rate and gradually increase to target LR over first N steps.

```
Without warmup:
lr |████████████████████████████████████████
   +----------------------------------------→ Steps
   Full LR from step 1 → Unstable early training

With warmup (1000 steps):
lr |    ████████████████████████████████████
   |   █
   |  █
   | █
   |█
   +----------------------------------------→ Steps
   Gradual ramp-up → Stable early training
```

**Why it matters:**
```
Step 1 with random weights:

Without warmup (lr = 3e-4):
- Gradients are huge and noisy (random weights)
- Big LR × huge gradients = massive weight updates
- Weights explode → Loss = NaN

With warmup (lr = 3e-6 → 3e-4 over 1000 steps):
- Small LR × huge gradients = manageable updates
- Weights stabilize before full LR kicks in
- Training proceeds smoothly
```

**Implementation:**
```python
def get_lr(step, warmup_steps, max_lr, total_steps):
    """Linear warmup + cosine decay"""
    if step < warmup_steps:
        # Linear warmup: 0 → max_lr
        return max_lr * step / warmup_steps
    else:
        # Cosine decay: max_lr → min_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

# In training loop:
for step, batch in enumerate(dataloader):
    lr = get_lr(step, warmup_steps=1000, max_lr=3e-4, total_steps=86000)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # ... rest of training step
```

**Visual: Warmup + Cosine Decay Combined:**
```
lr |       ████████████
   |      █            ████████
   |     █                     ████████
   |    █                              ████████
   |   █                                       ████
   |  █
   | █
   |█
   +------------------------------------------------→ Steps
   |←─────→|←────────────────────────────────────→|
    Warmup              Cosine Decay
   (1000 steps)
```

---

### Gradient Accumulation

**What it is:** Simulate larger batch sizes without using more VRAM by accumulating gradients over multiple forward passes.

```
Normal training (batch_size=64):
┌─────────────────────────────────────────────────────┐
│ Forward(64 samples) → Backward → Update weights     │
│ Forward(64 samples) → Backward → Update weights     │
│ Forward(64 samples) → Backward → Update weights     │
└─────────────────────────────────────────────────────┘
3 steps, 3 weight updates, 192 samples processed

Gradient Accumulation (batch=64, accumulate=4, effective_batch=256):
┌─────────────────────────────────────────────────────┐
│ Forward(64) → Backward → accumulate gradients       │
│ Forward(64) → Backward → accumulate gradients       │
│ Forward(64) → Backward → accumulate gradients       │
│ Forward(64) → Backward → UPDATE weights (÷4)        │
└─────────────────────────────────────────────────────┘
4 forward passes, 1 weight update, 256 samples averaged
```

**Why it matters:**
```
Larger batches = smoother gradients = better convergence

But: batch_size=256 needs ~4x memory → OOM!

Solution: Accumulate 4 batches of 64
- Same gradient quality as batch_size=256
- Same memory usage as batch_size=64
- Trade-off: 4x more forward passes per update
```

**Implementation:**
```python
accumulation_steps = 4
optimizer.zero_grad()

for step, batch in enumerate(dataloader):
    # Forward + backward (don't update yet)
    with autocast():
        loss = model(batch)
        loss = loss / accumulation_steps  # Scale loss

    scaler.scale(loss).backward()  # Accumulate gradients

    # Only update weights every N steps
    if (step + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()  # Reset for next accumulation
```

**Key detail: Scale the loss!**
```
Without scaling:
  Batch 1 gradient: 0.1
  Batch 2 gradient: 0.2
  Batch 3 gradient: 0.1
  Batch 4 gradient: 0.2
  ────────────────────
  Accumulated: 0.6  ← 4x too large!

With loss / accumulation_steps:
  Batch 1 gradient: 0.025 (0.1/4)
  Batch 2 gradient: 0.050 (0.2/4)
  Batch 3 gradient: 0.025 (0.1/4)
  Batch 4 gradient: 0.050 (0.2/4)
  ────────────────────
  Accumulated: 0.15  ← Same as averaging!
```

---

### Memory vs Effective Batch Size

| Physical Batch | Accumulation Steps | Effective Batch | VRAM Usage |
|---------------|-------------------|-----------------|------------|
| 64 | 1 | 64 | ~10 GB |
| 64 | 2 | 128 | ~10 GB |
| 64 | 4 | 256 | ~10 GB |
| 64 | 8 | 512 | ~10 GB |
| 128 | 1 | 128 | ~18 GB |
| 256 | 1 | 256 | OOM! |

**Gradient accumulation lets you use larger effective batches without OOM.**

---

### When to Use Each Technique

| Technique | Problem It Solves | When to Use |
|-----------|-------------------|-------------|
| Cosine Decay | LR too high late in training | Always |
| Warmup | Unstable early training, NaN loss | Large models, large LR |
| Gradient Accumulation | Want larger batches but OOM | Limited VRAM |

---

### Complete Training Loop with All Techniques

```python
import torch
import math
from torch.cuda.amp import autocast, GradScaler

# Hyperparameters
max_lr = 3e-4
min_lr = 1e-5
warmup_steps = 1000
total_steps = 86000
accumulation_steps = 4

# Setup
model = TransformerLanguageModel(...).cuda()
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
scaler = GradScaler()

def get_lr(step):
    # Warmup
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# Training loop
optimizer.zero_grad()
for step, batch in enumerate(dataloader):
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Forward + backward with AMP
    with autocast():
        loss = model(batch)
        loss = loss / accumulation_steps

    scaler.scale(loss).backward()

    # Gradient accumulation: update every N steps
    if (step + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if step % 100 == 0:
        print(f"Step {step}, LR: {lr:.2e}, Loss: {loss.item() * accumulation_steps:.4f}")
```

---

### Impact on Training Quality

| Without These Techniques | With These Techniques |
|-------------------------|----------------------|
| Training may diverge early | Stable from step 1 |
| Loss plateaus mid-training | Continuous improvement |
| Stuck in local minima | Better final loss |
| Noisy gradients (small batch) | Smooth gradients (effective large batch) |

---

## Summary: The 7x Journey

### Session 1 (Epochs 1-3): The Dark Ages
- ~105 hours for 3 epochs
- Memory-mapped data (slow random access)
- No torch.compile
- No AMP
- **1.6s per step**

### Session 2 (Epochs 4-10): Enlightened
- RAM preload (instant access)
- torch.compile (kernel fusion)
- AMP (FP16 compute)
- Vectorized batching
- **0.225s per step**

### Time Saved

| Scenario | Time for 10 Epochs |
|----------|-------------------|
| Unoptimized | ~350 hours |
| Optimized | ~50 hours |
| **Saved** | **~300 hours** |

---

## Quick Reference: Enabling Optimizations

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# 1. Load data into RAM (not mmap)
data = np.fromfile('tokens.bin', dtype=np.int16)

# 2. Compile model
model = TransformerLanguageModel(...)
model = torch.compile(model)

# 3. Setup AMP
scaler = GradScaler()

# 4. Training loop
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        with autocast():  # FP16 forward
            loss = model(batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

---

*Notes from GPT-2 training journey, February 2026*
