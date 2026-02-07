# GPT-2 Training Optimizations Summary

## Session Date: February 7, 2026

---

## Overview

This document summarizes the key optimizations applied to accelerate GPT-2 Small (134M parameters) training on Google Colab with NVIDIA A100 GPU.

### Performance Improvement

| Configuration | Speed (s/step) | Relative |
|--------------|----------------|----------|
| CPU (baseline) | 0.365 | 1.0x |
| GPU only | 0.365 | 1.0x (bottlenecked) |
| GPU + AMP | ~0.20 | 1.8x |
| GPU + AMP + torch.compile() | **0.113** | **3.2x** |

**Final result: 3.2x faster training**

---

## Key Issues Discovered

### 1. GPU Not Being Used
**Symptom:** Training showed "Using GPU" but ran at CPU speed (0.365s/step)

**Root Cause:** The `--use_gpu` flag was required but not included in the command

**Fix:** Added `--use_gpu` to training command

### 2. No Mixed Precision (AMP)
**Symptom:** GPU utilization was low, speed didn't improve much with GPU

**Root Cause:** Script was running in FP32 (full precision) instead of FP16

**Fix:** Added `--use_amp` flag and AMP logic to training script

### 3. No Model Compilation
**Symptom:** Good but not optimal GPU performance

**Root Cause:** PyTorch was running in eager mode, not compiled

**Fix:** Added `--compile` flag with `torch.compile()` support

### 4. Output Buffering in Colab
**Symptom:** Training appeared stuck with no output for minutes

**Root Cause:** `subprocess.run()` buffers output until process completes

**Fix:** Changed to `os.system()` with `shlex.quote()` for real-time output

---

## Optimizations Explained

### 1. Flash Attention
**What:** Optimized attention algorithm that reduces memory I/O

**How it accelerates:**
- Fuses multiple operations into single GPU kernel
- Reduces memory bandwidth bottleneck
- Avoids materializing large attention matrices

**Implementation:** Already in `model.py` using PyTorch's `scaled_dot_product_attention` with `is_causal=True`

**Speedup:** 1.5-2x for attention operations

```python
# In model.py MultiHeadAttention
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=dropout_p,
    is_causal=True  # Enables Flash Attention path
)
```

---

### 2. Automatic Mixed Precision (AMP)
**What:** Uses FP16 (half precision) for most operations, FP32 for sensitive ones

**How it accelerates:**
- A100 Tensor Cores are optimized for FP16 (~300 TFLOPS vs ~20 TFLOPS for FP32)
- Halves memory bandwidth requirements
- Allows larger batch sizes due to reduced memory usage

**Implementation:** Added to `train_colab_mmap.py`

**Speedup:** 2-5x on modern GPUs with Tensor Cores

```python
# Initialization
use_amp = args.use_amp and device.type == 'cuda'
scaler = torch.cuda.amp.GradScaler() if use_amp else None

# Forward pass
with torch.cuda.amp.autocast():
    logits = model(X)
    loss = criterion(logits.view(B * T, C), Y.view(B * T))

# Backward pass
scaler.scale(loss).backward()

# Optimizer step
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

---

### 3. torch.compile()
**What:** JIT compilation of PyTorch models using TorchDynamo + Inductor

**How it accelerates:**
- Traces and optimizes computation graph
- Fuses operations to reduce kernel launches
- Generates optimized CUDA code
- Reduces Python overhead

**Implementation:** Added to `train_colab_mmap.py`

**Speedup:** 1.2-1.5x (additional, on top of other optimizations)

**Note:** First few hundred steps are slower as the model compiles ("warmup")

```python
if args.compile and hasattr(torch, 'compile'):
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)
    print("Model compiled!")
```

---

### 4. Gradient Accumulation
**What:** Accumulates gradients over multiple mini-batches before updating weights

**How it accelerates:**
- Enables larger effective batch sizes without more GPU memory
- Reduces optimizer overhead (fewer weight updates)
- Better gradient estimates = more stable training

**Implementation:** Already in `train_colab_mmap.py`

```python
# Accumulate gradients
scaled_loss = loss / accumulation_steps
scaled_loss.backward()

# Only step optimizer every N batches
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

**Configuration used:**
- Batch size: 32
- Accumulation steps: 2
- Effective batch: 64

---

### 5. Learning Rate Scheduler
**What:** Cosine annealing with warmup

**How it helps:**
- Warmup prevents early training instability
- Cosine decay finds better minima than constant LR
- Gradual reduction allows fine-grained convergence

**Configuration:**
- Initial LR: 0.001
- Warmup: 1000 steps
- Final LR: 1e-5 (cosine decay)

---

## Files Modified

### `train_colab_mmap.py`
Added:
- `--use_amp` argument and AMP training logic
- `--compile` argument and torch.compile() support
- GradScaler initialization
- autocast context manager in forward pass
- Scaled backward pass and optimizer step

### `colab_unified_training_v6_mmap.ipynb`
- Fixed output buffering (subprocess → os.system)
- Added checkpoint discovery
- Improved configuration UI

---

## Training Command (Final)

```bash
!python train_colab_mmap.py \
    --bin_file /content/datasets_cache/tokens_bpe.bin \
    --tokenizer_file /content/datasets_cache/bpe_tokenizer.json \
    --metadata_file /content/datasets_cache/tokens_bpe_metadata.json \
    --model_size gpt2_small \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 0.001 \
    --epochs 10 \
    --checkpoint_interval 1 \
    --use_gpu \
    --use_amp \
    --compile \
    --use_lr_scheduler \
    --warmup_steps 1000 \
    --min_lr 1e-5 \
    --resume_from /path/to/checkpoint \
    --output_dir /content/drive/MyDrive/trained_models/...
```

---

## Key Learnings

### 1. Always Verify GPU Usage
Just passing `--use_gpu` isn't enough. Check:
- nvidia-smi shows memory usage
- Output confirms "Using GPU: ..."
- Speed is appropriate for GPU

### 2. AMP is Critical for Modern GPUs
A100/H100/RTX 30/40 series have Tensor Cores optimized for FP16. Without AMP, you're using only ~10% of potential performance.

### 3. torch.compile() Needs Warmup
First 100-500 steps are slower as PyTorch compiles the model. Speed improves dramatically after warmup.

### 4. Memory-Mapped Files Work Well
The mmap approach (`np.memmap`) allows training on datasets larger than RAM with minimal performance impact, especially with fast SSDs.

### 5. Checkpoint Frequently
Training can be interrupted at any time. Saving checkpoints every epoch ensures minimal lost progress.

---

## Performance Timeline

```
Step 100:  0.402 s/step (torch.compile warmup)
Step 200:  0.252 s/step (warming up)
Step 300:  0.202 s/step (warming up)
Step 400:  0.177 s/step (warming up)
Step 500:  0.162 s/step (warming up)
Step 1000: 0.131 s/step (nearly warmed)
Step 2000: 0.116 s/step (stabilizing)
Step 2500: 0.113 s/step (stable) ← Final speed
```

---

## Compute Budget Analysis

| Metric | Value |
|--------|-------|
| A100 cost | ~7 compute units/hour |
| Speed | 0.113 s/step |
| Steps/epoch | 159,015 |
| Time/epoch | ~5 hours |
| Epochs planned | 3 (from checkpoint 7) |
| Total time needed | ~15 hours |
| Compute units needed | ~105 units |
| User's budget | 120 units |
| **Verdict** | Sufficient |

---

## Next Steps

1. Monitor training progress (loss should continue decreasing from ~3.0)
2. After training completes, test generation quality
3. Consider pushing optimized `train_colab_mmap.py` to GitHub
4. Document final model performance and sample outputs

---

## Summary

By adding AMP and torch.compile(), training speed improved from **0.365 s/step to 0.113 s/step** - a **3.2x speedup**. This reduced estimated training time from ~32 hours to ~5 hours per epoch, making it feasible to complete training within the available compute budget.
