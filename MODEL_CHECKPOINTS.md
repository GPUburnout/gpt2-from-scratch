# Model Checkpoints Summary

This document summarizes the model checkpoints from the GPT-2 training journey.

## Overview

| Phase | Model Name | Parameters | Tokenizer | Training Data |
|-------|------------|------------|-----------|---------------|
| 1 | Tiny Shakespeare | 3.2M | Character (vocab: 65) | Shakespeare corpus |
| 2 | Medium Character | 3.3M | Character (vocab: 190) | 250MB foundational dataset |
| 3 | GPT-2 Small | 134M | BPE (vocab: 32,000) | 12GB dataset |

---

## Phase 1: Tiny Shakespeare

**Folder:** `checkpoint_tiny/`

| Property | Value |
|----------|-------|
| Total Parameters | 3,221,825 |
| Vocab Size | 65 |
| Embedding Dimension | 384 |
| Attention Heads | 6 |
| Layers | 6 |
| Feed-Forward Dimension | 1536 |
| Max Sequence Length | 256 |
| Tokenizer Type | Character-level |
| Model File Size | ~13 MB |

**Notes:**
- First model in the training progression
- Trained on Shakespeare text corpus
- Character-level tokenization (each character is a token)

---

## Phase 2: Medium Character

**Folder:** `checkpoint_medium/`

| Property | Value |
|----------|-------|
| Total Parameters | 3,322,368 |
| Vocab Size | 190 |
| Embedding Dimension | 256 |
| Attention Heads | 4 |
| Layers | 4 |
| Feed-Forward Dimension | 1024 |
| Max Sequence Length | 256 |
| Tokenizer Type | Character-level |
| Model File Size | ~13 MB |

**Notes:**
- Trained on 250MB foundational dataset
- Larger vocabulary (190 vs 65) to handle more diverse text
- Similar parameter count to Tiny but different architecture
- Originally labeled as "10-50M params" but actual size is 3.3M

---

## Phase 3: GPT-2 Small

**Folder:** `checkpoint_gpt2_small/`

| Property | Value |
|----------|-------|
| Total Parameters | 134,601,216 |
| Vocab Size | 32,000 |
| Embedding Dimension | 768 |
| Attention Heads | 12 |
| Layers | 12 |
| Feed-Forward Dimension | 3072 |
| Max Sequence Length | 512 |
| Tokenizer Type | BPE (Byte-Pair Encoding) |
| Model File Size | ~515 MB |

**Notes:**
- Final checkpoint (epoch 11) - training complete
- Uses BPE tokenization instead of character-level
- Trained on 12GB dataset (2.8B tokens)
- Model file too large for GitHub (>100MB limit) - hosted on HuggingFace

---

## File Structure

```
checkpoint_tiny/
├── config.json          # Model configuration
├── pytorch_model.bin    # Model weights (~13 MB)
└── tokenizer.json       # Character tokenizer

checkpoint_medium/
├── config.json          # Model configuration
├── pytorch_model.bin    # Model weights (~13 MB)
└── tokenizer.json       # Character tokenizer

checkpoint_gpt2_small/
├── config.json          # Model configuration
├── pytorch_model.bin    # Model weights (~515 MB) [NOT on GitHub]
├── tokenizer.json       # BPE tokenizer
└── NOTE.txt             # Training status note
```

---

## Model Size Calculation

Model file sizes follow the formula: `parameters × 4 bytes` (for float32)

| Model | Parameters | Expected Size | Actual Size |
|-------|------------|---------------|-------------|
| Tiny | 3.2M | ~13 MB | 13 MB |
| Medium | 3.3M | ~13 MB | 13 MB |
| GPT-2 Small | 134M | ~536 MB | 515 MB |

---

## GitHub vs HuggingFace

Due to GitHub's 100MB file size limit:
- **On GitHub:** Tiny and Medium checkpoints are complete; GPT-2 Small has config/tokenizer only
- **On HuggingFace Spaces:** All models including full GPT-2 Small weights

---

*Last updated: February 2026*
