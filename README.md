# Claude First Run - Tiny Language Model

Building a tiny language model from scratch with Claude Code.

## Project Structure

```
Claude_First_Run/
├── dataset_browser.py       # Browse and select datasets
├── dataset_loader.py        # Download and preprocess data
├── tokenizer.py             # Convert text to numbers
├── model.py                 # Model architecture
├── train.py                 # Training loop
├── generate.py              # Text generation
├── data/                    # Downloaded datasets (gitignored)
├── models/                  # Trained models (gitignored)
└── checkpoints/             # Training checkpoints (gitignored)
```

## Steps

1. ✅ Dataset Browser - Select training data
2. ⏳ Tokenizer - Convert text to numbers
3. ⏳ Model Architecture - Build the neural network
4. ⏳ Training Loop - Train the model
5. ⏳ Text Generation - Generate new text

## Getting Started

1. Select a dataset:
   ```bash
   python dataset_browser.py
   ```

2. Train the model:
   ```bash
   python train.py
   ```

3. Generate text:
   ```bash
   python generate.py
   ```

## Requirements

- Python 3.7+
- PyTorch
- datasets (Hugging Face)
- numpy
