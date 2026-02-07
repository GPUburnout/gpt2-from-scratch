# GPT-2 From Scratch

Training GPT-2 Small (134M parameters) from scratch on 12GB of conversational data.

This is the code behind my blog series: [Building a Language Model From Scratch](https://your-blog-url-here)

## What's Here

| File | Purpose |
|------|---------|
| `model.py` | GPT-style Transformer architecture with Flash Attention support |
| `tokenizer.py` | Character-level tokenizer (Phase 1-2) |
| `tokenizer_bpe.py` | BPE tokenizer wrapper using HuggingFace tokenizers (Phase 3) |
| `train_colab_mmap.py` | Memory-mapped training script for Google Colab |
| `generate.py` | Text generation with trained models |
| `tokenize_local_bpe.py` | Pre-tokenize datasets to binary format |
| `dataset_cleaner.py` | Text cleaning utilities |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Pre-tokenize your text file to binary format:

```bash
python tokenize_local_bpe.py \
    --input your_dataset.txt \
    --vocab_size 32000 \
    --compress
```

This creates:
- `tokens_bpe.bin` - Binary token file
- `tokens_bpe.bin.gz` - Compressed version (upload this to Drive)
- `bpe_tokenizer.json` - Tokenizer vocabulary
- `tokens_bpe_metadata.json` - Dataset metadata

### 3. Train on Colab

Upload the compressed files to Google Drive, then run:

```python
# In Colab: decompress the data
!gunzip /content/drive/MyDrive/tokens_bpe.bin.gz -c > /content/tokens_bpe.bin

# Clone this repo
!git clone https://github.com/YOUR_USERNAME/gpt2-from-scratch.git
%cd gpt2-from-scratch

# Train
!python train_colab_mmap.py \
    --bin_file /content/tokens_bpe.bin \
    --tokenizer_file /content/drive/MyDrive/bpe_tokenizer.json \
    --metadata_file /content/drive/MyDrive/tokens_bpe_metadata.json \
    --model_size gpt2_small \
    --batch_size 64 \
    --epochs 10 \
    --use_gpu \
    --checkpoint_dir /content/drive/MyDrive/checkpoints
```

### 4. Generate Text

```python
from model import TransformerLanguageModel
from tokenizer_bpe import BPETokenizer
from generate import generate_text
import torch

# Load model and tokenizer
model = TransformerLanguageModel(...)
model.load_state_dict(torch.load('checkpoint/pytorch_model.bin'))
tokenizer = BPETokenizer()
tokenizer.load('checkpoint/tokenizer.json')

# Generate
text = generate_text(
    model, tokenizer,
    prompt="What is the capital of France?",
    max_new_tokens=100,
    temperature=0.8
)
print(text)
```

## Model Configurations

| Size | Layers | Heads | Dim | Params | Use Case |
|------|--------|-------|-----|--------|----------|
| `tiny` | 2 | 4 | 128 | ~400K | CPU debugging |
| `small` | 4 | 4 | 256 | ~10M | Quick experiments |
| `medium` | 6 | 6 | 384 | ~50M | Colab T4/V100 |
| `gpt2_small` | 12 | 12 | 768 | 134M | Colab A100 |

## Training Tips

1. **Pre-tokenize locally.** Processing text during training is slow.
2. **Use RAM preload over mmap.** If your data fits in Colab's RAM, preload it.
3. **Enable torch.compile + AMP.** Free 2-3x speedup.
4. **Checkpoint to Drive.** Colab will disconnect. Save every 2-3 epochs.

## The Journey (Blog Posts)

1. [Why I Decided to Build a Language Model from Scratch](link)
2. [Data Preparation: Building a 12GB Training Corpus](link)
3. [Scaling Up: From Tiny Model to GPT-2 Small](link)
4. [10 Training Challenges and How I Solved Them](link)
5. [The Results Are In (And My Wallet Is Empty)](link)

## Results

After 10 epochs (~50 hours on A100):

| Prompt | Output |
|--------|--------|
| "What is the capital of France?" | "The capital of France is Paris." |
| "Explain machine learning in simple terms." | "Machine learning is a way for computers to learn from data without being explicitly programmed." |

## License

MIT
