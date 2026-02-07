"""
Memory-Mapped Training Script for Google Colab
================================================
Trains a transformer language model using pre-tokenized binary data
loaded via memory-mapping. Uses near-zero RAM for dataset storage.

Requires pre-tokenized data from tokenize_local.py or tokenize_local_bpe.py:
  - tokens.bin         (int16 or int32 binary token IDs)
  - tokenizer.json     (vocabulary mappings — character or BPE)
  - tokens_metadata.json (token counts, split info, dtype, tokenizer_type)

Usage:
    python train_colab_mmap.py \
        --bin_file datasets_cache/tokens.bin \
        --tokenizer_file datasets_cache/tokenizer.json \
        --metadata_file datasets_cache/tokens_metadata.json \
        --model_size gpt2_small --batch_size 8 --epochs 15 --use_gpu
"""

import argparse
import json
import math
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
from datetime import datetime

from model import TransformerLanguageModel


# Model size configurations (identical to train_colab.py)
MODEL_CONFIGS = {
    'tiny': {
        'num_layers': 2,
        'embed_dim': 128,
        'num_heads': 4,
        'ff_dim': 512,
        'dropout': 0.1,
        'max_seq_len': 128
    },
    'small': {
        'num_layers': 4,
        'embed_dim': 256,
        'num_heads': 4,
        'ff_dim': 1024,
        'dropout': 0.1,
        'max_seq_len': 256
    },
    'medium': {
        'num_layers': 6,
        'embed_dim': 384,
        'num_heads': 6,
        'ff_dim': 1536,
        'dropout': 0.1,
        'max_seq_len': 512
    },
    'large': {
        'num_layers': 8,
        'embed_dim': 512,
        'num_heads': 8,
        'ff_dim': 2048,
        'dropout': 0.1,
        'max_seq_len': 512
    },
    'xlarge': {
        'num_layers': 10,
        'embed_dim': 768,
        'num_heads': 12,
        'ff_dim': 3072,
        'dropout': 0.1,
        'max_seq_len': 512
    },
    'gpt2_small': {
        'num_layers': 12,
        'embed_dim': 768,
        'num_heads': 12,
        'ff_dim': 3072,
        'dropout': 0.1,
        'max_seq_len': 512
    }
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train transformer with memory-mapped dataset')

    # Dataset (memory-mapped)
    parser.add_argument('--bin_file', type=str, required=True,
                        help='Path to pre-tokenized .bin file')
    parser.add_argument('--tokenizer_file', type=str, required=True,
                        help='Path to tokenizer.json')
    parser.add_argument('--metadata_file', type=str, required=True,
                        help='Path to tokens_metadata.json')

    # Model architecture
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'gpt2_small'],
                        help='Predefined model size configuration')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')

    # Hardware
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU if available')

    # Output
    parser.add_argument('--output_dir', type=str, default='trained_model',
                        help='Directory to save the trained model')

    # Checkpoint saving
    parser.add_argument('--checkpoint_interval', type=int, default=3,
                        help='Save checkpoint every N epochs (default: 3)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Optional directory for checkpoint backups (e.g., Google Drive path)')

    # Steps per epoch cap
    parser.add_argument('--max_steps_per_epoch', type=int, default=0,
                        help='Max steps per epoch (0 = full dataset, default: 0)')

    # Resume from checkpoint
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint directory to resume training from')

    # Evaluation
    parser.add_argument('--eval_batches', type=int, default=50,
                        help='Number of batches for loss estimation (default: 50)')

    # Optimization features
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Accumulate gradients over N steps (effective batch = batch_size * N)')
    parser.add_argument('--use_lr_scheduler', action='store_true',
                        help='Use cosine annealing learning rate scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum learning rate for scheduler (default: 1e-5)')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Number of warmup steps (default: 0, no warmup)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision (FP16) for faster training')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for faster training (PyTorch 2.0+)')

    return parser.parse_args()


def get_batch_mmap(mmap_data, start_idx, end_idx, seq_len, batch_size, device):
    """Generate a random batch from memory-mapped data.

    Reads small slices from disk (via OS page cache), converts int16 to
    torch.long only for the batch. Memory usage per call: ~batch_size * seq_len * 8 bytes.

    Args:
        mmap_data: numpy.memmap of dtype int16
        start_idx: Start of valid range (inclusive)
        end_idx: End of valid range (exclusive)
        seq_len: Sequence length for the model
        batch_size: Number of sequences per batch
        device: torch device (cpu/cuda)

    Returns:
        x: (batch_size, seq_len) tensor of torch.long
        y: (batch_size, seq_len) tensor of torch.long
    """
    max_start = end_idx - seq_len - 1  # -1 for the target shift

    # Random start positions, sorted for better sequential disk access
    ix = np.random.randint(start_idx, max_start, size=(batch_size,))
    ix.sort()

    # Pre-allocate numpy arrays
    x = np.zeros((batch_size, seq_len), dtype=np.int64)
    y = np.zeros((batch_size, seq_len), dtype=np.int64)

    for j, i in enumerate(ix):
        chunk = mmap_data[i: i + seq_len + 1]  # read seq_len+1 int16 values
        x[j] = chunk[:-1]
        y[j] = chunk[1:]

    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def estimate_loss_mmap(model, mmap_data, train_end_idx, total_tokens,
                       seq_len, batch_size, device, criterion, eval_batches=50):
    """Estimate loss on train and val splits using memmap."""
    model.eval()
    losses = {}

    splits = {
        'train': (0, train_end_idx),
        'val': (train_end_idx, total_tokens)
    }

    for split_name, (start_idx, end_idx) in splits.items():
        losses_list = []
        for _ in range(eval_batches):
            X, Y = get_batch_mmap(mmap_data, start_idx, end_idx,
                                  seq_len, batch_size, device)
            logits = model(X)
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), Y.view(B * T))
            losses_list.append(loss.item())
        losses[split_name] = sum(losses_list) / len(losses_list)

    model.train()
    return losses


def save_checkpoint(model, optimizer, tokenizer, config, args, epoch,
                    train_losses, val_losses):
    """Save training checkpoint with full state for resuming"""
    output_dir = args.output_dir
    checkpoint_name = f'checkpoint_epoch_{epoch + 1}'
    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"SAVING CHECKPOINT - Epoch {epoch + 1}/{args.epochs}")
    print('=' * 80)

    # Save model weights
    model_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
    torch.save(model.state_dict(), model_path)
    print(f"  Model weights: {model_path}")

    # Save optimizer state
    optimizer_path = os.path.join(checkpoint_dir, 'optimizer_state.bin')
    torch.save(optimizer.state_dict(), optimizer_path)
    print(f"  Optimizer state: {optimizer_path}")

    # Save tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.json')
    tokenizer.save(tokenizer_path)
    print(f"  Tokenizer: {tokenizer_path}")

    # Save config
    config_to_save = {
        'vocab_size': config['vocab_size'],
        'embed_dim': config['embed_dim'],
        'num_heads': config['num_heads'],
        'num_layers': config['num_layers'],
        'ff_dim': config['ff_dim'],
        'max_seq_len': config['max_seq_len'],
        'dropout': config['dropout'],
        'model_type': 'TransformerLanguageModel',
        'architecture': args.model_size,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'tokenizer_type': config.get('tokenizer_type', 'character')
    }

    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    print(f"  Config: {config_path}")

    # Save training history
    training_history = {
        'epochs_completed': epoch + 1,
        'total_epochs': args.epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_perplexities': [torch.exp(torch.tensor(loss)).item() for loss in train_losses],
        'val_perplexities': [torch.exp(torch.tensor(loss)).item() for loss in val_losses],
        'best_train_loss': min(train_losses),
        'best_val_loss': min(val_losses),
        'best_epoch': val_losses.index(min(val_losses)) + 1
    }

    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"  Training history: {history_path}")

    # Save checkpoint metadata
    checkpoint_metadata = {
        'checkpoint_epoch': epoch + 1,
        'total_epochs': args.epochs,
        'dataset_bin': args.bin_file,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'train_perplexity': torch.exp(torch.tensor(train_losses[-1])).item(),
        'val_perplexity': torch.exp(torch.tensor(val_losses[-1])).item(),
        'best_val_loss': min(val_losses),
        'best_epoch': val_losses.index(min(val_losses)) + 1,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_path = os.path.join(checkpoint_dir, 'checkpoint_info.json')
    with open(metadata_path, 'w') as f:
        json.dump(checkpoint_metadata, f, indent=2)
    print(f"  Checkpoint info: {metadata_path}")

    # Save dataset source reference
    with open(os.path.join(checkpoint_dir, 'dataset_source.txt'), 'w') as f:
        f.write(f"Pre-tokenized binary: {args.bin_file}\n")
        f.write(f"Metadata: {args.metadata_file}\n")
    print(f"  Dataset source: referenced (not copied)")

    print(f"\n[SUCCESS] Checkpoint saved to: {checkpoint_dir}")

    # Backup to Google Drive if specified
    if args.checkpoint_dir:
        try:
            backup_dir = os.path.join(args.checkpoint_dir, checkpoint_name)
            os.makedirs(backup_dir, exist_ok=True)

            for filename in ['pytorch_model.bin', 'optimizer_state.bin', 'tokenizer.json',
                             'config.json', 'training_history.json', 'checkpoint_info.json']:
                src = os.path.join(checkpoint_dir, filename)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(backup_dir, filename))

            print(f"  Checkpoint backed up to: {backup_dir}")
        except Exception as e:
            print(f"  Warning: Could not backup to {args.checkpoint_dir}: {e}")

    print('=' * 80 + '\n')


def save_model(model, optimizer, tokenizer, config, args,
               train_losses, val_losses, duration):
    """Save trained model with complete training state"""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving model to: {output_dir}")

    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    print(f"  Model weights saved")

    # Save optimizer state
    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer_state.bin'))
    print(f"  Optimizer state saved")

    # Save tokenizer
    tokenizer.save(os.path.join(output_dir, 'tokenizer.json'))
    print(f"  Tokenizer saved")

    # Save config
    config_to_save = {
        'vocab_size': config['vocab_size'],
        'embed_dim': config['embed_dim'],
        'num_heads': config['num_heads'],
        'num_layers': config['num_layers'],
        'ff_dim': config['ff_dim'],
        'max_seq_len': config['max_seq_len'],
        'dropout': config['dropout'],
        'model_type': 'TransformerLanguageModel',
        'architecture': args.model_size,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'tokenizer_type': config.get('tokenizer_type', 'character')
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=2)
    print(f"  Config saved")

    # Save training history
    training_history = {
        'epochs_completed': args.epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_perplexities': [torch.exp(torch.tensor(loss)).item() for loss in train_losses],
        'val_perplexities': [torch.exp(torch.tensor(loss)).item() for loss in val_losses],
        'best_train_loss': min(train_losses),
        'best_val_loss': min(val_losses),
        'best_epoch': val_losses.index(min(val_losses)) + 1,
        'duration_minutes': duration / 60
    }

    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"  Training history saved")

    # Save training info
    metadata = {
        'dataset_bin': args.bin_file,
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'duration_minutes': duration / 60,
            'device': 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
        },
        'performance': {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_train_perplexity': torch.exp(torch.tensor(train_losses[-1])).item(),
            'final_val_perplexity': torch.exp(torch.tensor(val_losses[-1])).item(),
            'best_val_loss': min(val_losses),
            'best_epoch': val_losses.index(min(val_losses)) + 1
        },
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Training info saved")

    # Save dataset reference
    with open(os.path.join(output_dir, 'dataset_source.txt'), 'w') as f:
        f.write(f"Pre-tokenized binary: {args.bin_file}\n")
        f.write(f"Metadata: {args.metadata_file}\n")
    print(f"  Dataset source referenced")

    print(f"\n[SUCCESS] All files saved to: {output_dir}")


def train(args):
    """Main training function using memory-mapped dataset"""
    print("\n" + "=" * 80)
    print("MEMORY-MAPPED TRANSFORMER TRAINING")
    print("=" * 80)

    # --- Load metadata ---
    with open(args.metadata_file, 'r') as f:
        metadata = json.load(f)

    total_tokens = metadata['total_tokens']
    vocab_size = metadata['vocab_size']
    train_split_index = metadata['train_split_index']
    val_tokens = total_tokens - train_split_index

    print(f"\nDataset:       {args.bin_file}")
    print(f"Total tokens:  {total_tokens:,}")
    print(f"Vocab size:    {vocab_size}")
    print(f"Train tokens:  {train_split_index:,}")
    print(f"Val tokens:    {val_tokens:,}")

    # --- Load tokenizer (dynamic based on metadata) ---
    tokenizer_type = metadata.get('tokenizer_type', 'character')
    if tokenizer_type == 'bpe':
        from tokenizer_bpe import BPETokenizer
        tokenizer = BPETokenizer()
    else:
        from tokenizer import CharacterTokenizer
        tokenizer = CharacterTokenizer()
    tokenizer.load(args.tokenizer_file)
    print(f"Tokenizer loaded: {tokenizer.vocab_size:,} tokens (type: {tokenizer_type})")

    # --- Open memory-mapped binary file ---
    print(f"\nOpening memory-mapped file...")
    dtype_str = metadata.get('dtype', 'int16')
    mmap_data = np.memmap(args.bin_file, dtype=dtype_str, mode='r')
    print(f"  Data type: {dtype_str}")

    assert len(mmap_data) == total_tokens, \
        f"Memmap size {len(mmap_data)} != metadata total_tokens {total_tokens}"

    bin_size_gb = os.path.getsize(args.bin_file) / (1024 ** 3)
    print(f"  File: {args.bin_file} ({bin_size_gb:.2f} GB)")
    print(f"  Tokens: {len(mmap_data):,}")
    print(f"  RAM usage: near zero (OS page cache)")

    # Quick validation
    print(f"  First 10 tokens: {mmap_data[:10].tolist()}")
    sample_text = tokenizer.decode(mmap_data[:100].astype(int).tolist())
    print(f"  Sample decode: {sample_text[:80]}...")

    # --- Model configuration ---
    config = MODEL_CONFIGS[args.model_size].copy()
    config['vocab_size'] = vocab_size
    config['tokenizer_type'] = tokenizer_type
    seq_len = config['max_seq_len']

    print(f"\nModel size:    {args.model_size}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Epochs:        {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Seq length:    {seq_len}")

    # --- Device setup ---
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("\nUsing CPU")

    # --- Create model ---
    print("\nInitializing model...")
    model = TransformerLanguageModel(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Architecture: {config['num_layers']} layers, {config['embed_dim']} dim, "
          f"{config['num_heads']} heads")

    # --- Training setup ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # --- Resume from checkpoint ---
    start_epoch = 0
    train_losses = []
    val_losses = []

    if args.resume_from:
        print(f"\n{'=' * 80}")
        print(f"RESUMING FROM CHECKPOINT: {args.resume_from}")
        print('=' * 80)

        # Load model weights
        model_path = os.path.join(args.resume_from, 'pytorch_model.bin')
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  Model weights loaded")

        # Load optimizer state
        opt_path = os.path.join(args.resume_from, 'optimizer_state.bin')
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            print(f"  Optimizer state loaded")

        # Load training history
        history_path = os.path.join(args.resume_from, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            train_losses = history.get('train_losses', [])
            val_losses = history.get('val_losses', [])
            start_epoch = history.get('epochs_completed', 0)
            print(f"  Training history loaded: {start_epoch} epochs completed")
            print(f"  Last train loss: {train_losses[-1]:.4f}")
            print(f"  Last val loss:   {val_losses[-1]:.4f}")

        print(f"\n  Resuming from epoch {start_epoch + 1}/{args.epochs}")
        print('=' * 80)

    # Steps per epoch
    full_steps_per_epoch = train_split_index // (args.batch_size * seq_len)
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(args.max_steps_per_epoch, full_steps_per_epoch)
        print(f"\nFull steps per epoch: {full_steps_per_epoch:,}")
        print(f"Capped to: {steps_per_epoch:,} steps per epoch")
    else:
        steps_per_epoch = full_steps_per_epoch
        print(f"\nSteps per epoch: {steps_per_epoch:,}")
    print(f"Tokens per step: {args.batch_size * seq_len:,}")
    remaining_epochs = args.epochs - start_epoch
    total_tokens_seen = steps_per_epoch * remaining_epochs * args.batch_size * seq_len
    print(f"Remaining epochs: {remaining_epochs}")
    print(f"Total tokens to train on: {total_tokens_seen:,} ({total_tokens_seen / train_split_index:.1f}x dataset)")

    # --- Learning rate scheduler (optional) ---
    scheduler = None
    if args.use_lr_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
        total_steps = steps_per_epoch * remaining_epochs

        if args.warmup_steps > 0:
            # Warmup + cosine decay
            def lr_lambda(current_step):
                if current_step < args.warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, args.warmup_steps))
                # Cosine decay after warmup
                progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
                return max(args.min_lr / args.learning_rate, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = LambdaLR(optimizer, lr_lambda)
            print(f"\nLR Scheduler: Warmup ({args.warmup_steps} steps) + Cosine decay")
        else:
            # Pure cosine decay
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)
            print(f"\nLR Scheduler: Cosine annealing (min_lr={args.min_lr})")

        print(f"Total scheduled steps: {total_steps:,}")

    # --- AMP and torch.compile setup ---
    use_amp = args.use_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) - FP16")

    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
        print("Model compiled!")

    # --- Training loop ---
    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80)

    start_time = time.time()

    # Gradient accumulation setup
    accumulation_steps = args.gradient_accumulation_steps
    effective_batch_size = args.batch_size * accumulation_steps
    if accumulation_steps > 1:
        print(f"Gradient accumulation: {accumulation_steps} steps")
        print(f"Effective batch size: {effective_batch_size}")

    global_step = 0
    log_interval = 100  # Log every N steps

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        model.train()
        optimizer.zero_grad()

        for step in range(steps_per_epoch):
            X, Y = get_batch_mmap(mmap_data, 0, train_split_index,
                                  seq_len, args.batch_size, device)

            # Forward pass with optional AMP
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(X)
                    B, T, C = logits.shape
                    loss = criterion(logits.view(B * T, C), Y.view(B * T))
            else:
                logits = model(X)
                B, T, C = logits.shape
                loss = criterion(logits.view(B * T, C), Y.view(B * T))

            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps

            # Backward pass with optional AMP
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            epoch_loss_sum += loss.item()
            epoch_loss_count += 1

            # Gradient step (every accumulation_steps or at epoch end)
            if (step + 1) % accumulation_steps == 0 or (step + 1) == steps_per_epoch:
                # Gradient clipping for stability
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Optimizer step with optional AMP
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # Step the scheduler if using one
                if scheduler is not None:
                    scheduler.step()

                global_step += 1

            # Progress update with s/step and ETA
            if (step + 1) % log_interval == 0:
                elapsed = time.time() - epoch_start
                steps_done = step + 1
                steps_remaining = steps_per_epoch - steps_done
                s_per_step = elapsed / steps_done
                eta_seconds = steps_remaining * s_per_step
                eta_minutes = eta_seconds / 60

                avg_loss = epoch_loss_sum / epoch_loss_count
                current_lr = optimizer.param_groups[0]['lr']

                print(f"Epoch {epoch + 1}/{args.epochs} | "
                      f"Step {steps_done}/{steps_per_epoch} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"{s_per_step:.3f}s/step | "
                      f"ETA: {eta_minutes:.0f}min | "
                      f"LR: {current_lr:.2e}")

        # Epoch evaluation
        losses = estimate_loss_mmap(model, mmap_data, train_split_index,
                                    total_tokens, seq_len, args.batch_size,
                                    device, criterion, args.eval_batches)
        train_loss = losses['train']
        val_loss = losses['val']

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_perp = torch.exp(torch.tensor(train_loss)).item()
        val_perp = torch.exp(torch.tensor(val_loss)).item()

        epoch_time = time.time() - epoch_start

        print("\n" + "-" * 80)
        print(f"Epoch {epoch + 1}/{args.epochs} Complete:")
        print(f"  Train Loss: {train_loss:.4f} | Perplexity: {train_perp:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Perplexity: {val_perp:.2f}")
        print(f"  Time: {epoch_time / 60:.1f} min")
        print("-" * 80 + "\n")

        # Save checkpoint every N epochs
        if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, tokenizer, config, args,
                            epoch, train_losses, val_losses)

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE - {total_time / 60:.1f} minutes")
    print("=" * 80)

    # Save final model
    save_model(model, optimizer, tokenizer, config, args,
               train_losses, val_losses, total_time)

    print("\n" + "=" * 80)
    print("TRAINING FINISHED SUCCESSFULLY!")
    print("=" * 80)


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
