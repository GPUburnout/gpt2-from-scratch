"""
LoRA Fine-tuning Script for Instruction Following
==================================================
Fine-tunes a pre-trained GPT-2 model using LoRA with anti-forgetting protections.

Key features:
- LoRA: Only ~2% of parameters trainable
- Low learning rate (2e-5) to prevent catastrophic forgetting
- Short epochs (2-3) to avoid overfitting
- Optional replay buffer from original data
- Gradient clipping for stability

Speed Optimizations:
- torch.compile() for fused kernels (2x speedup)
- AMP (FP16) for faster compute
- Flash Attention via scaled_dot_product_attention
- Pinned memory for faster GPU transfers
- Optimized gradient accumulation

Usage (Local):
    py finetune_lora.py \
        --base_model models/gpt2_trained \
        --instruction_data datasets_cache/instruction/instruction_tokens.bin \
        --output_dir models/gpt2_instruct \
        --epochs 3 --learning_rate 2e-5

Usage (Colab):
    python finetune_lora.py \
        --base_model /content/drive/MyDrive/trained_models/gpt2_e11 \
        --instruction_data /content/instruction/instruction_tokens.bin \
        --output_dir /content/drive/MyDrive/trained_models/gpt2_instruct \
        --epochs 3 --use_gpu --use_amp --compile
"""

import argparse
import json
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from model import TransformerLanguageModel
from lora import apply_lora, get_lora_state_dict, merge_lora_weights


def parse_args():
    parser = argparse.ArgumentParser(description='LoRA fine-tuning for instruction following')

    # Model paths
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to pre-trained model directory (with pytorch_model.bin and config.json)')
    parser.add_argument('--output_dir', type=str, default='models/gpt2_instruct',
                        help='Output directory for fine-tuned model')

    # Dataset
    parser.add_argument('--instruction_data', type=str, required=True,
                        help='Path to instruction_tokens.bin')
    parser.add_argument('--metadata_file', type=str, default=None,
                        help='Path to instruction_tokens_metadata.json (auto-detected if not provided)')

    # Replay buffer (anti-forgetting)
    parser.add_argument('--replay_data', type=str, default=None,
                        help='Optional: Path to original pre-training tokens.bin for replay')
    parser.add_argument('--replay_ratio', type=float, default=0.1,
                        help='Ratio of replay samples per batch (default: 0.1)')

    # LoRA configuration
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank (default: 16, lower=fewer params)')
    parser.add_argument('--lora_alpha', type=float, default=32,
                        help='LoRA alpha scaling (default: 32, typically 2*rank)')
    parser.add_argument('--lora_targets', type=str, nargs='+', default=['qkv', 'out_proj'],
                        help='Which layers to apply LoRA to (default: qkv out_proj)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8, keep small for instruction tuning)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs (default: 3, keep short)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5, keep LOW)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping (default: 1.0)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio (default: 0.1)')

    # Hardware & Speed
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision (FP16)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for faster training (PyTorch 2.0+)')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode (default: reduce-overhead)')

    # Output options
    parser.add_argument('--save_lora_only', action='store_true',
                        help='Only save LoRA weights (small file, ~1-5 MB)')
    parser.add_argument('--merge_weights', action='store_true',
                        help='Merge LoRA into base weights before saving')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='Save checkpoint every N epochs')

    # Logging
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log every N steps')

    return parser.parse_args()


def load_base_model(model_dir, device):
    """Load pre-trained base model"""
    print(f"\n[Loading] Base model from: {model_dir}")

    # Load config
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"   Config: {config['embed_dim']}d, {config['num_layers']}L, {config['num_heads']}H")
    print(f"   Vocab: {config['vocab_size']}, Seq: {config['max_seq_len']}")

    # Create model
    model = TransformerLanguageModel(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config.get('ff_dim', 4 * config['embed_dim']),
        max_seq_len=config['max_seq_len'],
        dropout=config.get('dropout', 0.1)
    )

    # Load weights
    weights_path = os.path.join(model_dir, 'pytorch_model.bin')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Handle torch.compile() prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)

    print(f"   Parameters: {model.count_parameters():,}")

    return model, config


def create_optimized_dataloader(bin_path, metadata_path, seq_len, batch_size, device):
    """Create optimized data loader with pinned memory for fast GPU transfers"""
    print(f"\n[Loading] Instruction data: {bin_path}")

    # Load metadata
    if metadata_path is None:
        metadata_path = bin_path.replace('.bin', '_metadata.json')

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    dtype_str = metadata.get('dtype', 'int32')
    total_tokens = metadata['total_tokens']

    print(f"   Tokens: {total_tokens:,}")
    print(f"   dtype: {dtype_str}")

    # Memory-map the data
    mmap_data = np.memmap(bin_path, dtype=dtype_str, mode='r')
    data_len = len(mmap_data)

    # Calculate number of complete sequences
    num_sequences = (data_len - 1) // seq_len
    print(f"   Sequences (len={seq_len}): {num_sequences:,}")

    # Pre-allocate pinned memory buffers for faster GPU transfers
    use_pinned = device.type == 'cuda'
    if use_pinned:
        print("   Using pinned memory for fast GPU transfers")
        x_buffer = torch.empty((batch_size, seq_len), dtype=torch.int64, pin_memory=True)
        y_buffer = torch.empty((batch_size, seq_len), dtype=torch.int64, pin_memory=True)
    else:
        x_buffer = None
        y_buffer = None

    def get_batch(batch_idx):
        """Get a batch of sequences with optimized memory transfer"""
        # Random sampling for instruction tuning
        indices = np.random.randint(0, data_len - seq_len - 1, size=batch_size)

        # Vectorized batch creation
        x_np = np.stack([mmap_data[i:i+seq_len] for i in indices]).astype(np.int64)
        y_np = np.stack([mmap_data[i+1:i+seq_len+1] for i in indices]).astype(np.int64)

        if use_pinned:
            # Copy to pinned memory, then async transfer to GPU
            x_buffer.copy_(torch.from_numpy(x_np))
            y_buffer.copy_(torch.from_numpy(y_np))
            x = x_buffer.to(device, non_blocking=True)
            y = y_buffer.to(device, non_blocking=True)
        else:
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)

        return x, y

    steps_per_epoch = num_sequences // batch_size

    return get_batch, steps_per_epoch, metadata


def train_epoch(model, get_batch, steps_per_epoch, optimizer, scheduler, scaler,
                criterion, device, args, epoch, global_step):
    """Train for one epoch with all optimizations"""
    model.train()

    total_loss = 0
    log_loss = 0
    log_steps = 0
    step_times = []
    start_time = time.time()
    step_start = time.time()

    # Use set_to_none for faster zeroing
    optimizer.zero_grad(set_to_none=True)

    for step in range(steps_per_epoch):
        # Get batch
        X, Y = get_batch(step)

        # Sync for accurate timing on first few steps
        if step < 5 and device.type == 'cuda':
            torch.cuda.synchronize()

        B, T = X.shape
        C = model.vocab_size

        # Forward pass with AMP
        if args.use_amp and scaler is not None:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(X)
                loss = criterion(logits.view(B * T, C), Y.view(B * T))
        else:
            logits = model(X)
            loss = criterion(logits.view(B * T, C), Y.view(B * T))

        # Scale loss for gradient accumulation
        scaled_loss = loss / args.gradient_accumulation_steps

        # Backward pass
        if args.use_amp and scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Update weights every N steps
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            if args.use_amp and scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Optimizer step
            if args.use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        total_loss += loss.item()
        log_loss += loss.item()
        log_steps += 1

        # Track step time
        if device.type == 'cuda':
            torch.cuda.synchronize()
        step_time = time.time() - step_start
        step_times.append(step_time)
        step_start = time.time()

        # Logging
        if (step + 1) % args.log_interval == 0:
            avg_loss = log_loss / log_steps
            avg_step_time = sum(step_times[-args.log_interval:]) / min(len(step_times), args.log_interval)
            steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
            lr = optimizer.param_groups[0]['lr']

            ppl = math.exp(min(avg_loss, 20))  # Cap to avoid overflow

            # Calculate tokens/sec
            tokens_per_step = args.batch_size * model.max_seq_len
            tokens_per_sec = tokens_per_step * steps_per_sec

            print(f"  Step {step+1:5d}/{steps_per_epoch} | "
                  f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                  f"LR: {lr:.2e} | "
                  f"{avg_step_time*1000:.0f}ms/step | "
                  f"{tokens_per_sec/1000:.1f}K tok/s")

            log_loss = 0
            log_steps = 0

    avg_epoch_loss = total_loss / steps_per_epoch
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0

    return avg_epoch_loss, global_step, avg_step_time


def save_model(model, config, output_dir, args, epoch=None, lora_only=False):
    """Save model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_e{epoch}" if epoch is not None else ""

    # Handle compiled model
    model_to_save = model
    if hasattr(model, '_orig_mod'):
        model_to_save = model._orig_mod

    if lora_only:
        # Save only LoRA weights (small file)
        lora_state = get_lora_state_dict(model_to_save)
        lora_path = os.path.join(output_dir, f"lora_weights{suffix}.pt")
        torch.save(lora_state, lora_path)
        print(f"   Saved LoRA weights: {lora_path} ({os.path.getsize(lora_path)/1024:.1f} KB)")
    else:
        # Save full model
        weights_path = os.path.join(output_dir, f"pytorch_model{suffix}.bin")
        torch.save(model_to_save.state_dict(), weights_path)
        print(f"   Saved model: {weights_path} ({os.path.getsize(weights_path)/1024/1024:.1f} MB)")

    # Save config
    config_to_save = config.copy()
    config_to_save['fine_tuned'] = True
    config_to_save['fine_tune_type'] = 'lora_instruction'
    config_to_save['lora_config'] = getattr(model_to_save, 'lora_config', None)
    config_to_save['fine_tune_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)


def main():
    args = parse_args()

    print("="*70)
    print("LoRA INSTRUCTION FINE-TUNING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Base model:     {args.base_model}")
    print(f"  Instruction:    {args.instruction_data}")
    print(f"  Output:         {args.output_dir}")
    print(f"\nLoRA:")
    print(f"  Rank:           {args.lora_rank}")
    print(f"  Alpha:          {args.lora_alpha}")
    print(f"  Targets:        {args.lora_targets}")
    print(f"\nTraining:")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Grad accum:     {args.gradient_accumulation_steps}")
    print(f"  Effective BS:   {args.batch_size * args.gradient_accumulation_steps}")
    print(f"\nOptimizations:")
    print(f"  AMP (FP16):     {args.use_amp}")
    print(f"  torch.compile:  {args.compile}")
    if args.compile:
        print(f"  Compile mode:   {args.compile_mode}")

    # Device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        compute_cap = torch.cuda.get_device_capability()
        print(f"\nDevice: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"   Compute capability: {compute_cap[0]}.{compute_cap[1]}")

        # Enable TF32 for faster matmuls on Ampere+ (compute >= 8.0)
        if compute_cap[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("   TF32 enabled (Ampere+ GPU detected)")
        else:
            print("   TF32 not available (requires Ampere+ GPU)")

        # Enable cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # GPU-specific recommendations
        if 'A100' in gpu_name:
            print("   A100 detected - optimal for training!")
            if args.batch_size < 16:
                print(f"   TIP: Consider increasing batch_size to 16+ for better A100 utilization")
        elif 'V100' in gpu_name:
            print("   V100 detected - good for training")
        elif 'T4' in gpu_name:
            print("   T4 detected - will work but slower than A100/V100")
            print("   TIP: T4 is optimized for inference; expect ~3-5x slower than A100")
            if args.batch_size > 8:
                print(f"   TIP: T4 has 16GB VRAM - may need smaller batch_size if OOM")
        elif 'L4' in gpu_name:
            print("   L4 detected - good balance of speed and availability")
    else:
        device = torch.device('cpu')
        print(f"\nDevice: CPU")
        print("   WARNING: Training on CPU will be very slow!")

    # Load base model
    model, config = load_base_model(args.base_model, device)

    # Apply LoRA
    print("\n[Applying LoRA]")
    model = apply_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        targets=args.lora_targets
    )

    # torch.compile() for speed
    if args.compile and hasattr(torch, 'compile'):
        # Check PyTorch version for compile support
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 0):
            print(f"\n[Compiling] Skipped - requires PyTorch 2.0+ (you have {torch.__version__})")
        else:
            # Use 'default' mode on T4 (more stable), 'reduce-overhead' on newer GPUs
            compile_mode = args.compile_mode
            if device.type == 'cuda':
                compute_cap = torch.cuda.get_device_capability()
                if compute_cap[0] < 8 and compile_mode == 'reduce-overhead':
                    print(f"\n[Compiling] Using 'default' mode (more stable on older GPUs)")
                    compile_mode = 'default'
                else:
                    print(f"\n[Compiling] Using torch.compile(mode='{compile_mode}')")

            try:
                model = torch.compile(model, mode=compile_mode)
                print("   Model compiled successfully")
                print("   Note: First few steps will be slower due to compilation")
            except Exception as e:
                print(f"   Compilation failed: {e}")
                print("   Continuing without compilation (still fast with other optimizations)")

    # Load instruction data with optimized dataloader
    get_batch, steps_per_epoch, data_metadata = create_optimized_dataloader(
        args.instruction_data,
        args.metadata_file,
        seq_len=config['max_seq_len'],
        batch_size=args.batch_size,
        device=device
    )

    # Cap steps per epoch for instruction tuning (we don't need to see every example)
    # This helps prevent overfitting
    max_steps = min(steps_per_epoch, 5000)
    if steps_per_epoch > max_steps:
        print(f"\n[Note] Capping steps/epoch from {steps_per_epoch:,} to {max_steps:,}")
        steps_per_epoch = max_steps

    # Optimizer (only LoRA parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n[Optimizer] Trainable parameters: {num_trainable:,}")

    # Use fused AdamW on GPU if available (PyTorch 2.0+)
    use_fused = device.type == 'cuda' and hasattr(torch.optim.AdamW, 'fused')
    try:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=0.01,
            fused=use_fused
        )
        if use_fused:
            print("   Using fused AdamW optimizer")
    except TypeError:
        # Fallback for older PyTorch versions
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=0.01
        )

    # Scheduler with warmup
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if args.use_amp and device.type == 'cuda' else None
    if scaler:
        print("\n[AMP] Using Automatic Mixed Precision (FP16)")

    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    print(f"\nTotal steps: {total_steps:,} ({steps_per_epoch:,}/epoch x {args.epochs} epochs)")
    print(f"Warmup steps: {warmup_steps:,}")

    global_step = 0
    best_loss = float('inf')
    all_step_times = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")
        epoch_start = time.time()

        avg_loss, global_step, avg_step_time = train_epoch(
            model, get_batch, steps_per_epoch,
            optimizer, scheduler, scaler, criterion,
            device, args, epoch, global_step
        )
        all_step_times.append(avg_step_time)

        epoch_time = time.time() - epoch_start
        ppl = math.exp(min(avg_loss, 20))

        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")
        print(f"    Time: {epoch_time/60:.1f}m | Avg step: {avg_step_time*1000:.1f}ms")

        # Save checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            print(f"\n  Saving checkpoint...")
            save_model(model, config, args.output_dir, args, epoch=epoch,
                      lora_only=args.save_lora_only)

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Final save
    print("\n" + "="*70)
    print("SAVING FINAL MODEL")
    print("="*70)

    # Get the original model if compiled
    model_to_save = model
    if hasattr(model, '_orig_mod'):
        model_to_save = model._orig_mod

    if args.merge_weights:
        print("\nMerging LoRA weights into base model...")
        model_to_save = merge_lora_weights(model_to_save)

    save_model(model_to_save, config, args.output_dir, args,
               lora_only=args.save_lora_only and not args.merge_weights)

    # Copy tokenizer
    tokenizer_src = os.path.join(args.base_model, 'tokenizer.json')
    if os.path.exists(tokenizer_src):
        import shutil
        tokenizer_dst = os.path.join(args.output_dir, 'tokenizer.json')
        shutil.copy(tokenizer_src, tokenizer_dst)
        print(f"\nCopied tokenizer to: {tokenizer_dst}")

    # Final summary
    avg_overall_step_time = sum(all_step_times) / len(all_step_times) if all_step_times else 0
    tokens_per_step = args.batch_size * config['max_seq_len']
    tokens_per_sec = tokens_per_step / avg_overall_step_time if avg_overall_step_time > 0 else 0

    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE")
    print("="*70)
    print(f"\nResults:")
    print(f"  Best loss:        {best_loss:.4f} (PPL: {math.exp(min(best_loss, 20)):.2f})")
    print(f"  Avg step time:    {avg_overall_step_time*1000:.1f}ms")
    print(f"  Throughput:       {tokens_per_sec/1000:.1f}K tokens/sec")
    print(f"\nModel saved to: {args.output_dir}")
    print(f"\nTo test the model:")
    print(f"  py generate_instruct.py --model {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
