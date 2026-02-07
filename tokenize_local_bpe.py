"""
Pre-Tokenize Dataset with BPE for Memory-Mapped Training
==========================================================
Converts a large text file into a binary int32 token file (.bin)
using BPE tokenization for efficient training on Google Colab.

BPE produces ~4-6x fewer tokens than character-level tokenization,
resulting in faster training and better model quality.

Usage:
    py tokenize_local_bpe.py --input datasets_cache/dataset_chatgpt_style_10GB.txt --compress
    py tokenize_local_bpe.py --input datasets_cache/dataset_chatgpt_style_10GB.txt --vocab_size 32000 --clean --compress

Output:
    - tokens_bpe.bin              (~8-12GB for 12GB text, int32)
    - tokens_bpe.bin.gz           (~3-4GB compressed, for upload)
    - bpe_tokenizer.json          (BPE vocabulary and merge rules)
    - tokens_bpe_metadata.json    (token count, vocab size, split info)
"""

import argparse
import gzip
import json
import numpy as np
import os
import time
from datetime import datetime

from tokenizer_bpe import BPETokenizer
from dataset_cleaner import DatasetCleaner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pre-tokenize dataset with BPE to binary for memory-mapped training')

    parser.add_argument('--input', type=str, required=True,
                        help='Path to raw text file')
    parser.add_argument('--output_dir', type=str, default='datasets_cache',
                        help='Output directory (default: datasets_cache)')
    parser.add_argument('--output_name', type=str, default='tokens_bpe',
                        help='Base name for output files (default: tokens_bpe)')

    # BPE parameters
    parser.add_argument('--vocab_size', type=int, default=32000,
                        help='BPE vocabulary size (default: 32000)')
    parser.add_argument('--min_frequency', type=int, default=2,
                        help='Minimum token frequency for BPE (default: 2)')

    # Shared parameters
    parser.add_argument('--clean', action='store_true',
                        help='Clean dataset before tokenizing')
    parser.add_argument('--cleaning_mode', type=str, default='balanced',
                        choices=['strict', 'balanced', 'lenient'],
                        help='Cleaning mode (default: balanced)')
    parser.add_argument('--chunk_size_mb', type=int, default=100,
                        help='Processing chunk size in MB (default: 100)')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Train/val split ratio (default: 0.9)')
    parser.add_argument('--compress', action='store_true',
                        help='Compress output .bin file with gzip')

    return parser.parse_args()


def clean_dataset_streaming(input_path, output_path, cleaning_mode, chunk_size):
    """Clean a large text file in streaming fashion."""
    cleaner = DatasetCleaner(mode=cleaning_mode)
    total_in = 0
    total_out = 0
    input_size = os.path.getsize(input_path)

    print(f"\nCleaning dataset with mode: {cleaning_mode}")
    print(f"Input: {input_path} ({input_size / (1024**3):.2f} GB)")

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break

            total_in += len(chunk)
            cleaned = cleaner.clean_text(chunk, show_stats=False)
            total_out += len(cleaned)
            f_out.write(cleaned)

            progress_gb = total_in / (1024**3)
            pct = (total_in / (input_size * 0.9)) * 100
            if pct <= 100:
                print(f"  Cleaning: {progress_gb:.2f} GB ({pct:.1f}%)", end='\r')

    reduction = (1 - total_out / total_in) * 100 if total_in > 0 else 0
    print(f"\n  Done: {total_in:,} -> {total_out:,} chars ({reduction:.1f}% removed)")

    return total_out


def encode_to_binary(tokenizer, source_path, output_bin_path, chunk_size):
    """Encode text file to int32 binary using BPE, streaming.

    Args:
        tokenizer: BPETokenizer with trained vocab
        source_path: Path to text file
        output_bin_path: Path for output .bin file
        chunk_size: Read chunk size in bytes

    Returns:
        Total number of tokens written
    """
    total_tokens = 0
    source_size = os.path.getsize(source_path)

    print(f"\nEncoding to binary (int32): {output_bin_path}")
    print(f"Source: {source_path} ({source_size / (1024**3):.2f} GB)")

    with open(source_path, 'r', encoding='utf-8') as f_in, \
         open(output_bin_path, 'wb') as f_out:
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break

            token_ids = tokenizer.encode(chunk)
            arr = np.array(token_ids, dtype=np.int32)
            arr.tofile(f_out)

            total_tokens += len(token_ids)
            bin_size_gb = (total_tokens * 4) / (1024**3)
            print(f"  Encoded: {total_tokens:,} tokens ({bin_size_gb:.2f} GB)", end='\r')

    print(f"\n  Done: {total_tokens:,} tokens total")
    return total_tokens


def compress_bin(bin_path):
    """Compress .bin file with gzip."""
    gz_path = bin_path + '.gz'
    input_size = os.path.getsize(bin_path)
    bytes_processed = 0

    print(f"\nCompressing: {bin_path}")
    print(f"Input size: {input_size / (1024**3):.2f} GB")

    with open(bin_path, 'rb') as f_in:
        with gzip.open(gz_path, 'wb', compresslevel=6) as f_out:
            while True:
                chunk = f_in.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)
                bytes_processed += len(chunk)

                if bytes_processed % (100 * 1024 * 1024) == 0:
                    pct = (bytes_processed / input_size) * 100
                    print(f"  Compressing: {pct:.1f}%", end='\r')

    output_size = os.path.getsize(gz_path)
    ratio = (1 - output_size / input_size) * 100
    print(f"\n  Done: {input_size / (1024**3):.2f} GB -> {output_size / (1024**3):.2f} GB ({ratio:.1f}% reduction)")

    return gz_path


def main():
    args = parse_args()
    start_time = time.time()

    chunk_size = args.chunk_size_mb * 1024 * 1024
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("BPE PRE-TOKENIZATION FOR MEMORY-MAPPED TRAINING")
    print("=" * 80)

    input_size = os.path.getsize(args.input)
    print(f"\nInput file: {args.input}")
    print(f"Input size: {input_size / (1024**3):.2f} GB")
    print(f"BPE vocab size: {args.vocab_size:,}")
    print(f"Min frequency: {args.min_frequency}")
    print(f"Chunk size: {args.chunk_size_mb} MB")
    print(f"Clean: {args.clean} (mode: {args.cleaning_mode})")
    print(f"Train split: {args.train_split}")

    # --- Step 1: Optional cleaning ---
    if args.clean:
        print("\n" + "-" * 80)
        cleaned_path = os.path.join(args.output_dir, 'cleaned_tmp.txt')
        clean_dataset_streaming(args.input, cleaned_path, args.cleaning_mode, chunk_size)
        source_path = cleaned_path
    else:
        source_path = args.input

    # --- Step 2: Train BPE vocabulary ---
    print("\n" + "-" * 80)
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.build_vocab_from_file(
        source_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )

    print(f"\nBPE vocab size: {tokenizer.vocab_size:,} (uses int32)")

    # --- Step 3: Save tokenizer ---
    tokenizer_path = os.path.join(args.output_dir, 'bpe_tokenizer.json')
    tokenizer.save(tokenizer_path)

    # --- Step 4: Encode to binary ---
    print("\n" + "-" * 80)
    bin_path = os.path.join(args.output_dir, f'{args.output_name}.bin')
    total_tokens = encode_to_binary(tokenizer, source_path, bin_path, chunk_size)

    # --- Step 5: Save metadata ---
    train_split_index = int(total_tokens * args.train_split)
    metadata = {
        'total_tokens': total_tokens,
        'vocab_size': tokenizer.vocab_size,
        'tokenizer_type': 'bpe',
        'dtype': 'int32',
        'bytes_per_token': 4,
        'train_split': args.train_split,
        'train_split_index': train_split_index,
        'train_tokens': train_split_index,
        'val_tokens': total_tokens - train_split_index,
        'source_file': os.path.basename(args.input),
        'source_size_bytes': input_size,
        'bin_file': f'{args.output_name}.bin',
        'tokenizer_file': 'bpe_tokenizer.json',
        'cleaned': args.clean,
        'cleaning_mode': args.cleaning_mode if args.clean else None,
        'bpe_vocab_size_requested': args.vocab_size,
        'bpe_min_frequency': args.min_frequency,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_path = os.path.join(args.output_dir, f'{args.output_name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # --- Step 6: Compress (optional) ---
    gz_path = None
    if args.compress:
        print("\n" + "-" * 80)
        gz_path = compress_bin(bin_path)

    # --- Step 7: Cleanup temp files ---
    if args.clean and os.path.exists(source_path) and source_path != args.input:
        print(f"\nRemoving temporary cleaned file: {source_path}")
        os.remove(source_path)

    # --- Summary ---
    elapsed = time.time() - start_time
    bin_size = os.path.getsize(bin_path)

    print("\n" + "=" * 80)
    print("BPE PRE-TOKENIZATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal tokens:  {total_tokens:,}")
    print(f"Vocab size:    {tokenizer.vocab_size:,}")
    print(f"Token type:    BPE (int32, 4 bytes/token)")
    print(f"Train tokens:  {train_split_index:,} ({args.train_split * 100:.0f}%)")
    print(f"Val tokens:    {total_tokens - train_split_index:,} ({(1 - args.train_split) * 100:.0f}%)")
    print(f"Binary file:   {bin_path} ({bin_size / (1024**3):.2f} GB)")
    if gz_path:
        gz_size = os.path.getsize(gz_path)
        print(f"Compressed:    {gz_path} ({gz_size / (1024**3):.2f} GB)")
    print(f"Tokenizer:     {tokenizer_path}")
    print(f"Metadata:      {metadata_path}")
    print(f"Time elapsed:  {elapsed / 60:.1f} minutes")

    # Compression ratio vs character tokenization
    char_tokens_est = input_size  # ~1 byte per char
    if total_tokens > 0:
        ratio = char_tokens_est / total_tokens
        print(f"\nBPE compression: ~{ratio:.1f}x fewer tokens than character-level")

    print("\n" + "-" * 80)
    print("UPLOAD THESE FILES TO GOOGLE DRIVE:")
    print("-" * 80)
    if gz_path:
        print(f"  1. {gz_path}")
    else:
        print(f"  1. {bin_path}")
    print(f"  2. {tokenizer_path}")
    print(f"  3. {metadata_path}")
    print("\nPlace them in: Google Drive > LLM_Training > datasets/")
    print("=" * 80)


if __name__ == '__main__':
    main()
