"""
BPE Tokenizer Wrapper
=====================
Wraps HuggingFace `tokenizers` library to provide the same interface
as CharacterTokenizer. Uses byte-level BPE (GPT-2 style).

Requires: pip install tokenizers
"""

import json
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class BPETokenizer:
    """Byte-level BPE tokenizer compatible with CharacterTokenizer interface."""

    def __init__(self):
        self.tokenizer = None
        self._vocab_size = 0

    def build_vocab_from_file(self, filepath, vocab_size=32000,
                              min_frequency=2, chunk_size=None):
        """Train BPE tokenizer on a text file.

        Args:
            filepath: Path to text file
            vocab_size: Target vocabulary size (default: 32000)
            min_frequency: Minimum token frequency (default: 2)
            chunk_size: Unused, kept for interface compatibility
        """
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
            show_progress=True
        )

        file_size = os.path.getsize(filepath)
        print(f"\nTraining BPE tokenizer on: {filepath}")
        print(f"File size: {file_size / (1024**3):.2f} GB")
        print(f"Target vocab size: {vocab_size:,}")
        print(f"Min frequency: {min_frequency}")

        tokenizer.train(files=[filepath], trainer=trainer)

        self.tokenizer = tokenizer
        self._vocab_size = tokenizer.get_vocab_size()

        print(f"\nBPE vocabulary built: {self._vocab_size:,} tokens")
        # Show some sample tokens
        vocab = tokenizer.get_vocab()
        sample = sorted(vocab.items(), key=lambda x: x[1])[:20]
        sample_str = ', '.join(f"'{k}'" for k, v in sample)
        print(f"Sample tokens: {sample_str}")

        return self._vocab_size

    def encode(self, text):
        """Encode text to list of token IDs.

        Args:
            text: Input string

        Returns:
            List of integer token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. "
                             "Call build_vocab_from_file() or load() first.")
        return self.tokenizer.encode(text).ids

    def decode(self, tokens):
        """Decode token IDs back to text.

        Args:
            tokens: List of integer token IDs

        Returns:
            Decoded string
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self):
        """Number of tokens in vocabulary."""
        return self._vocab_size

    def save(self, filepath):
        """Save tokenizer to a JSON file.

        Args:
            filepath: Path to save tokenizer (e.g. 'bpe_tokenizer.json')
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")

        self.tokenizer.save(filepath)
        print(f"\nBPE tokenizer saved to: {filepath}")

    def load(self, filepath):
        """Load tokenizer from a JSON file.

        Args:
            filepath: Path to tokenizer JSON file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer file not found: {filepath}")

        self.tokenizer = Tokenizer.from_file(filepath)
        self._vocab_size = self.tokenizer.get_vocab_size()

        print(f"BPE tokenizer loaded: {self._vocab_size:,} tokens")
        return self
