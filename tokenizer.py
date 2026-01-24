"""
Tokenizer for Language Model
Converts text to numbers (tokens) and back
"""

import json
import os


class CharacterTokenizer:
    """Simple character-level tokenizer for tiny language models"""

    def __init__(self):
        """Initialize tokenizer"""
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        """Build vocabulary from text"""
        print("\nBuilding character vocabulary...")

        # Get unique characters and sort them
        chars = sorted(set(text))
        self.vocab_size = len(chars)

        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        print(f"Vocabulary size: {self.vocab_size} characters")
        print(f"Characters: {''.join(chars[:50])}" + ("..." if len(chars) > 50 else ""))

        return self.vocab_size

    def encode(self, text):
        """Convert text to list of token IDs"""
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(self, tokens):
        """Convert list of token IDs back to text"""
        return ''.join([self.idx_to_char[idx] for idx in tokens if idx in self.idx_to_char])

    def save(self, filepath='models/tokenizer.json'):
        """Save tokenizer to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        tokenizer_data = {
            'type': 'character',
            'vocab_size': self.vocab_size,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()}
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)

        print(f"\nTokenizer saved to: {filepath}")
        return filepath

    def load(self, filepath='models/tokenizer.json'):
        """Load tokenizer from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        self.vocab_size = tokenizer_data['vocab_size']
        self.char_to_idx = tokenizer_data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in tokenizer_data['idx_to_char'].items()}

        print(f"\nTokenizer loaded from: {filepath}")
        print(f"Vocabulary size: {self.vocab_size}")
        return self

    def get_stats(self):
        """Print tokenizer statistics"""
        print("\n" + "="*80)
        print("TOKENIZER STATISTICS")
        print("="*80)
        print(f"Type: Character-level")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample characters: {list(self.char_to_idx.keys())[:20]}")
        print("="*80)


def main():
    """Main function to build and test tokenizer"""
    print("\n" + "="*80)
    print("TOKENIZER BUILDER")
    print("="*80)

    # Load dataset
    dataset_file = 'data/tiny_shakespeare.txt'
    if not os.path.exists(dataset_file):
        print(f"\nError: Dataset not found at {dataset_file}")
        print("Please run dataset_loader.py first.")
        return

    print(f"\nLoading text from: {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters")

    # Build tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(text)

    # Test tokenizer
    print("\n" + "="*80)
    print("TESTING TOKENIZER")
    print("="*80)

    test_text = "Hello, World!"
    print(f"\nOriginal text: {test_text}")

    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    if test_text == decoded:
        print("Test passed!")
    else:
        print("Test failed!")

    # Test with Shakespeare sample
    shakespeare_sample = text[:100]
    print(f"\nShakespeare sample: {shakespeare_sample}")
    encoded_sample = tokenizer.encode(shakespeare_sample)
    print(f"Encoded (first 20 tokens): {encoded_sample[:20]}")
    decoded_sample = tokenizer.decode(encoded_sample)
    assert shakespeare_sample == decoded_sample, "Encoding/decoding mismatch!"
    print("Shakespeare encoding test passed!")

    # Show statistics
    tokenizer.get_stats()

    # Save tokenizer
    tokenizer.save()

    print("\n" + "="*80)
    print("TOKENIZER BUILD COMPLETE")
    print("="*80)
    print(f"\nTokenizer ready for model training!")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Saved to: models/tokenizer.json")
    print(f"\nNext step: Build the model architecture")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
