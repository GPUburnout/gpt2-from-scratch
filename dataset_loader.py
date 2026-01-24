"""
Dataset Loader for Language Model Training
Downloads and preprocesses datasets from Hugging Face
"""

import json
import os
import requests
from datasets import load_dataset


class DatasetLoader:
    """Load and preprocess datasets for language model training"""

    def __init__(self, selection_file='dataset_selection.json'):
        """Initialize with dataset selection file"""
        self.selection_file = selection_file
        self.dataset_info = None
        self.text_data = None

    def load_selection(self):
        """Load dataset selection from JSON file"""
        if not os.path.exists(self.selection_file):
            raise FileNotFoundError(
                f"Dataset selection file not found: {self.selection_file}\n"
                "Please run dataset_browser.py first to select a dataset."
            )

        with open(self.selection_file, 'r') as f:
            self.dataset_info = json.load(f)

        print(f"\nLoaded dataset selection: {self.dataset_info['name']}")
        print(f"Description: {self.dataset_info['description']}")
        return self.dataset_info

    def download_tiny_shakespeare(self):
        """Download tiny_shakespeare directly from raw text URL"""
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading from: {url}")

        response = requests.get(url)
        response.raise_for_status()

        text = response.text
        print(f"Downloaded {len(text):,} characters")
        return text

    def download_dataset(self):
        """Download dataset from Hugging Face or direct URL"""
        print(f"\nDownloading dataset: {self.dataset_info['name']}...")

        try:
            # Special handling for tiny_shakespeare (old dataset script format)
            if self.dataset_info['name'] == 'tiny_shakespeare':
                print("Using direct download for tiny_shakespeare...")
                self.text_data = self.download_tiny_shakespeare()
                print("Download complete!")
                return None  # No dataset object, text is already loaded

            # Standard Hugging Face datasets
            if self.dataset_info['config']:
                dataset = load_dataset(
                    self.dataset_info['name'],
                    self.dataset_info['config']
                )
            else:
                dataset = load_dataset(self.dataset_info['name'])

            print("Download complete!")
            return dataset

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise

    def extract_text(self, dataset):
        """Extract text from dataset"""
        print("\nExtracting text from dataset...")

        all_text = []

        # Handle different dataset structures
        # Most text datasets have 'train', 'validation', 'test' splits
        for split_name in dataset.keys():
            split = dataset[split_name]
            print(f"Processing {split_name} split ({len(split)} examples)...")

            # Try common text field names
            text_field = None
            for field in ['text', 'content', 'sentence', 'document']:
                if field in split.column_names:
                    text_field = field
                    break

            if text_field is None:
                print(f"Available fields: {split.column_names}")
                text_field = split.column_names[0]
                print(f"Using first field: {text_field}")

            # Extract text
            for example in split:
                text = example[text_field]
                if isinstance(text, str):
                    all_text.append(text)
                elif isinstance(text, list):
                    all_text.extend([str(t) for t in text])

        # Join all text
        self.text_data = '\n'.join(all_text)
        print(f"\nTotal text length: {len(self.text_data):,} characters")

        return self.text_data

    def save_to_file(self, output_dir='data'):
        """Save preprocessed text to file"""
        os.makedirs(output_dir, exist_ok=True)

        # Create filename from dataset name
        dataset_name = self.dataset_info['name'].replace('/', '_')
        if self.dataset_info['config']:
            config_name = self.dataset_info['config'].replace('/', '_')
            filename = f"{dataset_name}_{config_name}.txt"
        else:
            filename = f"{dataset_name}.txt"

        filepath = os.path.join(output_dir, filename)

        print(f"\nSaving to: {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.text_data)

        print(f"Saved {len(self.text_data):,} characters to {filepath}")

        # Save metadata
        metadata_file = filepath.replace('.txt', '_metadata.json')
        metadata = {
            **self.dataset_info,
            'text_length': len(self.text_data),
            'filepath': filepath
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {metadata_file}")

        return filepath

    def get_text_stats(self):
        """Print statistics about the text data"""
        if self.text_data is None:
            print("No text data loaded yet.")
            return

        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        print(f"Total characters: {len(self.text_data):,}")
        print(f"Total lines: {self.text_data.count(chr(10)):,}")

        # Character vocabulary
        unique_chars = sorted(set(self.text_data))
        print(f"Unique characters: {len(unique_chars)}")
        print(f"Character set: {''.join(unique_chars[:50])}" + ("..." if len(unique_chars) > 50 else ""))

        # Word count (approximate)
        word_count = len(self.text_data.split())
        print(f"Approximate word count: {word_count:,}")

        print("="*80)

    def run(self):
        """Run the complete dataset loading pipeline"""
        print("\n" + "="*80)
        print("DATASET LOADER - Preparing data for training")
        print("="*80)

        # Load selection
        self.load_selection()

        # Download dataset
        dataset = self.download_dataset()

        # Extract text (skip if already loaded directly)
        if self.text_data is None:
            self.extract_text(dataset)

        # Show statistics
        self.get_text_stats()

        # Save to file
        filepath = self.save_to_file()

        print("\n" + "="*80)
        print("DATASET PREPARATION COMPLETE")
        print("="*80)
        print(f"\nDataset ready for training!")
        print(f"Text file: {filepath}")
        print(f"\nNext step: Build the tokenizer")
        print("="*80 + "\n")

        return filepath


def main():
    """Main function to run dataset loader"""
    loader = DatasetLoader()
    loader.run()


if __name__ == "__main__":
    main()
