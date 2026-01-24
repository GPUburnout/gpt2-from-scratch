"""
Dataset Browser for Language Model Training
Browses Hugging Face datasets and helps select appropriate text datasets
"""

import sys
import json
from datetime import datetime

class DatasetBrowser:
    """Browse and select datasets from Hugging Face"""

    # Curated list of good text datasets for language modeling
    RECOMMENDED_DATASETS = [
        {
            'name': 'wikitext',
            'config': 'wikitext-2-raw-v1',
            'description': 'WikiText-2: Clean Wikipedia text, ~2M tokens. Good for learning.',
            'size': 'Small (~4 MB)',
            'recommended': True,
            'reason': 'Perfect size for CPU training, clean English text'
        },
        {
            'name': 'tiny_shakespeare',
            'config': None,
            'description': 'Complete works of Shakespeare, ~1M characters. Classic tiny dataset.',
            'size': 'Tiny (~1 MB)',
            'recommended': True,
            'reason': 'Very small, trains fast on CPU, fun to generate Shakespeare-like text'
        },
        {
            'name': 'wikitext',
            'config': 'wikitext-103-raw-v1',
            'description': 'WikiText-103: Larger Wikipedia text, ~100M tokens.',
            'size': 'Medium (~181 MB)',
            'recommended': False,
            'reason': 'Better for GPU training, may be slow on CPU'
        },
        {
            'name': 'bookcorpus',
            'config': None,
            'description': 'Books corpus used to train BERT, ~800M words.',
            'size': 'Large (~4.6 GB)',
            'recommended': False,
            'reason': 'Too large for our tiny model, requires GPU'
        },
        {
            'name': 'openwebtext',
            'config': None,
            'description': 'Web text similar to GPT-2 training data, ~8M documents.',
            'size': 'Very Large (~12 GB)',
            'recommended': False,
            'reason': 'Too large for beginners, definitely needs GPU'
        },
        {
            'name': 'ptb_text_only',
            'config': None,
            'description': 'Penn Treebank: Classic benchmark, ~1M words.',
            'size': 'Small (~5 MB)',
            'recommended': True,
            'reason': 'Standard benchmark for language models, good for CPU'
        }
    ]

    def display_datasets(self):
        """Display available datasets with details"""
        print("\n" + "="*80)
        print("AVAILABLE TEXT DATASETS FOR LANGUAGE MODEL TRAINING")
        print("="*80 + "\n")

        recommended_count = 0
        for idx, dataset in enumerate(self.RECOMMENDED_DATASETS, 1):
            marker = "** RECOMMENDED **" if dataset['recommended'] else "  "

            print(f"{idx}. {dataset['name']}" + (f" ({dataset['config']})" if dataset['config'] else ""))
            print(f"   {marker}")
            print(f"   Description: {dataset['description']}")
            print(f"   Size: {dataset['size']}")
            if dataset['recommended']:
                print(f"   Why recommended: {dataset['reason']}")
                recommended_count += 1
            print()

        print("="*80)
        print(f"\n{recommended_count} datasets marked as RECOMMENDED for our tiny language model.\n")

    def get_my_recommendation(self):
        """Get Claude's top recommendation"""
        print("\nMY TOP RECOMMENDATION:")
        print("-"*80)
        print("\nFor your first tiny language model, I recommend:")
        print("\n  Dataset: tiny_shakespeare")
        print("  Reason:")
        print("    - Very small and fast to train on CPU")
        print("    - Character-level works great (simple tokenization)")
        print("    - Fun results - generates Shakespeare-style text")
        print("    - You'll see results in minutes, not hours")
        print("\n  Alternative: wikitext-2 if you prefer modern English")
        print("-"*80 + "\n")

    def get_user_choice(self):
        """Get user's dataset selection"""
        while True:
            try:
                choice = input("Enter the number of your chosen dataset (or 'q' to quit): ").strip()

                if choice.lower() == 'q':
                    print("Exiting dataset browser.")
                    return None

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(self.RECOMMENDED_DATASETS):
                    selected = self.RECOMMENDED_DATASETS[choice_idx]
                    return selected
                else:
                    print(f"Please enter a number between 1 and {len(self.RECOMMENDED_DATASETS)}")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def confirm_selection(self, dataset):
        """Confirm and display selected dataset"""
        print("\n" + "="*80)
        print("SELECTED DATASET")
        print("="*80)
        print(f"\nDataset: {dataset['name']}")
        if dataset['config']:
            print(f"Config: {dataset['config']}")
        print(f"Description: {dataset['description']}")
        print(f"Size: {dataset['size']}")
        print("\nThis selection has been saved for the next step (tokenization).")
        print("="*80 + "\n")

        # Construct Hugging Face dataset URL
        dataset_url = f"https://huggingface.co/datasets/{dataset['name']}"

        # Create comprehensive metadata
        selection_metadata = {
            'name': dataset['name'],
            'config': dataset['config'],
            'description': dataset['description'],
            'size': dataset['size'],
            'date_selected': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'purpose': 'Training tiny language model - First run with Claude Code',
            'from_where': dataset_url
        }

        return selection_metadata

    def run(self):
        """Run the dataset browser"""
        self.display_datasets()
        self.get_my_recommendation()

        selected = self.get_user_choice()
        if selected:
            return self.confirm_selection(selected)
        return None


def main():
    """Main function to run dataset browser"""
    print("\nWelcome to the Dataset Browser!")
    print("Let's find the perfect dataset for your tiny language model.\n")

    browser = DatasetBrowser()
    selection = browser.run()

    if selection:
        # Save selection to a file for next steps
        with open('dataset_selection.json', 'w') as f:
            json.dump(selection, f, indent=2)
        print(f"Dataset selection saved to dataset_selection.json")

    return selection


if __name__ == "__main__":
    main()
