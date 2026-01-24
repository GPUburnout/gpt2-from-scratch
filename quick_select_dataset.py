"""
Quick Dataset Selection - Automatically selects tiny_shakespeare
This is a helper script to quickly get started without interactive input
"""

import json
from datetime import datetime


def select_tiny_shakespeare():
    """Select tiny_shakespeare dataset (recommended for first run)"""

    selection = {
        'name': 'tiny_shakespeare',
        'config': None,
        'description': 'Complete works of Shakespeare, ~1M characters. Classic tiny dataset.',
        'size': 'Tiny (~1 MB)',
        'date_selected': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'purpose': 'Training tiny language model - First run with Claude Code',
        'from_where': 'https://huggingface.co/datasets/tiny_shakespeare'
    }

    with open('dataset_selection.json', 'w') as f:
        json.dump(selection, f, indent=2)

    print("\n" + "="*80)
    print("DATASET SELECTED: tiny_shakespeare")
    print("="*80)
    print("\nDataset: tiny_shakespeare")
    print("Description: Complete works of Shakespeare, ~1M characters")
    print("Size: Tiny (~1 MB)")
    print("Perfect for: First training run on CPU")
    print("\nSelection saved to dataset_selection.json")
    print("="*80 + "\n")

    return selection


if __name__ == "__main__":
    select_tiny_shakespeare()
