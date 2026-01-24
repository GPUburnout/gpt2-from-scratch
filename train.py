"""
Training Script for Language Model
Trains the Transformer model on text data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import time
from datetime import datetime
import math

from model import create_small_transformer
from tokenizer import CharacterTokenizer


class TextDataset(Dataset):
    """Dataset for language modeling"""

    def __init__(self, text, tokenizer, seq_len=256):
        """
        Initialize dataset

        Args:
            text: Input text string
            tokenizer: Tokenizer instance
            seq_len: Sequence length for training
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Encode entire text
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        print(f"Dataset size: {len(self.data):,} tokens")

    def __len__(self):
        # Number of sequences we can create
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        """
        Get a training sample
        Returns input and target sequences
        """
        # Get sequence
        chunk = self.data[idx:idx + self.seq_len + 1]

        # Input and target (shifted by 1)
        x = chunk[:-1]
        y = chunk[1:]

        return x, y


def create_dataloaders(text, tokenizer, seq_len=256, batch_size=32, train_split=0.9):
    """Create train and validation dataloaders"""

    # Split data
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    print(f"\nCreating datasets...")
    print(f"Training text: {len(train_text):,} characters")
    print(f"Validation text: {len(val_text):,} characters")

    # Create datasets
    train_dataset = TextDataset(train_text, tokenizer, seq_len)
    val_dataset = TextDataset(val_text, tokenizer, seq_len)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(x)

        # Calculate loss
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Print progress
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Time: {elapsed:.1f}s")

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir='checkpoints'):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    filepath = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def save_final_model(model, model_dir='models/tiny_lm_v1', metadata=None):
    """Save the final trained model with metadata"""
    os.makedirs(model_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

    # Save configuration
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(model.get_config(), f, indent=2)
    print(f"Config saved: {config_path}")

    # Save metadata
    if metadata:
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved: {metadata_path}")

    return model_path


def main():
    """Main training function"""
    print("\n" + "="*80)
    print("TRAINING TINY LANGUAGE MODEL")
    print("="*80)

    # Configuration
    CONFIG = {
        'seq_len': 256,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 20,
        'train_split': 0.9,
        'checkpoint_freq': 5,  # Save checkpoint every N epochs
    }

    print("\nTraining configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load dataset
    print("\n" + "-"*80)
    print("Loading dataset...")
    print("-"*80)
    dataset_file = 'data/tiny_shakespeare.txt'
    with open(dataset_file, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = CharacterTokenizer()
    tokenizer.load('models/tokenizer.json')

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        text, tokenizer,
        seq_len=CONFIG['seq_len'],
        batch_size=CONFIG['batch_size'],
        train_split=CONFIG['train_split']
    )

    # Create model
    print("\n" + "-"*80)
    print("Creating model...")
    print("-"*80)
    model = create_small_transformer(tokenizer.vocab_size)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: ~{model.count_parameters() * 4 / 1024 / 1024:.1f} MB")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    training_start = time.time()
    best_val_loss = float('inf')
    training_history = []

    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Calculate perplexity
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Perplexity: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_perplexity': train_ppl,
            'val_perplexity': val_ppl
        })

        # Save checkpoint
        if epoch % CONFIG['checkpoint_freq'] == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best validation loss!")

    # Training complete
    training_time = time.time() - training_start
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total training time: {training_time/60:.1f} minutes")
    print(f"Final train loss: {train_loss:.4f}")
    print(f"Final val loss: {val_loss:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")

    # Save final model
    print("\n" + "-"*80)
    print("Saving final model...")
    print("-"*80)

    metadata = {
        'model_name': 'tiny_lm_v1',
        'version': '1.0',
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'architecture': model.get_config(),
        'dataset': {
            'name': 'tiny_shakespeare',
            'size': f'{len(text):,} characters'
        },
        'training': {
            'epochs': CONFIG['epochs'],
            'batch_size': CONFIG['batch_size'],
            'seq_len': CONFIG['seq_len'],
            'learning_rate': CONFIG['learning_rate'],
            'duration_minutes': round(training_time / 60, 1),
            'device': str(device),
            'started_from': 'scratch'
        },
        'performance': {
            'final_train_loss': round(train_loss, 4),
            'final_val_loss': round(val_loss, 4),
            'best_val_loss': round(best_val_loss, 4),
            'final_train_perplexity': round(train_ppl, 2),
            'final_val_perplexity': round(val_ppl, 2)
        },
        'training_history': training_history
    }

    save_final_model(model, 'models/tiny_lm_v1', metadata)

    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"\nYour first language model is ready!")
    print(f"Model saved to: models/tiny_lm_v1/")
    print(f"\nNext step: Generate text with your model!")
    print("Run: python generate.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
