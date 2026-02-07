"""
Transformer Language Model Architecture
Modern architecture (GPT-style) scalable from tiny to large
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism with Flash Attention support"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        # Q, K, V projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Check if Flash Attention is available (PyTorch 2.0+)
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')

        # Fallback dropout for non-flash path
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (batch, seq, 3*embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_flash:
            # Use PyTorch's scaled_dot_product_attention (Flash Attention when available)
            # This is 1.5-2x faster and more memory efficient
            dropout_p = self.dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,  # We use is_causal instead
                dropout_p=dropout_p,
                is_causal=True  # Causal mask for autoregressive generation
            )
        else:
            # Fallback to manual attention for older PyTorch versions
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Apply causal mask (for autoregressive generation)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            # Attention weights
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            # Apply attention to values
            out = torch.matmul(attn, v)

        # Reshape: (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)

        # Output projection
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block (attention + feed-forward)"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x


class TransformerLanguageModel(nn.Module):
    """
    GPT-style Transformer Language Model
    Scalable from tiny (CPU) to large (GPU cluster)
    """

    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4,
                 ff_dim=None, max_seq_len=256, dropout=0.1):
        """
        Initialize Transformer model

        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of Transformer blocks
            ff_dim: Feed-forward dimension (default: 4 * embed_dim)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        if ff_dim is None:
            ff_dim = 4 * embed_dim

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embeddings (learned)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

        # Output projection
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

        # Create causal mask
        self.register_buffer("causal_mask", self._create_causal_mask(max_seq_len))

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _create_causal_mask(self, seq_len):
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.view(1, 1, seq_len, seq_len)
        return mask

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len)

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Token embeddings
        token_emb = self.token_embedding(x)  # (batch, seq_len, embed_dim)

        # Positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.positional_embedding(positions)  # (1, seq_len, embed_dim)

        # Combine embeddings
        x = self.dropout_layer(token_emb + pos_emb)

        # Get causal mask for this sequence length
        mask = self.causal_mask[:, :, :seq_len, :seq_len]

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm
        x = self.ln_f(x)

        # Output logits
        logits = self.head(x)  # (batch, seq_len, vocab_size)

        return logits

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        """Get model configuration"""
        return {
            'model_type': 'Transformer',
            'architecture': 'GPT-style (decoder-only)',
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'total_parameters': self.count_parameters()
        }

    def save_config(self, filepath='models/model_config.json'):
        """Save model configuration"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        config = self.get_config()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model config saved to: {filepath}")
        return filepath


def create_tiny_transformer(vocab_size):
    """Create a tiny Transformer (fastest on CPU)"""
    return TransformerLanguageModel(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        max_seq_len=128,
        dropout=0.1
    )


def create_small_transformer(vocab_size):
    """Create a small Transformer (recommended for first run)"""
    return TransformerLanguageModel(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=256,
        dropout=0.1
    )


def create_medium_transformer(vocab_size):
    """Create a medium Transformer (GPU recommended)"""
    return TransformerLanguageModel(
        vocab_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=512,
        dropout=0.1
    )


def create_large_transformer(vocab_size):
    """Create a large Transformer (GPU cluster)"""
    return TransformerLanguageModel(
        vocab_size=vocab_size,
        embed_dim=1024,
        num_heads=16,
        num_layers=12,
        max_seq_len=1024,
        dropout=0.1
    )


def main():
    """Test model creation"""
    print("\n" + "="*80)
    print("TRANSFORMER MODEL ARCHITECTURE")
    print("="*80)

    # Load tokenizer to get vocab size
    tokenizer_path = 'models/tokenizer.json'
    if not os.path.exists(tokenizer_path):
        print(f"\nError: Tokenizer not found at {tokenizer_path}")
        print("Please run tokenizer.py first.")
        return

    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
        vocab_size = tokenizer_data['vocab_size']

    print(f"\nVocabulary size: {vocab_size}")
    print("Architecture: GPT-style Transformer (decoder-only)")

    # Create models of different sizes
    print("\n" + "-"*80)
    print("TINY TRANSFORMER (fastest on CPU)")
    print("-"*80)
    tiny_model = create_tiny_transformer(vocab_size)
    print(f"Parameters: {tiny_model.count_parameters():,}")
    print(f"Embed dim: {tiny_model.embed_dim}")
    print(f"Attention heads: {tiny_model.num_heads}")
    print(f"Layers: {tiny_model.num_layers}")
    print(f"Context length: {tiny_model.max_seq_len}")

    print("\n" + "-"*80)
    print("SMALL TRANSFORMER (recommended for first run)")
    print("-"*80)
    small_model = create_small_transformer(vocab_size)
    print(f"Parameters: {small_model.count_parameters():,}")
    print(f"Embed dim: {small_model.embed_dim}")
    print(f"Attention heads: {small_model.num_heads}")
    print(f"Layers: {small_model.num_layers}")
    print(f"Context length: {small_model.max_seq_len}")

    print("\n" + "-"*80)
    print("MEDIUM TRANSFORMER (GPU recommended)")
    print("-"*80)
    medium_model = create_medium_transformer(vocab_size)
    print(f"Parameters: {medium_model.count_parameters():,}")
    print(f"Embed dim: {medium_model.embed_dim}")
    print(f"Attention heads: {medium_model.num_heads}")
    print(f"Layers: {medium_model.num_layers}")
    print(f"Context length: {medium_model.max_seq_len}")

    # Use small model for our tiny LM
    print("\n" + "="*80)
    print("SELECTED MODEL: SMALL TRANSFORMER")
    print("="*80)
    print("Good balance for CPU training with modern architecture")
    model = small_model

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    seq_len = 32
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: (batch={batch_size}, seq_len={seq_len}, vocab={vocab_size})")
    assert logits.shape == (batch_size, seq_len, vocab_size), "Shape mismatch!"
    print("Forward pass test passed!")

    # Save configuration
    model.save_config()

    print("\n" + "="*80)
    print("MODEL CREATION COMPLETE")
    print("="*80)
    print(f"\nModel ready for training!")
    print(f"Architecture: {model.get_config()['model_type']}")
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Configuration saved to: models/model_config.json")
    print(f"\nNext step: Implement the training loop")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
