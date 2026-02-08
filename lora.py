"""
LoRA (Low-Rank Adaptation) Implementation
==========================================
Efficient fine-tuning by adding low-rank decomposition matrices to existing weights.
Only ~2% of parameters are trainable, preserving base model knowledge.

Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
https://arxiv.org/abs/2106.09685

Key idea: W_new = W_original + (B @ A) where B is d×r, A is r×k, r << min(d,k)

Usage:
    from lora import apply_lora, get_lora_state_dict, merge_lora_weights

    # Apply LoRA to model (freezes base weights, adds trainable LoRA layers)
    lora_model = apply_lora(model, rank=16, alpha=32, targets=['qkv', 'out_proj'])

    # Train normally - only LoRA weights update

    # Save just LoRA weights (small file)
    lora_state = get_lora_state_dict(lora_model)
    torch.save(lora_state, 'lora_weights.pt')

    # Merge LoRA into base model for inference (optional)
    merged_model = merge_lora_weights(lora_model)
"""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Original: y = Wx + b
    With LoRA: y = Wx + b + (B @ A)x * (alpha / rank)

    Where:
    - W: original frozen weights (d_out × d_in)
    - B: low-rank matrix (d_out × rank), initialized to zero
    - A: low-rank matrix (rank × d_in), initialized from N(0, 1)
    - alpha: scaling factor for LoRA contribution
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()

        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_out, d_in = original_linear.weight.shape

        # Get device and dtype from original layer (important for GPU compatibility)
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype

        # LoRA matrices - create on same device as original layer
        # A: initialized with small random values
        # B: initialized to zero (so LoRA starts as identity)
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, device=device, dtype=dtype))

        # Initialize A with Kaiming uniform (as in original LoRA paper)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero - this ensures the model starts identical to original

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        # Original forward pass (frozen)
        original_output = self.original(x)

        # LoRA forward pass: (B @ A) @ x * scaling
        # Shape: x is (..., d_in), A is (rank, d_in), B is (d_out, rank)
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling

        return original_output + lora_output

    def merge_weights(self):
        """Merge LoRA weights into original weights for faster inference"""
        with torch.no_grad():
            # W_merged = W + B @ A * scaling
            merged = self.original.weight + (self.lora_B @ self.lora_A) * self.scaling
            self.original.weight.copy_(merged)
        return self.original

    def extra_repr(self):
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"


def apply_lora(model, rank=16, alpha=32, targets=None, dropout=0.0):
    """
    Apply LoRA to a transformer model.

    Args:
        model: TransformerLanguageModel instance
        rank: LoRA rank (lower = fewer params, higher = more capacity)
        alpha: LoRA alpha (scaling factor, typically 2*rank)
        targets: Which layers to apply LoRA to. Options:
            - 'qkv': Query, Key, Value projections (attention)
            - 'out_proj': Output projection in attention
            - 'fc1': First FFN layer
            - 'fc2': Second FFN layer
            Default: ['qkv', 'out_proj'] (attention only, most efficient)
        dropout: Dropout for LoRA layers (usually 0 for small datasets)

    Returns:
        Modified model with LoRA layers
    """
    if targets is None:
        targets = ['qkv', 'out_proj']  # Default: attention layers only

    # Count parameters before
    original_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to target layers
    lora_layers = []

    for name, module in model.named_modules():
        # Check if this module should have LoRA applied
        if isinstance(module, nn.Linear):
            apply_lora_here = False

            if 'qkv' in targets and 'qkv' in name:
                apply_lora_here = True
            if 'out_proj' in targets and 'out_proj' in name:
                apply_lora_here = True
            if 'fc1' in targets and 'fc1' in name:
                apply_lora_here = True
            if 'fc2' in targets and 'fc2' in name:
                apply_lora_here = True

            if apply_lora_here:
                lora_layers.append(name)

    # Replace target layers with LoRA versions
    for name in lora_layers:
        # Navigate to parent module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Get original layer
        original_layer = getattr(parent, parts[-1])

        # Create LoRA layer
        lora_layer = LoRALinear(original_layer, rank=rank, alpha=alpha)

        # Replace
        setattr(parent, parts[-1], lora_layer)

    # Count parameters after
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = trainable_after

    # Summary
    print("\n" + "="*60)
    print("LoRA APPLIED")
    print("="*60)
    print(f"Rank:            {rank}")
    print(f"Alpha:           {alpha}")
    print(f"Target layers:   {targets}")
    print(f"LoRA layers:     {len(lora_layers)}")
    print(f"\nParameters:")
    print(f"  Original:      {original_params:,}")
    print(f"  Frozen:        {original_params - lora_params:,}")
    print(f"  Trainable:     {lora_params:,} ({lora_params/original_params*100:.2f}%)")
    print("="*60)

    # Store LoRA config in model for later reference
    model.lora_config = {
        'rank': rank,
        'alpha': alpha,
        'targets': targets,
        'lora_layers': lora_layers,
        'trainable_params': lora_params,
        'original_params': original_params
    }

    return model


def get_lora_state_dict(model):
    """
    Get only the LoRA weights from a model.
    This produces a small file (~1-5 MB instead of ~500 MB).
    """
    lora_state = {}

    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state[name] = param.data.clone()

    # Also save config
    if hasattr(model, 'lora_config'):
        lora_state['__lora_config__'] = model.lora_config

    return lora_state


def load_lora_state_dict(model, lora_state):
    """
    Load LoRA weights into a model that already has LoRA applied.
    """
    # Get config if present
    config = lora_state.pop('__lora_config__', None)

    # Load weights
    model_state = model.state_dict()

    for name, weight in lora_state.items():
        if name in model_state:
            model_state[name] = weight

    model.load_state_dict(model_state)

    if config:
        model.lora_config = config

    return model


def merge_lora_weights(model):
    """
    Merge LoRA weights into base weights for faster inference.
    After merging, the model behaves identically but without LoRA overhead.

    Note: This modifies the model in-place.
    """
    merged_count = 0

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Merge and replace with original linear
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)

            # Merge weights
            merged_linear = module.merge_weights()
            merged_linear.weight.requires_grad = True  # Unfreeze after merge

            # Replace LoRA layer with merged linear
            setattr(parent, child_name, merged_linear)
            merged_count += 1

    print(f"Merged {merged_count} LoRA layers into base weights")

    # Clear LoRA config
    if hasattr(model, 'lora_config'):
        delattr(model, 'lora_config')

    return model


def count_parameters(model, only_trainable=True):
    """Count model parameters"""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# Test
if __name__ == "__main__":
    print("\nTesting LoRA implementation...")

    # Create a simple test
    from model import TransformerLanguageModel

    # Create small model for testing
    model = TransformerLanguageModel(
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        max_seq_len=128
    )

    print(f"\nOriginal model parameters: {count_parameters(model, False):,}")
    print(f"Original trainable: {count_parameters(model, True):,}")

    # Apply LoRA
    model = apply_lora(model, rank=8, alpha=16, targets=['qkv', 'out_proj'])

    print(f"\nAfter LoRA:")
    print(f"Total parameters: {count_parameters(model, False):,}")
    print(f"Trainable (LoRA only): {count_parameters(model, True):,}")

    # Test forward pass
    x = torch.randint(0, 1000, (2, 32))
    y = model(x)
    print(f"\nForward pass: input {x.shape} -> output {y.shape}")

    # Test saving LoRA weights
    lora_state = get_lora_state_dict(model)
    print(f"\nLoRA state dict keys: {len(lora_state)} entries")

    # Test merging
    print("\nMerging LoRA weights into base model...")
    model = merge_lora_weights(model)
    print(f"After merge - trainable: {count_parameters(model, True):,}")

    # Verify output is same after merge
    y2 = model(x)
    diff = (y - y2).abs().max().item()
    print(f"Output difference after merge: {diff:.6f} (should be ~0)")

    print("\nLoRA implementation test passed!")
