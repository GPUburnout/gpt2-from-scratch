"""
Text Generation with Trained Language Model
Generates new text using the trained Transformer model
"""

import torch
import torch.nn.functional as F
import json
import os

from model import TransformerLanguageModel
from tokenizer import CharacterTokenizer


def load_model(model_dir='models/tiny_lm_v1'):
    """Load trained model and tokenizer"""
    print(f"\nLoading model from: {model_dir}")

    # Load tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer_path = os.path.join(model_dir.replace('tiny_lm_v1', ''), 'tokenizer.json')
    tokenizer.load(tokenizer_path)

    # Load config
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model
    model = TransformerLanguageModel(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        max_seq_len=config['max_seq_len'],
        dropout=0.0  # No dropout during generation
    )

    # Load weights
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    print(f"Model loaded: {config['total_parameters']:,} parameters")

    return model, tokenizer


def generate_text(model, tokenizer, prompt="", max_new_tokens=500, temperature=1.0, top_k=None, device='cpu'):
    """
    Generate text using the trained model

    Args:
        model: Trained Transformer model
        tokenizer: Character tokenizer
        prompt: Starting text (empty for random generation)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k most likely tokens
        device: Device to run on

    Returns:
        Generated text string
    """
    model = model.to(device)
    model.eval()

    # Encode prompt
    if prompt:
        tokens = tokenizer.encode(prompt)
    else:
        # Start with newline if no prompt
        tokens = tokenizer.encode('\n')

    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get predictions (use only last max_seq_len tokens if sequence is too long)
            if tokens.size(1) > model.max_seq_len:
                input_tokens = tokens[:, -model.max_seq_len:]
            else:
                input_tokens = tokens

            # Forward pass
            logits = model(input_tokens)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

    # Decode
    generated_tokens = tokens[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text


def main():
    """Main generation function"""
    print("\n" + "="*80)
    print("TEXT GENERATION - Tiny Language Model")
    print("="*80)

    # Load model
    model, tokenizer = load_model('models/tiny_lm_v1')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Generation settings
    print("="*80)
    print("GENERATION SETTINGS")
    print("="*80)
    print("max_new_tokens: 500 (length of generated text)")
    print("temperature: 0.8 (lower = more conservative, higher = more creative)")
    print("top_k: 40 (only sample from top 40 most likely tokens)")
    print()

    # Example 1: Generate from scratch
    print("="*80)
    print("EXAMPLE 1: Generate from scratch (no prompt)")
    print("="*80)
    print("\nGenerating...")
    text = generate_text(
        model, tokenizer,
        prompt="",
        max_new_tokens=500,
        temperature=0.8,
        top_k=40,
        device=device
    )
    print(text)

    # Example 2: Continue from prompt
    print("\n" + "="*80)
    print("EXAMPLE 2: Continue from prompt")
    print("="*80)
    prompt = "First Citizen:"
    print(f"\nPrompt: \"{prompt}\"")
    print("\nGenerating...")
    text = generate_text(
        model, tokenizer,
        prompt=prompt,
        max_new_tokens=500,
        temperature=0.8,
        top_k=40,
        device=device
    )
    print(text)

    # Example 3: More creative (higher temperature)
    print("\n" + "="*80)
    print("EXAMPLE 3: More creative generation (temperature=1.2)")
    print("="*80)
    prompt = "To be, or not to be"
    print(f"\nPrompt: \"{prompt}\"")
    print("\nGenerating...")
    text = generate_text(
        model, tokenizer,
        prompt=prompt,
        max_new_tokens=300,
        temperature=1.2,
        top_k=40,
        device=device
    )
    print(text)

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print("\nTry your own prompts!")
    print("Modify the prompts in generate.py or create a new script.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
