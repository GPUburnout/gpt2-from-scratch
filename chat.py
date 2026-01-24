"""
Interactive Chat with Trained Language Model
Chat interface to interact with your trained model in real-time
"""

import torch
import torch.nn.functional as F
import json
import os

from model import TransformerLanguageModel
from tokenizer import CharacterTokenizer


def load_model(model_dir='models/tiny_lm_v1'):
    """Load trained model and tokenizer"""
    print(f"Loading model from: {model_dir}")

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
        dropout=0.0
    )

    # Load weights
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Parameters: {config['total_parameters']:,}")

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_tokens=200, temperature=0.8, top_k=40, device='cpu'):
    """
    Generate response to a prompt

    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: User input text
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        device: Device to use

    Returns:
        Generated text
    """
    model.eval()

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        for _ in range(max_tokens):
            # Handle sequence length
            if tokens.size(1) > model.max_seq_len:
                input_tokens = tokens[:, -model.max_seq_len:]
            else:
                input_tokens = tokens

            # Forward pass
            logits = model(input_tokens)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            tokens = torch.cat([tokens, next_token], dim=1)

            # Stop at newline for chat-like responses (optional)
            if next_token.item() == tokenizer.char_to_idx.get('\n', -1):
                # Check if we have enough text (at least 20 tokens generated)
                if tokens.size(1) - len(tokenizer.encode(prompt)) > 20:
                    break

    # Decode
    generated = tokenizer.decode(tokens[0].tolist())
    return generated


def interactive_chat(model, tokenizer, device='cpu'):
    """Run interactive chat session"""
    print("\n" + "="*80)
    print("INTERACTIVE CHAT MODE")
    print("="*80)
    print("\nYou can now chat with your trained language model!")
    print("\nSettings:")
    print("  - Max tokens per response: 200")
    print("  - Temperature: 0.8 (balanced creativity)")
    print("  - Top-k: 40 (quality filtering)")
    print("\nCommands:")
    print("  - Type your prompt and press Enter")
    print("  - Type 'settings' to adjust generation parameters")
    print("  - Type 'quit' or 'exit' to end the chat")
    print("="*80 + "\n")

    # Default settings
    settings = {
        'max_tokens': 200,
        'temperature': 0.8,
        'top_k': 40
    }

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting chat...")
            break

        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        # Check for settings command
        if user_input.lower() == 'settings':
            print("\nCurrent settings:")
            print(f"  max_tokens: {settings['max_tokens']}")
            print(f"  temperature: {settings['temperature']}")
            print(f"  top_k: {settings['top_k']}")
            print("\nTo change, type: set <parameter> <value>")
            print("Example: set temperature 1.0")
            continue

        # Check for set command
        if user_input.lower().startswith('set '):
            parts = user_input.split()
            if len(parts) == 3:
                param, value = parts[1], parts[2]
                if param in settings:
                    try:
                        if param == 'top_k':
                            settings[param] = int(value)
                        else:
                            settings[param] = float(value)
                        print(f"Updated {param} to {settings[param]}")
                    except ValueError:
                        print(f"Invalid value for {param}")
                else:
                    print(f"Unknown parameter: {param}")
            continue

        # Skip empty input
        if not user_input:
            continue

        # Generate response
        print("\nModel: ", end='', flush=True)

        response = generate_response(
            model, tokenizer, user_input,
            max_tokens=settings['max_tokens'],
            temperature=settings['temperature'],
            top_k=settings['top_k'],
            device=device
        )

        # Print only the generated part (after the prompt)
        generated_text = response[len(user_input):]
        print(generated_text)
        print()


def main():
    """Main chat function"""
    print("\n" + "="*80)
    print("LANGUAGE MODEL CHAT")
    print("="*80)

    # Check if model exists
    model_dir = 'models/tiny_lm_v1'
    if not os.path.exists(os.path.join(model_dir, 'model.pth')):
        print(f"\nError: Model not found at {model_dir}")
        print("Please train the model first by running: python train.py")
        return

    # Load model
    print("\n")
    model, tokenizer = load_model(model_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # Start chat
    interactive_chat(model, tokenizer, device)

    print("\n" + "="*80)
    print("Chat session ended")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
