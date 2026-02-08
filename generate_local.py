"""
GPT-2 Local Text Generator
==========================
Interactive text generation with your trained GPT-2 model.

Usage:
    python generate_local.py                    # Launches web UI (requires gradio)
    python generate_local.py --terminal         # Terminal mode (no dependencies)
    python generate_local.py --prompt "Hello"   # Quick generation

Requirements:
    pip install torch gradio   (gradio optional, for web UI)
"""

import argparse
import json
import os
import sys
import torch

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from model import TransformerLanguageModel
from tokenizer_bpe import BPETokenizer


# ============================================================
# Configuration
# ============================================================
MODEL_PATH = os.path.join(script_dir, "models", "gpt2_trained")

DEFAULT_SETTINGS = {
    "temperature": 0.8,
    "max_tokens": 100,
    "top_k": 50,
    "rep_penalty": 1.2,
}


# ============================================================
# Load Model
# ============================================================
def load_model(model_path=MODEL_PATH, device=None):
    """Load the trained model and tokenizer."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")

    # Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Model: {config.get('architecture', 'unknown')}")
    print(f"Parameters: {config.get('total_parameters', 'unknown'):,}")

    # Create model
    model = TransformerLanguageModel(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        max_seq_len=config['max_seq_len'],
        dropout=config.get('dropout', 0.1)
    )

    # Load weights (handle torch.compile prefix)
    weights_path = os.path.join(model_path, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location=device)

    # Remove _orig_mod. prefix if present (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = os.path.join(model_path, "tokenizer.json")
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    print(f"Vocab size: {tokenizer.vocab_size:,}")
    print("Model loaded successfully!\n")

    return model, tokenizer, device, config


# ============================================================
# Generation Function
# ============================================================
def generate_text(
    prompt,
    model,
    tokenizer,
    device,
    max_tokens=100,
    temperature=0.8,
    top_k=50,
    rep_penalty=1.2,
    stop_sequences=None
):
    """Generate text from a prompt."""

    if stop_sequences is None:
        stop_sequences = ["\n\n", "\nQ:", "\n---", "Human:", "User:"]

    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens]).to(device)
    max_seq_len = model.max_seq_len

    generated_text = prompt

    with torch.no_grad():
        for _ in range(max_tokens):
            # Truncate to max sequence length
            logits = model(input_ids[:, -max_seq_len:])
            next_logits = logits[0, -1, :] / temperature

            # Repetition penalty
            if rep_penalty > 1.0:
                for token_id in set(input_ids[0].tolist()[-50:]):
                    next_logits[token_id] /= rep_penalty

            # Top-K sampling
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                next_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Decode current output
            generated_text = tokenizer.decode(input_ids[0].tolist())

            # Check for stop sequences
            for stop_seq in stop_sequences:
                if stop_seq in generated_text[len(prompt):]:
                    stop_idx = generated_text.find(stop_seq, len(prompt))
                    return generated_text[:stop_idx].strip()

    return generated_text.strip()


# ============================================================
# Terminal Interface
# ============================================================
def run_terminal_mode(model, tokenizer, device):
    """Interactive terminal mode."""

    print("=" * 60)
    print("GPT-2 Text Generator - Terminal Mode")
    print("=" * 60)
    print("Commands:")
    print("  /quit or /exit  - Exit the program")
    print("  /settings       - Show current settings")
    print("  /temp <value>   - Set temperature (0.1-2.0)")
    print("  /tokens <value> - Set max tokens (10-500)")
    print("  /topk <value>   - Set top-k (0-100)")
    print("  /rep <value>    - Set repetition penalty (1.0-2.0)")
    print("=" * 60)
    print()

    settings = DEFAULT_SETTINGS.copy()

    while True:
        try:
            prompt = input("📝 Enter prompt (or command): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        # Handle commands
        if prompt.lower() in ['/quit', '/exit', 'quit', 'exit']:
            print("Goodbye!")
            break

        elif prompt.lower() == '/settings':
            print(f"\nCurrent settings:")
            print(f"  Temperature: {settings['temperature']}")
            print(f"  Max tokens:  {settings['max_tokens']}")
            print(f"  Top-K:       {settings['top_k']}")
            print(f"  Rep penalty: {settings['rep_penalty']}")
            print()
            continue

        elif prompt.lower().startswith('/temp '):
            try:
                val = float(prompt.split()[1])
                settings['temperature'] = max(0.1, min(2.0, val))
                print(f"Temperature set to {settings['temperature']}")
            except:
                print("Usage: /temp <value>  (0.1-2.0)")
            continue

        elif prompt.lower().startswith('/tokens '):
            try:
                val = int(prompt.split()[1])
                settings['max_tokens'] = max(10, min(500, val))
                print(f"Max tokens set to {settings['max_tokens']}")
            except:
                print("Usage: /tokens <value>  (10-500)")
            continue

        elif prompt.lower().startswith('/topk '):
            try:
                val = int(prompt.split()[1])
                settings['top_k'] = max(0, min(100, val))
                print(f"Top-K set to {settings['top_k']}")
            except:
                print("Usage: /topk <value>  (0-100)")
            continue

        elif prompt.lower().startswith('/rep '):
            try:
                val = float(prompt.split()[1])
                settings['rep_penalty'] = max(1.0, min(2.0, val))
                print(f"Repetition penalty set to {settings['rep_penalty']}")
            except:
                print("Usage: /rep <value>  (1.0-2.0)")
            continue

        # Generate text
        print("\n⏳ Generating...\n")
        print("-" * 60)

        result = generate_text(
            prompt,
            model,
            tokenizer,
            device,
            max_tokens=settings['max_tokens'],
            temperature=settings['temperature'],
            top_k=settings['top_k'],
            rep_penalty=settings['rep_penalty']
        )

        print(result)
        print("-" * 60)
        print()


# ============================================================
# Gradio Web Interface
# ============================================================
def run_gradio_mode(model, tokenizer, device, config):
    """Launch Gradio web interface."""

    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install gradio")
        print("Falling back to terminal mode...\n")
        run_terminal_mode(model, tokenizer, device)
        return

    def generate_wrapper(prompt, temperature, max_tokens, top_k, rep_penalty):
        result = generate_text(
            prompt,
            model,
            tokenizer,
            device,
            max_tokens=int(max_tokens),
            temperature=temperature,
            top_k=int(top_k),
            rep_penalty=rep_penalty
        )
        return result

    # Create interface
    with gr.Blocks(title="GPT-2 Text Generator") as demo:
        gr.Markdown(f"""
        # 🤖 GPT-2 Text Generator

        **Model:** {config.get('architecture', 'GPT-2')} | **Parameters:** {config.get('total_parameters', 0):,} | **Device:** {device}

        Generate text using your locally trained GPT-2 model!
        """)

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Enter your prompt",
                    placeholder="The meaning of life is...",
                    lines=3,
                    value="The meaning of life is"
                )
                generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                    label="Temperature",
                    info="Lower = focused, Higher = creative"
                )
                max_tokens = gr.Slider(
                    minimum=10, maximum=500, value=100, step=10,
                    label="Max Tokens",
                    info="Length of generated text"
                )
                top_k = gr.Slider(
                    minimum=0, maximum=100, value=50, step=5,
                    label="Top-K (0 = off)",
                    info="Limits vocabulary diversity"
                )
                rep_penalty = gr.Slider(
                    minimum=1.0, maximum=2.0, value=1.2, step=0.1,
                    label="Repetition Penalty",
                    info="Higher = less repetition"
                )

        output = gr.Textbox(
            label="Generated Output",
            lines=10
        )

        gr.Markdown("""
        ### 💡 Tips
        - **Temperature 0.3-0.5:** Focused, factual, may repeat
        - **Temperature 0.7-0.9:** Balanced, creative
        - **Temperature 1.2+:** Wild, chaotic
        - **Use Q:/A: format** for better question answering

        ### 📝 Example Prompts
        - `Q: What is machine learning?\\nA:`
        - `The following is a poem about nature:\\n\\n`
        - `def fibonacci(n):\\n    `
        """)

        # Event handlers
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[prompt_input, temperature, max_tokens, top_k, rep_penalty],
            outputs=output
        )

        prompt_input.submit(
            fn=generate_wrapper,
            inputs=[prompt_input, temperature, max_tokens, top_k, rep_penalty],
            outputs=output
        )

    print("Launching web interface...")
    print("Open http://127.0.0.1:7860 in your browser\n")
    demo.launch(inbrowser=True)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='GPT-2 Local Text Generator')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='Path to model directory')
    parser.add_argument('--terminal', action='store_true',
                        help='Use terminal mode instead of web UI')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Generate from this prompt and exit')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-K sampling')
    parser.add_argument('--rep_penalty', type=float, default=1.2,
                        help='Repetition penalty')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')

    args = parser.parse_args()

    # Set device
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, tokenizer, device, config = load_model(args.model_path, device)

    # Quick generation mode
    if args.prompt:
        result = generate_text(
            args.prompt,
            model,
            tokenizer,
            device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            rep_penalty=args.rep_penalty
        )
        print(result)
        return

    # Interactive modes
    if args.terminal:
        run_terminal_mode(model, tokenizer, device)
    else:
        run_gradio_mode(model, tokenizer, device, config)


if __name__ == "__main__":
    main()
