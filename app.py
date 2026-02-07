"""
Gradio Demo for GPT-2 From Scratch
Deploy to HuggingFace Spaces for interactive blog demo
"""

import gradio as gr
import torch
import torch.nn.functional as F
from model import TransformerLanguageModel
import json
import os

# Model directory
MODEL_DIR = "checkpoint"

def load_model():
    """Load the trained model and tokenizer"""
    # Load config
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load tokenizer based on type
    tokenizer_type = config.get("tokenizer_type", "character")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")

    if tokenizer_type == "bpe":
        from tokenizer_bpe import BPETokenizer
        tokenizer = BPETokenizer()
        tokenizer.load(tokenizer_path)
    else:
        from tokenizer import CharacterTokenizer
        tokenizer = CharacterTokenizer()
        tokenizer.load(tokenizer_path)

    # Create model
    model = TransformerLanguageModel(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0
    )

    # Load weights
    model_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, tokenizer, config


def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=40):
    """Generate text from prompt"""
    if not prompt.strip():
        return "Please enter a prompt."

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    if len(tokens) == 0:
        return "Could not encode prompt. Try different characters."

    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_tokens):
            # Truncate if too long
            if tokens.size(1) > model.max_seq_len:
                input_tokens = tokens[:, -model.max_seq_len:]
            else:
                input_tokens = tokens

            # Forward pass
            logits = model(input_tokens)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())


# Try to load model
try:
    model, tokenizer, config = load_model()
    model_loaded = True
    model_name = config.get("model_name", "unknown")
    param_count = config.get("total_parameters", 0)
    tokenizer_type = config.get("tokenizer_type", "character")
    model_info = f"{model_name} ({param_count:,} params, {tokenizer_type} tokenizer)"
except Exception as e:
    model_loaded = False
    model_info = f"Model not loaded: {e}"
    model = None
    tokenizer = None


def generate_wrapper(prompt, max_tokens, temperature, top_k):
    if not model_loaded:
        return "Model not loaded. Please check the checkpoint folder."
    return generate(model, tokenizer, prompt, int(max_tokens), temperature, int(top_k))


# Gradio interface
with gr.Blocks(title="GPT From Scratch Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # GPT From Scratch Demo

        This is the **Phase 1 model** — a tiny transformer trained on Shakespeare text.
        3.2M parameters, character-level tokenizer.

        *The full 134M parameter GPT-2 Small is coming soon!*

        [Read the blog](https://gpuburnout.github.io/llm-journey/) |
        [View the code](https://github.com/GPUburnout/gpt2-from-scratch)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Enter your prompt",
                placeholder="ROMEO:",
                lines=2,
                value="ROMEO:"
            )

            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=50, maximum=500, value=200, step=50,
                    label="Max characters"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.8, step=0.1,
                    label="Temperature (higher = more creative)"
                )

            top_k = gr.Slider(
                minimum=1, maximum=65, value=40, step=1,
                label="Top-K sampling"
            )

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            output = gr.Textbox(label="Generated text", lines=15, show_copy_button=True)

    gr.Markdown(f"**Model:** {model_info}")

    # Example prompts for Shakespeare
    gr.Examples(
        examples=[
            ["ROMEO:"],
            ["JULIET:"],
            ["To be, or not to be"],
            ["First Citizen:"],
            ["KING HENRY:"],
        ],
        inputs=prompt,
        label="Try these Shakespeare prompts"
    )

    generate_btn.click(
        generate_wrapper,
        inputs=[prompt, max_tokens, temperature, top_k],
        outputs=output
    )

    prompt.submit(
        generate_wrapper,
        inputs=[prompt, max_tokens, temperature, top_k],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
