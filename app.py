"""
Gradio Demo for GPT-2 From Scratch
Deploy to HuggingFace Spaces for interactive blog demo
"""

import gradio as gr
import torch
import torch.nn.functional as F
from model import TransformerLanguageModel
from tokenizer_bpe import BPETokenizer
import json
import os

# Load model and tokenizer
def load_model():
    """Load the trained model and tokenizer"""
    model_dir = "checkpoint"  # Will contain uploaded model files

    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(model_dir, "tokenizer.json"))

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
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
    )
    model.eval()

    return model, tokenizer, config

# Generate text
def generate(prompt, max_tokens=100, temperature=0.8, top_k=40):
    """Generate text from prompt"""
    if not prompt.strip():
        return "Please enter a prompt."

    tokens = tokenizer.encode(prompt)
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

# Try to load model (will fail locally without checkpoint, but works on HF Spaces)
try:
    model, tokenizer, config = load_model()
    model_loaded = True
    model_info = f"GPT-2 Small ({config.get('total_parameters', 134000000):,} params)"
except Exception as e:
    model_loaded = False
    model_info = f"Model not loaded: {e}"

def generate_wrapper(prompt, max_tokens, temperature, top_k):
    if not model_loaded:
        return "Model not loaded. Deploy to HuggingFace Spaces with checkpoint files."
    return generate(prompt, int(max_tokens), temperature, int(top_k))

# Gradio interface
with gr.Blocks(title="GPT-2 From Scratch") as demo:
    gr.Markdown(
        """
        # GPT-2 From Scratch Demo

        This model was trained from scratch on 12GB of conversational data.
        134M parameters, 10 epochs, ~50 hours on A100.

        [Read the blog](https://gpuburnout.github.io/llm-journey/) |
        [View the code](https://github.com/GPUburnout/gpt2-from-scratch)
        """
    )

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Enter your prompt",
                placeholder="What is the capital of France?",
                lines=3
            )

            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=10, maximum=200, value=100, step=10,
                    label="Max tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.8, step=0.1,
                    label="Temperature"
                )
                top_k = gr.Slider(
                    minimum=1, maximum=100, value=40, step=1,
                    label="Top-K"
                )

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="Generated text", lines=10)

    gr.Markdown(f"**Model:** {model_info}")

    # Example prompts
    gr.Examples(
        examples=[
            ["What is the capital of France?"],
            ["Explain machine learning in simple terms."],
            ["Write a poem about coffee."],
            ["The meaning of life is"],
        ],
        inputs=prompt
    )

    generate_btn.click(
        generate_wrapper,
        inputs=[prompt, max_tokens, temperature, top_k],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
