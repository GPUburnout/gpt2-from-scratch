"""
Gradio Demo for GPT-2 From Scratch
Multi-model demo: Tiny → Medium → GPT-2 Small
Deploy to HuggingFace Spaces for interactive blog demo
"""

import gradio as gr
import torch
import torch.nn.functional as F
from model import TransformerLanguageModel
import json
import os

# Available models
MODELS = {
    "Tiny Shakespeare (3.2M params)": {
        "path": "checkpoint_tiny",
        "description": "Phase 1: Character-level model trained on Shakespeare"
    },
    "Medium Character (3.3M params)": {
        "path": "checkpoint_medium",
        "description": "Phase 2: Character-level model trained on 250MB foundational dataset"
    },
    "GPT-2 Small (134M params)": {
        "path": "checkpoint_gpt2_small",
        "description": "Phase 3: BPE tokenizer, 12GB data, epoch 11 (final)"
    }
}

# Cache for loaded models
loaded_models = {}


def load_model(model_name):
    """Load a model by name, with caching"""
    if model_name in loaded_models:
        return loaded_models[model_name]

    model_info = MODELS.get(model_name)
    if not model_info:
        return None, None, None

    model_dir = model_info["path"]
    config_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(config_path):
        return None, None, f"Model not found: {model_dir}"

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load tokenizer based on type
    tokenizer_type = config.get("tokenizer_type", "character")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")

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
    model_path = os.path.join(model_dir, "pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Cache it
    loaded_models[model_name] = (model, tokenizer, config)
    return model, tokenizer, config


def generate(model, tokenizer, config, prompt, max_tokens=100, temperature=0.8, top_k=40):
    """Generate text from prompt"""
    if model is None:
        return "Model not loaded."

    if not prompt.strip():
        return "Please enter a prompt."

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    if len(tokens) == 0:
        return "Could not encode prompt. Try different characters."

    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    max_seq_len = config.get("max_seq_len", 256)

    with torch.no_grad():
        for _ in range(max_tokens):
            # Truncate if too long
            if tokens.size(1) > max_seq_len:
                input_tokens = tokens[:, -max_seq_len:]
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


def generate_wrapper(model_name, prompt, max_tokens, temperature, top_k):
    """Wrapper that loads the selected model and generates"""
    model, tokenizer, config = load_model(model_name)

    if isinstance(config, str):  # Error message
        return config

    if model is None:
        return f"Model '{model_name}' not available. Check if checkpoint exists."

    return generate(model, tokenizer, config, prompt, int(max_tokens), temperature, int(top_k))


def get_model_info(model_name):
    """Get info string for selected model"""
    model, tokenizer, config = load_model(model_name)

    if model is None:
        return f"⚠️ {model_name} - Not loaded (checkpoint missing)"

    params = config.get("total_parameters", 0)
    tok_type = config.get("tokenizer_type", "character")
    return f"✅ {model_name} | {params:,} parameters | {tok_type} tokenizer"


def update_examples(model_name):
    """Update example prompts based on model"""
    if "Shakespeare" in model_name or "Tiny" in model_name:
        return gr.update(samples=[
            ["ROMEO:"],
            ["JULIET:"],
            ["To be, or not to be"],
            ["First Citizen:"],
        ])
    else:
        return gr.update(samples=[
            ["What is the capital of France?"],
            ["Explain machine learning in simple terms."],
            ["Write a poem about coffee."],
            ["The meaning of life is"],
        ])


# Check which models are available
available_models = []
for name, info in MODELS.items():
    config_path = os.path.join(info["path"], "config.json")
    if os.path.exists(config_path):
        available_models.append(name)

if not available_models:
    available_models = list(MODELS.keys())  # Show all, will error on use


# Gradio interface
with gr.Blocks(title="GPT From Scratch Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # GPT From Scratch Demo

        Compare models from my training journey — from tiny Shakespeare to GPT-2 Small.

        [Read the blog](https://gpuburnout.github.io/llm-journey/) |
        [View the code](https://github.com/GPUburnout/gpt2-from-scratch)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=available_models[0] if available_models else list(MODELS.keys())[0],
                label="Select Model",
                info="Choose which model to use for generation"
            )

            model_status = gr.Markdown(value="")

            prompt = gr.Textbox(
                label="Enter your prompt",
                placeholder="Type something...",
                lines=2,
                value="ROMEO:" if "Tiny" in available_models[0] else "What is the capital of France?"
            )

            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=50, maximum=500, value=200, step=50,
                    label="Max tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.8, step=0.1,
                    label="Temperature"
                )

            top_k = gr.Slider(
                minimum=1, maximum=100, value=40, step=1,
                label="Top-K sampling"
            )

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            output = gr.Textbox(label="Generated text", lines=15, show_copy_button=True)

    # Example prompts
    examples = gr.Examples(
        examples=[["ROMEO:"], ["JULIET:"], ["To be, or not to be"]],
        inputs=prompt,
        label="Example prompts"
    )

    # Update model status on load and selection change
    demo.load(
        fn=get_model_info,
        inputs=model_selector,
        outputs=model_status
    )

    model_selector.change(
        fn=get_model_info,
        inputs=model_selector,
        outputs=model_status
    )

    # Generate on button click or enter
    generate_btn.click(
        generate_wrapper,
        inputs=[model_selector, prompt, max_tokens, temperature, top_k],
        outputs=output
    )

    prompt.submit(
        generate_wrapper,
        inputs=[model_selector, prompt, max_tokens, temperature, top_k],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
