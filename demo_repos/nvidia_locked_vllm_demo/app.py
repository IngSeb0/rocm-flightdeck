"""
Demo vLLM inference app — intentionally uses NVIDIA/CUDA assumptions.
This file is part of the ROCm FlightDeck demo repo (nvidia_locked_vllm_demo).
"""

import torch

# NVIDIA assumption: hardcoded CUDA device
DEVICE = torch.device("cuda")


def load_model(model_name: str = "meta-llama/Llama-2-7b-hf"):
    """Load a model onto the CUDA device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to("cuda")   # hardcoded CUDA move
    return model, tokenizer


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """Run a simple inference pass."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This demo requires an NVIDIA GPU.")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()   # .cuda() call

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {DEVICE}")

    model, tokenizer = load_model()
    result = run_inference(model, tokenizer, "Tell me about AMD ROCm.")
    print(result)


if __name__ == "__main__":
    main()
