"""Partially portable PyTorch inference demo for ROCm FlightDeck."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str = "meta-llama/Llama-2-7b-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)
    return model, tokenizer


def run_inference(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=64)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
