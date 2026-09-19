"""
compare_models.py
--------------------
Step 4 & 5 of the pipeline: Test the fine-tuned model on a fixed set of
programming questions and compare its answers to the base (non-fine-tuned)
model, side by side. Saves results to results/comparison.json.

Usage (on the same GPU machine where you ran finetune_lora.py):
    python scripts/compare_models.py
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "lora-adapter")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

TEST_QUESTIONS = [
    "What is a variable in programming?",
    "Explain the difference between a list and a tuple in Python.",
    "What is Big O notation?",
    "What is recursion? Give a simple example.",
    "What is the difference between a stack and a queue?",
    "How do I debug my code effectively?",
    "What is a REST API?",
    "What is the difference between == and === in JavaScript?",
]

SYSTEM_PROMPT = (
    "You are a helpful AI assistant for engineering students. "
    "You answer programming and computer-science learning questions "
    "clearly, correctly, and concisely, with examples when useful."
)


def generate(model, tokenizer, question, max_new_tokens=150):
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{question}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("<|assistant|>")[-1].strip()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    print("Loading fine-tuned (LoRA) model...")
    ft_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_DIR)

    results = []
    for q in TEST_QUESTIONS:
        print(f"\n--- Question: {q} ---")
        base_answer = generate(base_model, tokenizer, q)
        ft_answer = generate(ft_model, tokenizer, q)
        print(f"[BASE]      {base_answer}")
        print(f"[FINE-TUNED] {ft_answer}")
        results.append({"question": q, "base_model": base_answer, "fine_tuned_model": ft_answer})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved comparison results to {out_path}")


if __name__ == "__main__":
    main()
