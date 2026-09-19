"""
local_model_backend.py
-------------------------
Loads the locally LoRA-fine-tuned model (produced by finetune_lora.py) and
runs inference. This is optional — only used if models/lora-adapter exists
(i.e. you actually ran the fine-tuning step on a GPU machine).

If the adapter isn't present, is_available() returns False and the app
will fall back to the Gemini API backend automatically.
"""

import os

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "lora-adapter")

SYSTEM_PROMPT = (
    "You are a helpful AI assistant for engineering students. "
    "You answer programming and computer-science learning questions "
    "clearly, correctly, and concisely, with short examples when useful."
)

_model = None
_tokenizer = None


def is_available() -> bool:
    return os.path.exists(ADAPTER_DIR) and any(os.scandir(ADAPTER_DIR))


def _load():
    global _model, _tokenizer
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    _model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    _model.eval()


def ask_local_model(question: str, history: list | None = None, max_new_tokens: int = 200) -> str:
    import torch

    _load()

    convo = f"<|system|>\n{SYSTEM_PROMPT}\n"
    if history:
        for user_msg, assistant_msg in history:
            convo += f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}\n"
    convo += f"<|user|>\n{question}\n<|assistant|>\n"

    inputs = _tokenizer(convo, return_tensors="pt")
    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=_tokenizer.eos_token_id,
        )
    text = _tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("<|assistant|>")[-1].strip()
