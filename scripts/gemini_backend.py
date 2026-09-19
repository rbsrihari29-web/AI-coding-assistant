"""
gemini_backend.py
--------------------
Wraps Google's free Gemini API so the app has a working, always-available
"AI coding assistant" backend, even without a local GPU to run the LoRA
fine-tuned model.

To make this act like our "fine-tuned" assistant (rather than a generic
chatbot), we ground every call with:
  1. A system prompt describing the assistant's role/persona.
  2. A handful of few-shot examples pulled straight from our training
     dataset (data/train.jsonl) — this steers Gemini's answer style to
     match the tone/format we fine-tuned toward, which is a lightweight
     stand-in for full weight fine-tuning when only a free API is used.

Setup:
  1. Get a free API key: https://aistudio.google.com/app/apikey
  2. Copy .env.example -> .env and paste your key in as GOOGLE_API_KEY=...
"""

import os
import json
import random
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")

SYSTEM_PROMPT = (
    "You are a helpful AI assistant for engineering students. "
    "You answer programming and computer-science learning questions "
    "clearly, correctly, and concisely, with short examples when useful. "
    "Keep answers focused and avoid unnecessary filler."
)

# Fallback list used only if we can't reach the API to list models live.
# Kept in "best/newest first" order.
DEFAULT_MODEL_CANDIDATES = [
    "gemini-3.5-flash",
    "gemini-3.5-pro",
    "gemini-3.0-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
]

_models_cache = {}  # model_name -> genai.GenerativeModel instance
_configured = False


def _ensure_configured():
    global _configured
    if _configured:
        return
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "paste_your_api_key_here":
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and paste "
            "your free Gemini API key (https://aistudio.google.com/app/apikey)."
        )
    import google.generativeai as genai

    genai.configure(api_key=GOOGLE_API_KEY)
    _configured = True


def list_available_models() -> list[str]:
    """
    Ask the Gemini API which models this key can actually use for
    generateContent right now, so the UI can offer a correct, up-to-date
    dropdown instead of a hardcoded (and easily outdated) model name.
    Falls back to DEFAULT_MODEL_CANDIDATES if listing fails for any reason.
    """
    try:
        _ensure_configured()
        import google.generativeai as genai

        names = []
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                name = m.name.split("/")[-1]  # e.g. "models/gemini-2.5-flash" -> "gemini-2.5-flash"
                if "gemini" in name and "vision" not in name:
                    names.append(name)
        if names:
            # Prefer flash/pro "latest"-style and non-preview names first
            names.sort(key=lambda n: (("preview" in n), ("exp" in n), n))
            return names
    except Exception:
        pass
    return DEFAULT_MODEL_CANDIDATES


def _get_model(model_name: str):
    _ensure_configured()
    if model_name in _models_cache:
        return _models_cache[model_name]

    import google.generativeai as genai

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
    )
    _models_cache[model_name] = model
    return model


def _load_few_shot_examples(n=3):
    """Pull a few examples from our training dataset to steer style/tone."""
    if not os.path.exists(TRAIN_PATH):
        return []
    examples = []
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            q = next(m["content"] for m in msgs if m["role"] == "user")
            a = next(m["content"] for m in msgs if m["role"] == "assistant")
            examples.append((q, a))
    random.shuffle(examples)
    return examples[:n]


def ask_gemini(question: str, history: list | None = None, use_few_shot: bool = True,
                model_name: str = "gemini-3.5-flash") -> str:
    """
    Send a question (plus optional chat history) to Gemini and return the answer.

    history: list of (user_msg, assistant_msg) tuples from prior turns in
             this conversation (for the "conversation history" bonus feature).
    model_name: which Gemini model to use (see list_available_models()).
    """
    model = _get_model(model_name)

    convo_parts = []

    if use_few_shot:
        for q, a in _load_few_shot_examples():
            convo_parts.append(f"Example question: {q}\nExample answer: {a}")

    if history:
        for user_msg, assistant_msg in history:
            convo_parts.append(f"Student: {user_msg}\nAssistant: {assistant_msg}")

    convo_parts.append(f"Student: {question}\nAssistant:")
    prompt = "\n\n".join(convo_parts)

    response = model.generate_content(prompt)
    return response.text.strip()


if __name__ == "__main__":
    # Quick manual test
    print(ask_gemini("What is a for loop?"))
