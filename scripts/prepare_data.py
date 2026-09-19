"""
prepare_data.py
----------------
Step 1 & 2 of the pipeline: Load the raw Q&A dataset, clean it,
and format it into an instruction-tuning JSONL format suitable
for fine-tuning with Hugging Face + PEFT (LoRA/QLoRA).

Usage:
    python scripts/prepare_data.py
"""

import json
import os
import re

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_dataset.json")
OUT_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.jsonl")
OUT_VAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "val.jsonl")

SYSTEM_PROMPT = (
    "You are a helpful AI assistant for engineering students. "
    "You answer programming and computer-science learning questions "
    "clearly, correctly, and concisely, with examples when useful."
)


def clean_text(text: str) -> str:
    """Basic cleaning: strip whitespace, collapse multiple spaces/newlines."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def to_instruction_format(item: dict) -> dict:
    """
    Convert a {question, answer} pair into the chat / instruction format
    used by most instruction-tuned open-source models (Alpaca/ChatML style).
    """
    question = clean_text(item["question"])
    answer = clean_text(item["answer"])

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        # Also keep a flattened "text" field — some trainers (e.g. TRL's
        # SFTTrainer with a plain text dataset) expect a single string.
        "text": (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{question}\n"
            f"<|assistant|>\n{answer}"
        ),
    }


def main():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Basic validation + cleaning
    cleaned = []
    seen_questions = set()
    for item in raw_data:
        if "question" not in item or "answer" not in item:
            continue
        q = clean_text(item["question"])
        a = clean_text(item["answer"])
        if not q or not a:
            continue
        if q.lower() in seen_questions:  # de-duplicate
            continue
        seen_questions.add(q.lower())
        cleaned.append({"question": q, "answer": a})

    print(f"Loaded {len(raw_data)} raw examples -> {len(cleaned)} after cleaning/dedup.")

    formatted = [to_instruction_format(item) for item in cleaned]

    # Simple 90/10 train/val split
    split_idx = max(1, int(len(formatted) * 0.9))
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:] if len(formatted) - split_idx > 0 else formatted[-2:]

    os.makedirs(os.path.dirname(OUT_TRAIN_PATH), exist_ok=True)

    with open(OUT_TRAIN_PATH, "w", encoding="utf-8") as f:
        for row in train_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(OUT_VAL_PATH, "w", encoding="utf-8") as f:
        for row in val_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_data)} training examples to {OUT_TRAIN_PATH}")
    print(f"Wrote {len(val_data)} validation examples to {OUT_VAL_PATH}")


if __name__ == "__main__":
    main()
