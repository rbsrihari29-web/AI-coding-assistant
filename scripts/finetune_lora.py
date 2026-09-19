"""
finetune_lora.py
------------------
Step 3 of the pipeline: Fine-tune an open-source LLM using LoRA (or QLoRA
if bitsandbytes + a CUDA GPU is available) on the prepared dataset.

IMPORTANT: This script needs a GPU (e.g. free Google Colab T4 GPU).
It will NOT run on a CPU-only laptop in reasonable time.

Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0  (small, open, fast to fine-tune)
You can swap MODEL_NAME for any other open HF model, e.g.:
    - microsoft/phi-2
    - Qwen/Qwen2.5-1.5B-Instruct
    - meta-llama/Llama-3.2-1B-Instruct (requires HF access approval)

Usage (on a GPU machine / Colab):
    python scripts/finetune_lora.py
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "lora-adapter")

USE_4BIT = torch.cuda.is_available()  # QLoRA only makes sense with a CUDA GPU


def load_base_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if USE_4BIT:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # CPU / no-quantization fallback (slower, plain LoRA)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )

    return model, tokenizer


def main():
    print(f"CUDA available: {torch.cuda.is_available()} | Using 4-bit QLoRA: {USE_4BIT}")

    model, tokenizer = load_base_model_and_tokenizer()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(DATA_DIR, "train.jsonl"),
            "validation": os.path.join(DATA_DIR, "val.jsonl"),
        },
    )
    # Our JSONL rows include both a "messages" field (chat format) and a
    # "text" field (flattened string). Newer trl versions auto-detect the
    # "messages" column and try to apply a chat template, which fails for
    # a base model with no chat template set. We only want the "text"
    # field used (via dataset_text_field below), so drop "messages" here.
    dataset = dataset.remove_columns("messages")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataset_text_field="text",
        max_length=512,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    trainer.train()

    # Save only the LoRA adapter (small, a few MB) — not the full base model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
