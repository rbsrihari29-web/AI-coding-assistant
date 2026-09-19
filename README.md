# AI Coding Assistant for Engineering Students

A small AI-powered assistant that answers programming and CS learning
questions. It's built two ways so you can get it running **immediately**
with a free API, and optionally go further and actually fine-tune your
own open-source model with LoRA/QLoRA.

- ✅ **Fast path (recommended)**: Chat interface powered by Google's
  **free Gemini API**. Works in minutes, no GPU needed.
- ✅ **Full path (optional)**: Fine-tune `TinyLlama-1.1B-Chat` with LoRA/QLoRA
  on our custom dataset, then compare it against the base model, and use
  it inside the same chat app.

---

## 1. Project Structure

```
ai-coding-assistant/
├── app.py                        # Gradio chat interface (main entry point)
├── .env.example                   # Put your free Google API key here
├── requirements.txt               # Deps for the app (Gemini path)
├── requirements-finetune.txt      # Extra deps only for LoRA fine-tuning (GPU)
├── data/
│   ├── raw_dataset.json           # Raw Q&A pairs (42 programming/CS questions)
│   ├── train.jsonl                # Cleaned + formatted training split (auto-generated)
│   └── val.jsonl                  # Cleaned + formatted validation split (auto-generated)
├── scripts/
│   ├── prepare_data.py            # Step 1+2: clean & format dataset
│   ├── finetune_lora.py           # Step 3: LoRA/QLoRA fine-tuning (needs GPU)
│   ├── compare_models.py          # Step 4+5: test & compare base vs fine-tuned
│   ├── gemini_backend.py          # Free Gemini API backend used by app.py
│   └── local_model_backend.py     # Local fine-tuned model backend used by app.py
├── models/
│   └── lora-adapter/              # (created after you run finetune_lora.py)
└── results/
    └── comparison.json            # (created after you run compare_models.py)
```

---

## 2. Quick Start — Chat App with Free Google API (no GPU needed)

### Step A — Get a free Google Gemini API key
1. Go to **https://aistudio.google.com/app/apikey**
2. Sign in with a Google account and click **"Create API key"**.
3. Copy the key.

### Step B — Put the key in the project
1. In the project folder, copy `.env.example` to a new file named `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and paste your key:
   ```
   GOOGLE_API_KEY=your_actual_key_here
   ```
   **This is the ONLY file you need to edit to get the app running.**

### Step C — Install dependencies & run
```bash
pip install -r requirements.txt
python app.py
```
This opens a local Gradio web app (usually at `http://127.0.0.1:7860`) where
you can chat with the assistant.

---

## 3. Dataset (Step 1 & 2 — Preparation, Cleaning, Formatting)

- **Source**: `data/raw_dataset.json` — 41 hand-curated question/answer
  pairs covering core CS & programming topics relevant to engineering
  students: variables, data structures (lists, tuples, stacks, queues,
  linked lists, hash tables), algorithms (recursion, binary search, Big-O,
  dynamic programming), OOP (classes, inheritance, polymorphism), web/API
  concepts (REST, GET/POST), databases (SQL vs NoSQL, normalization,
  indexing), and study/debugging advice.
- **Cleaning** (`scripts/prepare_data.py`): strips whitespace, collapses
  repeated spaces/newlines, removes duplicate questions, and validates
  that both a question and an answer exist for every row.
- **Formatting**: each example is converted into:
  1. A **chat-style `messages` list** (`system` / `user` / `assistant`
     roles) — the format most modern instruction-tuned models expect.
  2. A flattened **`text`** field wrapped in `<|system|>` / `<|user|>` /
     `<|assistant|>` tags — used directly by the LoRA trainer
     (Hugging Face TRL's `SFTTrainer`).
- **Output**: `data/train.jsonl` (36 examples) and `data/val.jsonl`
  (5 examples), a 90/10 split.

Run it yourself:
```bash
python scripts/prepare_data.py
```

---

## 4. Fine-Tuning (Step 3 — LoRA/QLoRA) — Optional, needs a GPU

**Base model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (small, open, and
fast enough to fine-tune on a free Google Colab T4 GPU in a few minutes).

**Method**: LoRA via Hugging Face **PEFT**, wrapped with TRL's
`SFTTrainer`. If a CUDA GPU is detected, the script automatically loads
the base model in 4-bit (QLoRA) using `bitsandbytes` to save memory;
otherwise it falls back to plain LoRA in full precision.

- LoRA rank `r=16`, `alpha=32`, dropout `0.05`
- Target modules: `q_proj, k_proj, v_proj, o_proj`
- 3 epochs, batch size 2 with gradient accumulation 4, learning rate `2e-4`

**How to run** (on Google Colab, free tier, with a GPU runtime selected):
```bash
pip install -r requirements.txt -r requirements-finetune.txt
python scripts/prepare_data.py         # if not already run
python scripts/finetune_lora.py
```
This saves a small LoRA adapter (a few MB, not the full model) to
`models/lora-adapter/`.

---

## 5. Testing & Comparison (Step 4 & 5)

`scripts/compare_models.py` runs a fixed set of 8 programming questions
through **both** the base model and the fine-tuned model, and saves the
side-by-side answers to `results/comparison.json`.

```bash
python scripts/compare_models.py
```

**What to look for when comparing**: the fine-tuned model should give
answers that more closely match the concise, example-driven style of our
training data, whereas the untuned base model (a small 1.1B chat model)
tends to be more generic, occasionally rambling, or drift off-topic since
it wasn't trained on this style of Q&A.

> Note: with only ~36 training examples this is a **demonstration-scale**
> fine-tune, meant to show the full pipeline working correctly — not a
> production-grade model. For a real deployment you'd want thousands of
> examples.

---

## 6. Interface (Step 6) — Redesigned, Multi-Tab UI

`app.py` launches a polished **Gradio** app with four tabs:

- **💬 Chat** — talk to the assistant with full **conversation history**.
  Pick your backend (Gemini API or local fine-tuned model) and, for Gemini,
  pick the **exact model** from a live-refreshed dropdown (🔄 button calls
  `list_available_models()`, so it never goes stale like a hardcoded name
  would). If the assistant's answer contains a code block, a
  **"▶ Run code from last answer"** button appears — click it and the code
  executes immediately, with output shown right below the chat.
- **🧪 Code Playground** — a full code editor (Python / JavaScript / Bash,
  whichever interpreters are installed) with a **Run** button and live
  stdout/stderr output. Great for testing what the assistant explains.
- **📊 Compare Models** — renders `results/comparison.json` (produced by
  `compare_models.py`) as an easy-to-read base-vs-fine-tuned report.
- **ℹ️ About** — quick project summary.

### Code execution safety note
Code runs in an isolated subprocess with a wall-clock timeout (default 8s)
in its own temp directory (`scripts/code_runner.py`). This is a
best-effort sandbox appropriate for a personal/student tool — it isolates
runaway loops and reports crashes cleanly, but it does **not** block
filesystem or network access. Don't expose this app to untrusted public
users without a hardened sandbox (Docker/gVisor/nsjail).

---

## 7. Bonus Features Implemented

- **Conversation history**: the chat keeps context across turns (see
  `respond()` in `app.py`, which passes prior turns to both backends).
- **Lightweight "grounding" for the API path**: `gemini_backend.py`
  automatically pulls a few examples from our own training data as
  few-shot examples in every Gemini call, steering its answers toward the
  same concise, example-driven style as the fine-tuned model — useful
  since not everyone has a GPU to run the full fine-tuning step.
- **Model evaluation**: `compare_models.py` produces a structured,
  saved JSON comparison of base vs. fine-tuned answers on a fixed test set.

### Ideas for further extension
- **RAG**: swap the few-shot examples in `gemini_backend.py` for a proper
  vector-store retrieval step (e.g. with `langchain` + `chromadb`) over a
  larger knowledge base of course notes/documentation.
- **Deployment**: host `app.py` on Hugging Face Spaces (Gradio SDK) or
  Streamlit Community Cloud, adding `GOOGLE_API_KEY` as a secret.
- **Inference optimization**: merge the LoRA adapter into the base model
  weights (`model.merge_and_unload()`) and export to GGUF for fast local
  inference with `llama.cpp`.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `models/gemini-1.5-flash is not found` or similar 404 model error | Google periodically retires model names. Click the **🔄 refresh button** next to the Gemini model dropdown in the app — it calls the API directly to list models your key currently supports, and picks a working one automatically. |
| `GOOGLE_API_KEY is not set` error in the app | Make sure you copied `.env.example` to `.env` (not just edited the example) and pasted a real key, no quotes. |
| `ModuleNotFoundError: No module named 'gradio'` | Run `pip install -r requirements.txt` inside your project folder/virtual environment. |
| Fine-tuning step is extremely slow / crashes | You need a GPU. Use a free Google Colab notebook (Runtime → Change runtime type → GPU) and re-upload the project files there. |
| "Local fine-tuned model" option doesn't appear in the app | That's expected until you actually run `scripts/finetune_lora.py` — the option only appears once `models/lora-adapter/` exists. |
