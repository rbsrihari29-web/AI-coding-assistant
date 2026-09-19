"""
app.py
--------
The main interface for the AI Coding Assistant — a redesigned, multi-tab
experience with:

  💬 Chat            - talk to the assistant (Gemini API or your local
                        fine-tuned LoRA model), with conversation history,
                        model picker, and auto-detected runnable code blocks.
  🧪 Code Playground  - write and RUN code (Python / JavaScript / Bash)
                        right in the app, no separate IDE needed.
  📊 Compare Models   - view saved base-vs-fine-tuned comparison results.
  ℹ️ About            - project/model/dataset info.

Run:
    pip install -r requirements.txt
    cp .env.example .env      # then paste your Gemini API key into .env
    python app.py
"""

import sys
import os
import json
import re
import importlib.util
import importlib

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

import gradio as gr

# Import modules from the `scripts/` folder without triggering static import-resolution errors
# in editors like VS Code/Pylance.
gemini_backend = importlib.import_module("gemini_backend")
code_runner = importlib.import_module("code_runner")

ask_gemini = gemini_backend.ask_gemini
list_available_models = gemini_backend.list_available_models
run_code = code_runner.run_code
format_result = code_runner.format_result
available_languages = code_runner.available_languages

_local_backend_path = os.path.join(os.path.dirname(__file__), "scripts", "local_model_backend.py")
local_model_backend = None
if os.path.exists(_local_backend_path):
    spec = importlib.util.spec_from_file_location("local_model_backend", _local_backend_path)
    local_model_backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_model_backend)

LOCAL_AVAILABLE = bool(local_model_backend and local_model_backend.is_available())
BACKEND_CHOICES = ["Gemini API (recommended)"] + (
    ["Local fine-tuned model"] if LOCAL_AVAILABLE else []
)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "comparison.json")
LANGS = available_languages()

CODE_BLOCK_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

CUSTOM_CSS = """
#header-banner {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #0ea5e9 100%);
    padding: 22px 28px; border-radius: 14px; margin-bottom: 6px;
}
#header-banner h1 { color: white !important; margin: 0 0 4px 0; }
#header-banner p { color: #e0e7ff !important; margin: 0; }
.status-pill {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 12px; font-weight: 600; background: #10b98122; color: #059669;
    border: 1px solid #10b98155;
}
#run-output textarea { font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; }
.gradio-container { max-width: 1200px !important; margin: auto; }
"""


# ----------------------------------------------------------------------
# Chat logic
# ----------------------------------------------------------------------

def extract_code_blocks(text: str):
    """Return list of (language, code) tuples found in a markdown response."""
    blocks = []
    for m in CODE_BLOCK_RE.finditer(text or ""):
        lang = (m.group(1) or "python").lower()
        lang = {"js": "javascript", "sh": "bash", "shell": "bash", "py": "python"}.get(lang, lang)
        if lang not in LANGS:
            lang = "python"
        blocks.append((lang, m.group(2).strip()))
    return blocks


def respond(message, chat_history, backend_choice, gemini_model):
    if not message or not message.strip():
        return "", chat_history, gr.update()

    tuple_history = []
    for i in range(0, len(chat_history) - 1, 2):
        if chat_history[i]["role"] == "user" and chat_history[i + 1]["role"] == "assistant":
            tuple_history.append((chat_history[i]["content"], chat_history[i + 1]["content"]))

    try:
        if backend_choice == "Local fine-tuned model" and LOCAL_AVAILABLE:
            answer = local_model_backend.ask_local_model(message, tuple_history)
        else:
            answer = ask_gemini(message, tuple_history, model_name=gemini_model)
    except Exception as e:
        answer = (
            f"⚠️ **Error:** {e}\n\n"
            "If this mentions `GOOGLE_API_KEY`, copy `.env.example` to `.env` and "
            "paste a valid free key from https://aistudio.google.com/app/apikey.\n\n"
            "If this mentions a **model not found**, click 🔄 next to the model "
            "dropdown to refresh the list of models your key actually supports."
        )

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})

    code_blocks = extract_code_blocks(answer)
    run_btn_update = gr.update(visible=bool(code_blocks),
                                value=f"▶ Run code from last answer ({len(code_blocks)} block{'s' if len(code_blocks) != 1 else ''})")
    return "", chat_history, run_btn_update


def run_last_code_block(chat_history):
    """Grab the last assistant message's first code block and execute it."""
    for msg in reversed(chat_history):
        if msg["role"] == "assistant":
            blocks = extract_code_blocks(msg["content"])
            if blocks:
                lang, code = blocks[0]
                result = run_code(code, lang)
                return code, lang, format_result(result)
            break
    return "", "python", "(No runnable code block found in the last answer.)"


def clear_chat():
    return [], gr.update(visible=False)


def refresh_models():
    models = list_available_models()
    return gr.update(choices=models, value=models[0] if models else None)


# ----------------------------------------------------------------------
# Code Playground logic
# ----------------------------------------------------------------------

def playground_run(code, language):
    result = run_code(code, language)
    return format_result(result)


SNIPPETS = {
    "python": "# Try me!\nfor i in range(5):\n    print(f\"square of {i} is {i**2}\")",
    "javascript": "// Try me!\nfor (let i = 0; i < 5; i++) {\n  console.log(`square of ${i} is ${i*i}`);\n}",
    "bash": "# Try me!\nfor i in 1 2 3; do echo \"hello $i\"; done",
}


def load_snippet(language):
    return SNIPPETS.get(language, "")


# ----------------------------------------------------------------------
# Compare tab logic
# ----------------------------------------------------------------------

def load_comparison_table():
    if not os.path.exists(RESULTS_PATH):
        return ("No comparison results yet. Run `python scripts/compare_models.py` "
                "on a GPU machine (after fine-tuning) to generate `results/comparison.json`.")
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    md = ""
    for row in data:
        md += f"### ❓ {row['question']}\n\n"
        md += f"**Base model:**\n> {row['base_model']}\n\n"
        md += f"**Fine-tuned model:**\n> {row['fine_tuned_model']}\n\n---\n\n"
    return md


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

with gr.Blocks(title="AI Coding Assistant", css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.HTML(
        """
        <div id="header-banner">
            <h1>🤖 AI Coding Assistant</h1>
            <p>Ask questions, get explanations, and run code — all in one place, built for engineering students.</p>
        </div>
        """
    )

    with gr.Tabs():
        # ---------------- CHAT TAB ----------------
        with gr.TabItem("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Conversation", height=480, avatar_images=(None, "🤖"))
                    msg = gr.Textbox(
                        label="Your question",
                        placeholder="e.g. What is the difference between a stack and a queue? Show me code.",
                        lines=2,
                    )
                    with gr.Row():
                        submit_btn = gr.Button("Ask", variant="primary")
                        clear_btn = gr.Button("Clear chat")
                    run_last_btn = gr.Button("▶ Run code from last answer", visible=False, variant="secondary")

                    gr.Examples(
                        examples=[
                            "What is recursion? Show me a Python code example.",
                            "Explain Big O notation with an example.",
                            "Write a Python function to check if a number is prime, and explain it.",
                            "What is the difference between a list and a tuple in Python?",
                            "How do I debug my code effectively?",
                        ],
                        inputs=msg,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Settings")
                    backend_dropdown = gr.Dropdown(
                        choices=BACKEND_CHOICES, value=BACKEND_CHOICES[0], label="Model backend",
                    )
                    with gr.Row():
                        gemini_model_dropdown = gr.Dropdown(
                            choices=["gemini-2.5-flash"], value="gemini-2.5-flash",
                            label="Gemini model", scale=4,
                        )
                        refresh_btn = gr.Button("🔄", scale=1, min_width=40)
                    gr.Markdown(
                        "<span class='status-pill'>Local fine-tuned model "
                        + ("detected ✅" if LOCAL_AVAILABLE else "not found — run finetune_lora.py")
                        + "</span>"
                    )
                    gr.Markdown(
                        "---\n**Tip:** Ask for code and a **▶ Run code** button will "
                        "appear so you can execute the assistant's answer instantly."
                    )
                    last_run_lang = gr.Textbox(visible=False)

            with gr.Accordion("🖥️ Code output", open=True, visible=False) as run_output_acc:
                run_code_preview = gr.Code(label="Code that ran", language="python")
                run_output_box = gr.Textbox(label="Output", lines=8, elem_id="run-output", interactive=False)

            demo.load(refresh_models, None, gemini_model_dropdown)
            refresh_btn.click(refresh_models, None, gemini_model_dropdown)

            submit_btn.click(respond, [msg, chatbot, backend_dropdown, gemini_model_dropdown],
                              [msg, chatbot, run_last_btn])
            msg.submit(respond, [msg, chatbot, backend_dropdown, gemini_model_dropdown],
                       [msg, chatbot, run_last_btn])
            clear_btn.click(clear_chat, None, [chatbot, run_last_btn])

            def _run_and_show(chat_history):
                code, lang, output = run_last_code_block(chat_history)
                return gr.update(visible=True), code, output

            run_last_btn.click(_run_and_show, chatbot, [run_output_acc, run_code_preview, run_output_box])

        # ---------------- CODE PLAYGROUND TAB ----------------
        with gr.TabItem("🧪 Code Playground"):
            gr.Markdown(
                "Write code, hit **Run**, see the output instantly. "
                "Great for testing what the assistant explains, or just practicing. "
                f"Supported languages on this machine: **{', '.join(LANGS)}**."
            )
            with gr.Row():
                pg_lang = gr.Dropdown(choices=LANGS, value=LANGS[0] if LANGS else "python", label="Language", scale=1)
                pg_load_btn = gr.Button("Load example snippet", scale=1)
            pg_code = gr.Code(label="Code editor", language="python", value=SNIPPETS.get("python", ""), lines=16)
            pg_run_btn = gr.Button("▶ Run", variant="primary")
            pg_output = gr.Textbox(label="Output", lines=10, elem_id="run-output", interactive=False)

            pg_lang.change(lambda l: gr.update(language=l if l in ("python", "javascript") else None), pg_lang, pg_code)
            pg_load_btn.click(load_snippet, pg_lang, pg_code)
            pg_run_btn.click(playground_run, [pg_code, pg_lang], pg_output)

        # ---------------- COMPARE TAB ----------------
        with gr.TabItem("📊 Compare Models"):
            gr.Markdown("Base model vs. LoRA fine-tuned model, side by side, on a fixed test set.")
            refresh_compare_btn = gr.Button("🔄 Reload results")
            compare_md = gr.Markdown(load_comparison_table())
            refresh_compare_btn.click(load_comparison_table, None, compare_md)

        # ---------------- ABOUT TAB ----------------
        with gr.TabItem("ℹ️ About"):
            gr.Markdown(
                """
                ## About this project
                An AI-powered assistant for engineering students, built around:
                - A **curated programming/CS dataset** (`data/raw_dataset.json`)
                - **LoRA/QLoRA fine-tuning** of `TinyLlama-1.1B-Chat` (`scripts/finetune_lora.py`)
                - A **free Gemini API** fallback so the app works without a GPU
                - A **Code Playground** to run Python / JavaScript / Bash snippets
                - **Conversation history** and one-click **run code from chat**

                See `README.md` in the project folder for full setup and documentation.
                """
            )

if __name__ == "__main__":
    if not LOCAL_AVAILABLE:
        print(
            "Note: No local fine-tuned model found at models/lora-adapter. "
            "Using Gemini API backend only. Run scripts/finetune_lora.py on a "
            "GPU machine (e.g. Colab) if you want the local backend too."
        )
    demo.launch()
