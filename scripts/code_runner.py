"""
code_runner.py
----------------
Powers the "Code Playground" tab and the inline "▶ Run this code" button
in the chat: executes a snippet of code in an isolated subprocess with a
timeout and captures stdout/stderr/return code.

This is a *best-effort* sandbox suitable for a student learning tool, not
a hardened multi-tenant sandbox: it isolates the code in its own process
and working directory and enforces a wall-clock timeout, but does not
block filesystem or network access. Don't expose this to untrusted
public users without adding a real sandbox (e.g. Docker, gVisor, nsjail).

Supported languages: python, javascript (via node), bash.
"""

import subprocess
import sys
import tempfile
import os
import shutil
import time

LANGUAGE_CONFIG = {
    "python": {
        "ext": ".py",
        "cmd": lambda path: [sys.executable, path],
    },
    "javascript": {
        "ext": ".js",
        "cmd": lambda path: ["node", path],
    },
    "bash": {
        "ext": ".sh",
        "cmd": lambda path: ["bash", path],
    },
}

DEFAULT_TIMEOUT = 8  # seconds


def available_languages() -> list[str]:
    """Only offer languages whose interpreter is actually installed."""
    langs = []
    for lang, cfg in LANGUAGE_CONFIG.items():
        interpreter = cfg["cmd"]("__probe__")[0]
        if lang == "python" or shutil.which(interpreter):
            langs.append(lang)
    return langs


def run_code(code: str, language: str = "python", timeout: int = DEFAULT_TIMEOUT) -> dict:
    """
    Execute `code` in the given language. Returns a dict with:
        stdout, stderr, returncode, timed_out (bool), elapsed_seconds (float)
    """
    if not code or not code.strip():
        return {"stdout": "", "stderr": "(no code provided)", "returncode": -1,
                "timed_out": False, "elapsed_seconds": 0.0}

    if language not in LANGUAGE_CONFIG:
        return {"stdout": "", "stderr": f"Unsupported language: {language}",
                "returncode": -1, "timed_out": False, "elapsed_seconds": 0.0}

    cfg = LANGUAGE_CONFIG[language]
    tmp_dir = tempfile.mkdtemp(prefix="ai_assistant_run_")
    file_path = os.path.join(tmp_dir, f"snippet{cfg['ext']}")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        cmd = cfg["cmd"](file_path)
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.time() - start
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "timed_out": False,
                "elapsed_seconds": round(elapsed, 3),
            }
        except subprocess.TimeoutExpired as e:
            return {
                "stdout": e.stdout or "",
                "stderr": (e.stderr or "") + f"\n[Execution timed out after {timeout}s]",
                "returncode": -1,
                "timed_out": True,
                "elapsed_seconds": timeout,
            }
        except FileNotFoundError:
            return {
                "stdout": "",
                "stderr": f"Interpreter for '{language}' is not installed on this machine.",
                "returncode": -1,
                "timed_out": False,
                "elapsed_seconds": 0.0,
            }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def format_result(result: dict) -> str:
    """Pretty-print a run_code() result for display in the UI."""
    parts = []
    if result["stdout"]:
        parts.append(result["stdout"].rstrip())
    if result["stderr"]:
        parts.append("── stderr ──\n" + result["stderr"].rstrip())
    parts.append(
        f"\n[exit code: {result['returncode']} | {result['elapsed_seconds']}s"
        f"{' | TIMED OUT' if result['timed_out'] else ''}]"
    )
    return "\n".join(parts) if parts else "(no output)"
