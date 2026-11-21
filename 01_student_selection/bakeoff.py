"""Step 1: run the candidate students (prompted, no fine-tuning) on the test set.

Usage (from the repo root, with Ollama running and the models pulled):
    python 01_student_selection/bakeoff.py

Writes results/bakeoff_results.csv: one row per sentence; output, validity check and
convention warnings per model.
"""

from pathlib import Path

from emoji_eval.runner import run_bakeoff

MODELS = [
    # (display name, provider, model id)
    ("SmolLM3-3B", "ollama", "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"),  # negative control
    ("ministral-3:3b", "ollama", "ministral-3:3b"),
    ("qwen3.5-2b", "ollama", "qwen3.5:2b"),
    ("qwen3.5-0.8b", "ollama", "qwen3.5:0.8b"),
]

RESULTS_FILE = Path(__file__).resolve().parent / "results" / "bakeoff_results.csv"

if __name__ == "__main__":
    run_bakeoff(MODELS, RESULTS_FILE)
