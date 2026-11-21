"""Step 2: run the candidate teachers on the same test set as the students.

Teachers run on Ollama Cloud, through the local Ollama (same API as the students):
    ollama signin                 # once; add credit at ollama.com (pay per token)
    python 02_teacher_selection/bakeoff.py

Writes results/bakeoff_results.csv, in the same format as step 1, so the same scorer
and reports apply. Cost: well under $1 for all three teachers (100 short requests each).
"""

from pathlib import Path

from emoji_eval.runner import run_bakeoff

MODELS = [
    # (display name, provider, model id). Pinned tags where Ollama offers one, so the
    # results stay tied to a known model version.
    # Mistral Large 4 on Ollama Cloud mostly ignores "think": false: it reasons for
    # 300-600 tokens before answering, so it needs a bigger budget than 100 tokens.
    ("mistral-large-4", "ollama", "mistral-large-4:cloud", {"max_tokens": 2000}),
    ("deepseek-v4-pro", "ollama", "deepseek-v4-pro:0813-cloud"),
    # The same model with thinking on, to measure whether reasoning improves the labels.
    # Only the final answer is kept: the thinking never goes into the dataset.
    ("deepseek-v4-pro-think", "ollama", "deepseek-v4-pro:0813-cloud", {"max_tokens": 3000, "think": True}),
    ("deepseek-v4.1-flash", "ollama", "deepseek-v4.1-flash:cloud"),
]

RESULTS_FILE = Path(__file__).resolve().parent / "results" / "bakeoff_results.csv"

if __name__ == "__main__":
    run_bakeoff(MODELS, RESULTS_FILE)
