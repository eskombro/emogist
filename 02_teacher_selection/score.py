"""Step 2: round-trip score of the teacher outputs (Haiku decodes, Sonnet judges the gist).

Usage (from the repo root, ANTHROPIC_API_KEY in .env):
    python 02_teacher_selection/score.py            # score results/bakeoff_results.csv
    python 02_teacher_selection/score.py --sanity   # check the scorer on the few-shot examples

Writes results/roundtrip_scores.csv (per sentence: decode, gist score, judge's reason).
"""

import argparse
from pathlib import Path

from emoji_eval.roundtrip import make_clients, sanity_check, score_results

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "bakeoff_results.csv"
SCORES_FILE = RESULTS_DIR / "roundtrip_scores.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sanity", action="store_true", help="check the scorer on the few-shot examples")
    args = parser.parse_args()

    client, embedder = make_clients()
    if args.sanity:
        sanity_check(client, embedder)
    else:
        score_results(RESULTS_FILE, SCORES_FILE, client, embedder)
