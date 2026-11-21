"""Project paths, so scripts work from any working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
TEST_FILE = DATA_DIR / "test_sentences.csv"
ENV_FILE = ROOT / ".env"
