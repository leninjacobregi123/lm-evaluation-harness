import os
from pathlib import Path

# Project root path detection
BACKEND_DIR = Path(__file__).parent.parent
PROJECT_ROOT = (BACKEND_DIR / "../..").resolve()

# Directory where benchmark results are saved
RESULTS_DIR = (PROJECT_ROOT / "eval_results").resolve()
os.makedirs(RESULTS_DIR, exist_ok=True)

# Path to python virtual environment interpreter
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path("python3")

# Path to the multi-run benchmarking pipeline script
PIPELINE_SCRIPT = PROJECT_ROOT / "benchmark_pipeline.py"

# CORS configurations
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

# Legacy filenames to check for fallback api keys in the project root
APIKEY_FALLBACK_FILES = ["apikey", "college key"]
