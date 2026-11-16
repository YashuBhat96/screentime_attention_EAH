"""Input filenames used throughout this analysis pipeline.

To rename an input again, change ONLY the filename here, and rename the matching
file in data/. Column names, subject identifiers, and statistical models remain unchanged.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

INPUT_FILES = {
    "participant": "participant_eah_eyetracking.xlsx",
    "screen_time": "weekly_screen_time.csv",
    "physical_activity": "baseline_physical_activity.csv",
    "sleep": "sleep_duration.csv",
    "preload": "preload_energy_intake.csv",
}


def data_file(name: str) -> Path:
    """Return the absolute path for a named analysis input."""
    return DATA_DIR / INPUT_FILES[name]


def missing_inputs() -> list[Path]:
    """List missing required inputs for early pipeline validation."""
    return [data_file(key) for key in INPUT_FILES if not data_file(key).is_file()]
