#!/usr/bin/env python3
"""We run the full screen time and EAH workflow and save outputs in results/."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
INPUTS = (
    "k_ab_bmi.xlsx",
    "total_media_use.csv",
    "K01_MVPA_BL_051425.csv",
    "K01_SleepSummaries_2023-01-20.csv",
    "5K01MediaUseFullStud_DATA_2020-06-22_1116.csv",
)

# We run the primary analysis first so the mediation table and figures use
# the model estimates from this run. We keep exploratory checks separate.
PIPELINE = (
    ("k01_analysis_corrected.py", (), "01_main_analysis"),
    ("cgd_analysis.py", (), "02_cgd"),
    ("make_tables_characteristics.py", (), "03_table1"),
    ("make_table_mediation.py", (), "04_table2"),
    ("make_figures.py", (), "05_figures"),
    ("kcal_backtransform.py", (), "06_kcal"),
    ("sensitivity_analysis.py", (), "07_sensitivity"),
    ("influence_analysis.py", (), "08_influence"),
    ("model_diagnostics.py", (DATA / INPUTS[0], DATA / INPUTS[1]), "09_diagnostics"),
    ("format_tables.py", (), "10_format_tables"),
)


def main():
    # We check all required inputs before creating any output. This prevents a
    # missing accelerometer or preload file from silently omitting an analysis.
    missing = [name for name in INPUTS if not (DATA / name).is_file()]
    if missing:
        sys.exit(f"Place these files in {DATA}:\n" + "\n".join(missing))

    # We keep existing directories. The scripts overwrite their named outputs.
    LOGS.mkdir(parents=True, exist_ok=True)
    for script, args, log_name in PIPELINE:
        path = ROOT / "scripts" / script
        log_path = LOGS / f"{log_name}.log"
        print(f"Running {script}", flush=True)
        # We use the active Python environment and capture both output streams
        # so each step can be checked without mixing its log with another step.
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, str(path), *map(str, args)],
                cwd=path.parent, stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            sys.exit(f"{script} failed. See {log_path}.")
    print(f"Complete. Results are in {RESULTS}.")


if __name__ == "__main__":
    main()
