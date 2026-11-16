#!/usr/bin/env python3
"""We run the full screen time and EAH workflow and save outputs in results/."""

from input_paths import ROOT, DATA_DIR, RESULTS_DIR, data_file, missing_inputs
import subprocess
import sys

DATA = DATA_DIR
RESULTS = RESULTS_DIR
LOGS = RESULTS / "logs"

# We run the primary analysis first so the mediation table and figures use
# the model estimates from this run. We keep exploratory checks separate.
PIPELINE = (
    ("main_analysis.py", (), "01_main_analysis"),
    ("cgd_analysis.py", (), "02_cgd"),
    ("make_tables_characteristics.py", (), "03_table1"),
    ("make_table_mediation.py", (), "04_table2"),
    ("make_figures.py", (), "05_figures"),
    ("kcal_backtransform.py", (), "06_kcal"),
    ("sensitivity_analysis.py", (), "07_sensitivity"),
    ("influence_analysis.py", (), "08_influence"),
    ("model_diagnostics.py", (data_file("participant"), data_file("screen_time")), "09_diagnostics"),
    ("format_tables.py", (), "10_format_tables"),
)


def main():
    # We check all required inputs before creating any output. This prevents a
    # missing accelerometer or preload file from silently omitting an analysis.
    missing = missing_inputs()
    if missing:
        sys.exit(f"Place these files in {DATA}:\n" + "\n".join(str(path) for path in missing))

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
            # Print the relevant error as well as its full log location.
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            detail = "\n".join(lines[-12:]) or "No error text was captured."
            sys.exit(f"{script} failed (exit {result.returncode}).\n{detail}\nFull log: {log_path}")
    print(f"Complete. Results are in {RESULTS}.")


if __name__ == "__main__":
    main()
