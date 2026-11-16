# Screen Time Is Associated with Eating in the Absence of Hunger in Preschoolers: Attentional Bias to Food Cues as a Candidate Mechanism

We use these scripts for the analyses, tables, and statistical figures for the preschool screen time study.

## Set up and run

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r dependencies/requirements.txt
.venv/bin/python scripts/run_all.py
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.
| File in `data/` | What we read |
| --- | --- |
| `participant_eah_eyetracking.xlsx` | Participant analysis workbook. We read EAH in kcal, food versus toy eye tracking measures, and baseline covariates, including `bmip_bl` for BMI percentile. |
| `weekly_screen_time.csv` | Weekly screen use in minutes, converted to hours per day. |
| `baseline_physical_activity.csv` | Baseline accelerometry for daily moderate to vigorous physical activity. |
| `sleep_duration.csv` | Nighttime sleep summaries. |
| `preload_energy_intake.csv` | Baseline preload energy intake in kcal. The figure script uses the study ID, event if present, and `preload_total_kcal`. |

We link records by the same study ID and use the expected column names and coding in the released inputs. We use only the necessary columns from the preload export. Install Times New Roman on your system before running for the requested figure typography. Matplotlib warns and uses a serif fallback if the font is unavailable. Word tables specify Times New Roman in their formatting.

## How filenames are configured

All input filenames are defined **once** in `root_folder/scripts/input_paths.py` 
The script `scripts/run_all.py` checks all five files before running the pipeline.

Participant name/IDs are masked with random IDs. Race and Ethnicity are not included in the data files for privacy concerns.


## Outputs

We create `results/` when the pipeline runs.

| Manuscript item | Generated file in `results/` |
| --- | --- |
| Table 1 | `tables/table1_detailed.csv` and `tables/table1_detailed.docx` |
| Supplementary Table S1 | `tables/table_s1_mediation.docx` and the detailed `tables/table2_mediation.csv` |
| Supplementary Note S2 sensitivity values | `main_analysis/table_s2_log_offset_sweep.csv` |
| Figure 3 | `figures/fig_aim1_screen_eah.png` |
| Figure 4 | `figures/fig_4_mediation.png` and its two component panels |
| Supplementary Figures S1, S2, S3 | `figures/fig_qq_eah_distributions.png`, `figures/fig_bars_ff.png`, `figures/fig_bars_cgd.png` |
| Supplementary Figures S4, S5, S6 | `figures/fig_med_ff.png`, `figures/fig_med_cgd.png`, `figures/fig_preload.png` |

Figures 1 and 2 are the study flow diagram and task screenshot, supplied with the manuscript. We generate the statistical plots at 300 dpi, with EAH and preload on kcal axes, eye tracking measures in milliseconds, and screen time in hours per day. We also retain exploratory analyses, diagnostics, and machine readable model outputs in their own results folders.
