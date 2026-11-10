# Scripts for manuscript "Screen Time Is Associated with Eating in the Absence of Hunger in Preschoolers: Attentional Bias to Food Cues as a Candidate Mechanism"

We use these scripts for the analyses, tables, and statistical figures for the preschool screen time study.

## Set up and run

1. Extract the archive. Open a terminal in the `EAH_Public_Pipeline` folder.
2. Create a folder named `data` beside `scripts` and `dependencies`. Obtain the study files separately through the applicable data access process and place them in `data` using the names below.
3. From the project root, run these commands. We verified this environment with Python 3.12.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r dependencies/requirements.txt
.venv/bin/python scripts/run_all.py
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

| File in `data/` | What we read |
| --- | --- |
| `k_ab_bmi.xlsx` | Participant analysis workbook. We read EAH in kcal, food versus toy eye tracking measures, and baseline covariates, including `bmip_bl` for BMI percentile. |
| `total_media_use.csv` | Weekly screen use in minutes, converted to hours per day. |
| `K01_MVPA_BL_051425.csv` | Baseline accelerometry for daily moderate to vigorous physical activity. |
| `K01_SleepSummaries_2023-01-20.csv` | Nighttime sleep summaries. |
| `5K01MediaUseFullStud_DATA_2020-06-22_1116.csv` | Baseline preload energy intake in kcal. The figure script uses the study ID, event if present, and `preload_total_kcal`. |

We link files by the same study ID and use the expected column names and coding in the released inputs. We use only the necessary columns from the preload export. Install Times New Roman on your system before running for the requested figure typography. Matplotlib warns and uses a serif fallback if the font is unavailable. Word tables specify Times New Roman in their formatting.

## Outputs

We create `results/` when the pipeline runs. Each step writes a log to `results/logs/`. We overwrite named outputs on rerun and leave other files alone.

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
