#!/usr/bin/env python3
"""
We check whether the consumer sample association is sensitive to additional covariates, including physical activity, sleep, and reported eating behaviors. We interpret these models as exploratory because covariate availability changes the analysis sample.

Run the full workflow with scripts/run_all.py.
"""
import os, re
import numpy as np, pandas as pd
from scipy import stats

# ------------------------------------------------------------------ CONFIG
BASE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DATA      = os.path.join(BASE, 'k_ab_bmi.xlsx')
SCREEN    = os.path.join(BASE, 'total_media_use.csv')
MVPA_F    = os.path.join(BASE, 'K01_MVPA_BL_051425.csv')
SLEEP_F   = os.path.join(BASE, 'K01_SleepSummaries_2023-01-20.csv')
LOG_OFFSET = 30          # log(EAH + 30); set to 0 for plain log(EAH) among eaters

# ------------------------------------------------------------------ HC3 OLS
def hc3_ols(y, X_df):
    """OLS point estimates with HC3 heteroskedasticity-consistent SEs.
    Returns DataFrame indexed by term with beta, se, t, p."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(X_df)), X_df.values.astype(float)])
    names = ['const'] + list(X_df.columns)
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    resid = y - X @ beta
    h = np.clip(np.diag(X @ XtXi @ X.T), None, 0.9999)
    e = resid / (1 - h)                       # HC3 adjustment
    meat = (X * e[:, None]).T @ (X * e[:, None])
    V = XtXi @ meat @ XtXi
    se = np.sqrt(np.diag(V))
    df = len(y) - X.shape[1]
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    return pd.DataFrame({'beta': beta, 'se': se, 't': t, 'p': p}, index=names), df

# ------------------------------------------------------------------ LOAD + BUILD SAMPLE
raw = pd.read_excel(DATA)
# drop pre-existing columns whose names would collide with our renamed ones
for dup in ['income', 'age', 'bmi_percentile', 'eah']:
    if dup in raw.columns and dup not in ('income_imputed','conf_age','bmip_bl','eah_total_kcal'):
        raw = raw.drop(columns=[dup])
df = raw.rename(columns={'conf_age':'age','bmip_bl':'bmi_percentile','income_imputed':'income',
                         'eah_total_kcal':'eah'}).copy()
# de-duplicate any remaining repeated column names (keep first)
df = df.loc[:, ~df.columns.duplicated()]
for c in ['age','bmi_percentile','income','eah']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['female'] = (pd.to_numeric(df['gender'], errors='coerce') - 1).clip(0, 1)
df['rid'] = pd.to_numeric(df['redcapid'], errors='coerce')

# We convert weekly minutes to hours per day to use the same exposure units as the manuscript.
# screen time (min/week -> hr/day), baseline rows
media = pd.read_csv(SCREEN)
if 'redcap_event_name' in media.columns:
    media = media[media['redcap_event_name'].astype(str).str.contains('baseline', case=False, na=False)]
media['screentime'] = pd.to_numeric(media['mediatotalperweek'], errors='coerce') / 60 / 7
media['rid'] = pd.to_numeric(media['redcapid'], errors='coerce')
df = df.merge(media[['rid','screentime']].dropna().drop_duplicates('rid'), on='rid', how='left')

# MVPA (min/day) from accelerometry
try:
    mv = pd.read_csv(MVPA_F, skiprows=1)
    mv = mv[mv['Subject Name'].astype(str).str.startswith('K01M', na=False)].copy()
    mv['rid'] = mv['Subject Name'].str.extract(r'K01M0*(\d+)').astype(float)
    def _hms(s):
        m = re.match(r'(\d+)H\s+(\d+)M\s+(\d+)S', str(s))
        return int(m.group(1))*60 + int(m.group(2)) + int(m.group(3))/60 if m else np.nan
    mv['mvpa'] = mv['Average MVPA Per Day'].apply(_hms)
    df = df.merge(mv[['rid','mvpa']].dropna().drop_duplicates('rid'), on='rid', how='left')
except Exception as e:
    print("WARN MVPA:", e); df['mvpa'] = np.nan

# sleep (mean nightly total sleep time, min) from accelerometry summaries
try:
    sl = pd.read_csv(SLEEP_F)
    sl['rid'] = sl['PID'].astype(str).str.extract(r'K01M0*(\d+)').astype(float)
    slp = sl.groupby('rid')['total.sleep.time'].mean().reset_index().rename(columns={'total.sleep.time':'sleep'})
    df = df.merge(slp, on='rid', how='left')
except Exception as e:
    print("WARN sleep:", e); df['sleep'] = np.nan

# We examine parent reported food responsiveness and satiety responsiveness as additional
# exploratory covariates.
# CEBQ subscales (food responsiveness, satiety responsiveness) from the main file
for c in ['fr','sr']:
    df[c] = pd.to_numeric(df[c], errors='coerce') if c in df.columns else np.nan

# outcome + z-scores (z-scored on full merged sample, as in the main analysis)
df['eah_log'] = np.log(df['eah'].clip(lower=0) + LOG_OFFSET)
for v in ['screentime','age','bmi_percentile','income','mvpa','sleep','fr','sr']:
    mu, sd = df[v].mean(), df[v].std(ddof=1)
    df[f'z_{v}'] = (df[v] - mu) / sd if sd and not np.isnan(sd) else np.nan

# analytic sample (complete outcome + core covariates) -> eaters (EAH > 0)
CORE = ['z_screentime','eah_log','eah','z_age','female','z_bmi_percentile','z_income']
analytic = df.dropna(subset=CORE).copy()
eaters = analytic[analytic['eah'] > 0].copy()
print(f"analytic n={len(analytic)}   eaters n={len(eaters)}\n")

# ------------------------------------------------------------------ MODELS
# We use the children with complete values for the specific covariates in each model and report
# its sample size.
def run(label, extra):
    cols = ['z_screentime','z_age','female','z_bmi_percentile','z_income'] + extra
    missing = [c for c in cols if c not in eaters.columns]
    if missing:
        print(f"{label:<28} SKIPPED (missing columns: {', '.join(missing)})")
        return None
    d = eaters.dropna(subset=cols)
    # guard: enough rows, and no all-constant / perfectly-collinear covariate
    if len(d) < len(cols) + 2:
        print(f"{label:<28} SKIPPED (n={len(d)} too small for {len(cols)} predictors)")
        return None
    Xchk = d[cols]
    if (Xchk.nunique() <= 1).any() or np.linalg.matrix_rank(np.column_stack([np.ones(len(d)), Xchk.values])) < len(cols) + 1:
        print(f"{label:<28} SKIPPED (degenerate/collinear design; n={len(d)})")
        return None
    try:
        res, dfree = hc3_ols(d['eah_log'], d[cols])
    except np.linalg.LinAlgError:
        print(f"{label:<28} SKIPPED (singular matrix; n={len(d)})")
        return None
    b = res.loc['z_screentime']
    dfree = len(d) - (len(cols) + 1)
    tcrit = stats.t.ppf(0.975, dfree)
    lo = b['beta'] - tcrit*b['se']; hi = b['beta'] + tcrit*b['se']
    print(f"{label:<28} n={len(d):<3}  screen-time  beta={b['beta']:+.3f}  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]  p={b['p']:.3f}")
    return res

print("SCREEN TIME -> log-EAH (eaters); coefficient on standardised screen time:")
print("="*84)
print("PRIMARY MODEL")
print("-"*84)
run("Primary (age,sex,BMI percentile,inc)", [])

print("\nSENSITIVITY 1 — device-based confounders (accelerometry subsample)")
print("-"*84)
run("+ MVPA",                      ['z_mvpa'])
run("+ Sleep duration",           ['z_sleep'])
run("+ MVPA + Sleep (grouped)",   ['z_mvpa','z_sleep'])


print("="*84)
print("Interpretation: if the screen-time beta stays similar in direction/size across rows,")
print("the screen time -> EAH effect is robust to that covariate set.")
print("Grouped models (MVPA+Sleep; FR+SR) are the reported sensitivities; single-covariate")
print("rows are shown for reference. Reduced-n rows are EXPLORATORY given small subsamples.")
