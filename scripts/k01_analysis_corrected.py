#!/usr/bin/env python3
"""
We test whether usual screen time is associated with eating in the absence of hunger (EAH) and whether food attentional bias statistically mediates that association. We fit separate single mediator models for first fixation and cumulative gaze, and retain the parallel model as an additional analysis.

Run the full workflow with scripts/run_all.py.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

BASE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DATA   = os.path.join(BASE, 'k_ab_bmi.xlsx')
SCREEN = os.path.join(BASE, 'total_media_use.csv')
MVPA_F = os.path.join(BASE, 'K01_MVPA_BL_051425.csv')
SLEEP_F= os.path.join(BASE, 'K01_SleepSummaries_2023-01-20.csv')
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'main_analysis')
os.makedirs(OUTDIR, exist_ok=True)

OFFSET   = 30
SEED     = 42
N_BOOT   = 2000
COV      = ['z_age', 'female', 'z_bmi_percentile', 'z_income']   # baseline covariates

# We use ordinary least squares with HC3 uncertainty to account for unequal residual variance and
# leverage in this small sample.
# ------------------------------------------------------------------ helpers
def hc3_ols(y, X):
    """OLS with HC3 robust SEs. X: DataFrame (no intercept col). Returns dict."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum('ij,jk,ik->i', X, XtX_inv, X)      # leverages
    omega = (resid**2) / (1.0 - h)**2                 # HC3
    cov = XtX_inv @ (X.T * omega) @ X @ XtX_inv
    se = np.sqrt(np.diag(cov))
    dfree = len(y) - X.shape[1]
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dfree)
    return {'beta': beta, 'se': se, 't': t, 'p': p, 'resid': resid,
            'fitted': X @ beta, 'n': len(y), 'k': X.shape[1]}

def focal(res, names, name):
    """Return (beta, se, t, p) for a named predictor (names excludes intercept)."""
    i = names.index(name) + 1     # +1 for intercept
    return res['beta'][i], res['se'][i], res['t'][i], res['p'][i]

def hc3_ci(res, names, name, n):
    """HC3-based 95% CI for a coefficient: beta ± t_crit * SE (same model as its p)."""
    b, se, t, p = focal(res, names, name)
    k = res['beta'].shape[0]
    tcrit = stats.t.ppf(0.975, n - k)
    return b, b - tcrit*se, b + tcrit*se, p

def cohens_f2(y, X_full_names, Xdf, focal_name):
    """Local f² for the focal predictor."""
    full = hc3_ols(y, Xdf[X_full_names])
    reduced_names = [c for c in X_full_names if c != focal_name]
    red = hc3_ols(y, Xdf[reduced_names]) if reduced_names else None
    def r2(res):
        ss_res = np.sum(res['resid']**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res/ss_tot
    r2f = r2(full)
    r2r = r2(red) if red is not None else 0.0
    return (r2f - r2r) / (1 - r2f) if (1 - r2f) > 0 else np.nan, r2f

# We use ranks for a robustness check because screen time is skewed. We remove the modeled
# covariate contributions before measuring the association.
def partial_spearman(x, y, covdf):
    """Covariate-adjusted partial Spearman: residualize ranks then correlate."""
    C = np.column_stack([np.ones(len(x)), np.asarray(covdf, float)])
    def resid(v):
        b = np.linalg.pinv(C.T @ C) @ C.T @ v
        return v - C @ b
    rx = resid(stats.rankdata(x)); ry = resid(stats.rankdata(y))
    rho, p = stats.spearmanr(rx, ry)
    return rho, p

# We resample children and refit both paths to estimate uncertainty in their product, which need
# not have a symmetric sampling distribution.
def bc_boot_indirect(dfS, med, cov, seed=SEED, nboot=N_BOOT):
    """Bias-corrected bootstrap of a×b within ONE sample (a and b same n)."""
    rng = np.random.default_rng(seed)
    n = len(dfS)
    def ab(d):
        ra = hc3_ols(d[med].values, d[cov + ['z_screentime']])
        a = ra['beta'][ra['beta'].shape[0] - 1]  # last col = z_screentime
        rb = hc3_ols(d['z_eah_log'].values, d[cov + ['z_screentime', med]])
        b = rb['beta'][rb['beta'].shape[0] - 1]  # last col = med
        return a * b, a, b
    pt, a_pt, b_pt = ab(dfS)
    boots = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        try:
            v, _, _ = ab(dfS.iloc[idx].reset_index(drop=True))
            boots.append(v)
        except Exception:
            pass
    boots = np.array(boots)
    if len(boots) < 100:
        return pt, a_pt, b_pt, np.nan, np.nan, 'n/a'
    z0 = stats.norm.ppf(np.mean(boots < pt))
    za = stats.norm.ppf(0.975)
    lo = np.percentile(boots, 100 * stats.norm.cdf(2*z0 - za))
    hi = np.percentile(boots, 100 * stats.norm.cdf(2*z0 + za))
    sig = 'SIG' if (lo > 0 or hi < 0) else 'ns'
    return pt, a_pt, b_pt, lo, hi, sig

def sig_lbl(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '†' if p < .10 else 'ns'

# ------------------------------------------------------------------ load
print("="*74)
print("K01 CORRECTED ANALYSIS — full sample (n=42) first, then eaters (n=37)")
print("="*74)

raw = pd.read_excel(DATA, sheet_name='Sheet 1')
num = lambda c: pd.to_numeric(raw[c], errors='coerce') if c in raw.columns else pd.Series(np.nan, index=raw.index)

df = pd.DataFrame({
    'redcapid': num('redcapid'),
    'eah':   num('eah_total_kcal'),
    'ff':    num('ff_foodvtoy'),
    'cgd':   num('cgd_foodvtoy'),
    'age':   num('conf_age'),
    'gender':num('gender'),
    'bmi_percentile':  num('bmip_bl'),
    'income':num('income_imputed'),
})
df['female'] = (df['gender'] - 1).clip(0, 1)

# We convert weekly minutes to daily hours so a one hour contrast is interpretable.
# screen time (hours/day) from media file
media = pd.read_csv(SCREEN)
mcol = [c for c in media.columns if 'redcap' in c.lower() or c.lower() in ('id','pid')]
scol = [c for c in media.columns if 'perweek' in c.lower() or 'total' in c.lower()]
if mcol and scol:
    m = media[[mcol[0], scol[0]]].copy()
    m.columns = ['redcapid', 'mpw']
    m['redcapid'] = pd.to_numeric(m['redcapid'], errors='coerce')
    m['screentime'] = pd.to_numeric(m['mpw'], errors='coerce') / 60.0 / 7.0
    df = df.merge(m[['redcapid','screentime']], on='redcapid', how='left')
else:
    df['screentime'] = np.nan

# We add 30 kcal before taking logs to retain the five children with zero intake.
# EAH transform
df['eah_log'] = np.log(df['eah'].clip(lower=0) + OFFSET)

def zscore(s):
    return (s - s.mean()) / s.std(ddof=1)

for v in ['age','bmi_percentile','income','screentime','ff','cgd']:
    df['z_'+v] = zscore(df[v])

# We require observed values for both mediators and all core model variables so the primary
# models use a common sample. We repeat the models among children with positive EAH to check
# sensitivity to zero intake.
# analytic sample: complete on ST, EAH, covariates, both mediators
need = ['screentime','eah','age','female','bmi_percentile','income','ff','cgd']
full = df.dropna(subset=need).reset_index(drop=True)          # n = 42
eaters = full[full['eah'] > 0].reset_index(drop=True)          # n = 37

# We standardize the transformed outcome and continuous predictors within each sample to express
# coefficients in standard deviation units.
def prep(d):
    d = d.copy()
    # z-score outcome WITHIN this sample; re-z covariates/predictors within sample
    for v in ['age','bmi_percentile','income','screentime','ff','cgd']:
        d['z_'+v] = zscore(d[v])
    d['z_eah_log'] = zscore(d['eah_log'])
    return d

full = prep(full)
eaters = prep(eaters)
print(f"\nSample sizes:  full n={len(full)}   eaters n={len(eaters)}   non-consumers={len(full)-len(eaters)}")

SAMPLES = [('Full sample (n=%d)' % len(full), full),
           ('Consumers only (n=%d)' % len(eaters), eaters)]

# ================================================================= AIM 1
print("\n" + "="*74)
print("AIM 1 — Total effect: usual screen time -> ln(EAH+30)")
print("="*74)
aim1_rows = []
for label, S in SAMPLES:
    Xnames = COV + ['z_screentime']
    res = hc3_ols(S['z_eah_log'].values, S[Xnames])
    b, se, t, p = focal(res, Xnames, 'z_screentime')
    _, b_lo, b_hi, _ = hc3_ci(res, Xnames, 'z_screentime', len(S))   # HC3 95% CI for beta
    f2, r2 = cohens_f2(S['z_eah_log'].values, Xnames, S, 'z_screentime')
    rho, prho = partial_spearman(S['z_screentime'].values, S['z_eah_log'].values, S[COV])
    # back-transform to kcal per +1 hr/day, with BC bootstrap CI
    st_sd = S['screentime'].std(ddof=1); st_mean = S['screentime'].std(ddof=0)  # for scaling
    # predict at mean and mean+1hr (covariates at 0 = sample mean since standardized)
    mu_log = S['eah_log'].mean(); sd_log = S['eah_log'].std(ddof=1)
    delta_z = (1.0 / S['screentime'].std(ddof=1))     # +1 hr in z units
    # predicted change in ln(EAH+30) for +1hr:
    dlog = b * sd_log * delta_z
    base_kcal = np.exp(mu_log) - OFFSET
    pred_kcal = np.exp(mu_log + dlog) - OFFSET
    kcal_per_hr = pred_kcal - base_kcal
    # bootstrap kcal CI
    rng = np.random.default_rng(SEED)
    ks = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(S), len(S))
        d = S.iloc[idx].reset_index(drop=True)
        try:
            r = hc3_ols(d['z_eah_log'].values, d[Xnames])
            bb = r['beta'][-1]
            ml = d['eah_log'].mean(); sl = d['eah_log'].std(ddof=1)
            dz = 1.0 / d['screentime'].std(ddof=1)
            dl = bb * sl * dz
            ks.append((np.exp(ml+dl)-OFFSET) - (np.exp(ml)-OFFSET))
        except Exception:
            pass
    ks = np.array(ks)
    z0 = stats.norm.ppf(np.mean(ks < kcal_per_hr)); za = stats.norm.ppf(.975)
    klo = np.percentile(ks, 100*stats.norm.cdf(2*z0-za))
    khi = np.percentile(ks, 100*stats.norm.cdf(2*z0+za))
    print(f"\n  {label}")
    print(f"    ST -> ln(EAH+30):  beta={b:+.3f}  SE={se:.3f}  t={t:+.2f}  p={p:.4f} {sig_lbl(p)}")
    print(f"    partial Spearman:  rho={rho:+.3f}  p={prho:.4f} {sig_lbl(prho)}")
    print(f"    Cohen's f2={f2:.3f}   R2={r2:.3f}")
    print(f"    +1 hr/day  ->  {kcal_per_hr:+.1f} kcal EAH   BC 95% CI [{klo:+.1f}, {khi:+.1f}]")
    aim1_rows.append({'sample': label, 'beta': round(b,3), 'se': round(se,3),
                      'beta_ci_lo': round(b_lo,3), 'beta_ci_hi': round(b_hi,3),
                      't': round(t,2), 'p': round(p,4), 'sig': sig_lbl(p),
                      'partial_spearman_rho': round(rho,3), 'spearman_p': round(prho,4),
                      'cohens_f2': round(f2,3), 'R2': round(r2,3),
                      'kcal_per_hr': round(kcal_per_hr,1),
                      'kcal_ci_lo': round(klo,1), 'kcal_ci_hi': round(khi,1)})
pd.DataFrame(aim1_rows).to_csv(os.path.join(OUTDIR,'table_aim1_total_effect.csv'), index=False)

# We vary the log offset while keeping each sample and its covariates fixed to see whether the
# screen time association depends on the chosen offset.
# Supplementary Note S2. Keep the analytic samples and covariates fixed.
offset_rows = []
for label, sample in SAMPLES:
    Xnames = COV + ['z_screentime']
    for offset in range(1, 101):
        outcome = zscore(np.log(sample['eah'].clip(lower=0) + offset))
        fit = hc3_ols(outcome.values, sample[Xnames])
        beta, se, t, p = focal(fit, Xnames, 'z_screentime')
        offset_rows.append({'sample': label, 'n': len(sample), 'offset_kcal': offset,
                            'beta': beta, 'hc3_se': se, 'p': p})
pd.DataFrame(offset_rows).to_csv(os.path.join(OUTDIR, 'table_s2_log_offset_sweep.csv'), index=False)


# We fit paths a, b and c prime on the same children within each sample and use their product a ×
# b for the indirect effect.
# ================================================================= AIM 2
print("\n" + "="*74)
print("AIM 2 — Mediation by FF and CGD (a, b, c' and a×b within each sample)")
print("="*74)
path_rows = []; ind_rows = []
MEDS = [('z_ff','FF (first-fixation bias)'), ('z_cgd','CGD (cumulative-gaze bias)')]

for label, S in SAMPLES:
    print(f"\n  ── {label} ──")
    # total effect c (same model as Aim 1)
    Xc = COV + ['z_screentime']
    rc = hc3_ols(S['z_eah_log'].values, S[Xc])
    c_b, c_lo, c_hi, c_p = hc3_ci(rc, Xc, 'z_screentime', len(S))
    print(f"    total effect c: beta={c_b:+.3f} [{c_lo:+.3f},{c_hi:+.3f}] p={c_p:.4f} {sig_lbl(c_p)}")
    for med, mlbl in MEDS:
        # path a: ST -> mediator (this sample)
        Xa = COV + ['z_screentime']
        ra = hc3_ols(S[med].values, S[Xa]); a_b,a_lo,a_hi,a_p = hc3_ci(ra, Xa, 'z_screentime', len(S))
        # path b + c': mediator -> EAH | ST (this sample)
        Xb = COV + ['z_screentime', med]
        rb = hc3_ols(S['z_eah_log'].values, S[Xb])
        b_b,b_lo,b_hi,b_p = hc3_ci(rb, Xb, med, len(S))
        cp_b,cp_lo,cp_hi,cp_p = hc3_ci(rb, Xb, 'z_screentime', len(S))
        pt, a_chk, b_chk, lo, hi, sig = bc_boot_indirect(S, med, COV)
        prop = (pt / c_b * 100) if c_b not in (0, np.nan) else np.nan
        print(f"    {mlbl}")
        print(f"       a (ST->M):  beta={a_b:+.3f} [{a_lo:+.3f},{a_hi:+.3f}] p={a_p:.4f} {sig_lbl(a_p)}")
        print(f"       b (M->EAH): beta={b_b:+.3f} [{b_lo:+.3f},{b_hi:+.3f}] p={b_p:.4f} {sig_lbl(b_p)}")
        print(f"       c' (ST->EAH|M): beta={cp_b:+.3f} [{cp_lo:+.3f},{cp_hi:+.3f}] p={cp_p:.4f} {sig_lbl(cp_p)}")
        print(f"       indirect a×b={pt:+.4f}  BC bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}  ({prop:.1f}% mediated)")
        path_rows.append({'sample':label,'mediator':mlbl,'model':'single',
                          'a_beta':round(a_b,3),'a_ci_lo':round(a_lo,3),'a_ci_hi':round(a_hi,3),'a_p':round(a_p,4),
                          'b_beta':round(b_b,3),'b_ci_lo':round(b_lo,3),'b_ci_hi':round(b_hi,3),'b_p':round(b_p,4),
                          'cp_beta':round(cp_b,3),'cp_ci_lo':round(cp_lo,3),'cp_ci_hi':round(cp_hi,3),'cp_p':round(cp_p,4),
                          'c_beta':round(c_b,3),'c_ci_lo':round(c_lo,3),'c_ci_hi':round(c_hi,3),'c_p':round(c_p,4)})
        ind_rows.append({'sample':label,'mediator':mlbl,'model':'single',
                         'indirect_ab':round(pt,4),'ci_lo':round(lo,4),'ci_hi':round(hi,4),
                         'significant':sig,'prop_mediated_pct':round(prop,1)})
    # parallel model (both mediators together)
    Xp = COV + ['z_screentime','z_ff','z_cgd']
    rp = hc3_ols(S['z_eah_log'].values, S[Xp])
    print(f"    parallel (FF+CGD together):")
    for med, mlbl in MEDS:
        ub,_,_,up = focal(rp, Xp, med)
        print(f"       {mlbl} unique b={ub:+.3f} p={up:.4f} {sig_lbl(up)}")
pd.DataFrame(path_rows).to_csv(os.path.join(OUTDIR,'table_aim2_paths.csv'), index=False)
pd.DataFrame(ind_rows).to_csv(os.path.join(OUTDIR,'table_aim2_indirect.csv'), index=False)

# We add device measured physical activity and sleep together in the subset with both measures.
# We treat the result as exploratory because the sample is smaller.
# ================================================================= SENSITIVITY
print("\n" + "="*74)
print("SENSITIVITY — Aim 1 + MVPA and sleep together (device-based confounders)")
print("="*74)
sens_rows = []
import re as _re
try:
    mv = pd.read_csv(MVPA_F, skiprows=1)
    mv = mv[mv['Subject Name'].astype(str).str.startswith('K01M', na=False)].copy()
    mv['redcapid'] = mv['Subject Name'].str.extract(r'K01M0*(\d+)').astype(float)
    def _hms(s):
        m = _re.match(r'(\d+)H\s+(\d+)M\s+(\d+)S', str(s))
        return int(m.group(1))*60 + int(m.group(2)) + int(m.group(3))/60 if m else np.nan
    mv['mvpa'] = mv['Average MVPA Per Day'].apply(_hms)
    mv2 = mv[['redcapid','mvpa']].dropna().drop_duplicates('redcapid')
    sl = pd.read_csv(SLEEP_F)
    sl['redcapid'] = sl['PID'].astype(str).str.extract(r'K01M0*(\d+)').astype(float)
    sl2 = sl.groupby('redcapid', as_index=False)['total.sleep.time'].mean().rename(columns={'total.sleep.time':'sleep'})

    for label, S in SAMPLES:
        d = S.merge(mv2[['redcapid','mvpa']],on='redcapid',how='left') \
             .merge(sl2[['redcapid','sleep']],on='redcapid',how='left')
        d = d.dropna(subset=['mvpa','sleep']).reset_index(drop=True)
        if len(d) < 8:
            print(f"\n  {label}: too few with MVPA+sleep (n={len(d)}) — skipped")
            continue
        d['z_mvpa']=zscore(d['mvpa']); d['z_sleep']=zscore(d['sleep'])
        d['z_eah_log']=zscore(d['eah_log'])
        for v in ['age','bmi_percentile','income','screentime']:
            d['z_'+v]=zscore(d[v])
        Xs = COV + ['z_screentime','z_mvpa','z_sleep']
        r = hc3_ols(d['z_eah_log'].values, d[Xs]); b,se,t,p = focal(r,Xs,'z_screentime')
        print(f"\n  {label} + MVPA + sleep (n={len(d)}):  ST beta={b:+.3f} p={p:.4f} {sig_lbl(p)}")
        sens_rows.append({'sample':label,'n':len(d),'model':'+MVPA+sleep',
                          'st_beta':round(b,3),'st_se':round(se,3),'st_p':round(p,4),'sig':sig_lbl(p)})
except Exception as e:
    print("  MVPA/sleep sensitivity failed:", e)
pd.DataFrame(sens_rows).to_csv(os.path.join(OUTDIR,'table_sensitivity_mvpa_sleep.csv'), index=False)

print("\n" + "="*74)
print("DONE. Tables written to:", OUTDIR)
print("="*74)
