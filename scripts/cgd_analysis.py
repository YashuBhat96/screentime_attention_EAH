#!/usr/bin/env python3
"""
We examine cumulative gaze separately to see whether sustained attention to food relative to toys could account for part of the screen time association. We keep these exploratory outputs separate from the manuscript figures.

Run the full workflow with scripts/run_all.py.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_style import use_manuscript_font

use_manuscript_font()

BASE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'cgd')
os.makedirs(OUTDIR, exist_ok=True)

DATA   = os.path.join(BASE, 'k_ab_bmi.xlsx')
SCREEN = os.path.join(BASE, 'total_media_use.csv')
OFFSET, SEED, N_BOOT = 30, 42, 2000
COV = ['z_age', 'female', 'z_bmi_percentile', 'z_income']

# ------------------------------------------------------------------ helpers
def hc3_ols(y, X):
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum('ij,jk,ik->i', X, XtX_inv, X)
    omega = (resid**2) / (1.0 - h)**2
    cov = XtX_inv @ (X.T * omega) @ X @ XtX_inv
    se = np.sqrt(np.diag(cov))
    dfree = len(y) - X.shape[1]
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dfree)
    return {'beta': beta, 'se': se, 't': t, 'p': p, 'resid': resid,
            'fitted': X @ beta, 'h': h, 'n': len(y), 'k': X.shape[1]}

def focal(res, names, name):
    i = names.index(name) + 1
    return res['beta'][i], res['se'][i], res['t'][i], res['p'][i]

def hc3_ci(res, names, name, n):
    b, se, t, p = focal(res, names, name)
    k = res['beta'].shape[0]
    tcrit = stats.t.ppf(0.975, n - k)
    return b, b - tcrit*se, b + tcrit*se, p

def sig(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '†' if p < .10 else 'ns'

def zscore(s):
    return (s - s.mean()) / s.std(ddof=1)

def cooks_d(res):
    k = res['k']; h = res['h']; resid = res['resid']
    s2 = np.sum(resid**2) / (res['n'] - k)
    return (resid**2 / (k * s2)) * (h / (1 - h)**2)

# We bootstrap the product of the two mediation paths because its sampling distribution can be
# asymmetric.
def bc_boot_indirect(dfS, med, seed=SEED, nboot=N_BOOT):
    rng = np.random.default_rng(seed); n = len(dfS)
    def ab(d):
        ra = hc3_ols(d[med].values, d[COV + ['z_screentime']]); a = ra['beta'][-1]
        rb = hc3_ols(d['z_eah_log'].values, d[COV + ['z_screentime', med]]); b = rb['beta'][-1]
        return a * b
    pt = ab(dfS); boots = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        try: boots.append(ab(dfS.iloc[idx].reset_index(drop=True)))
        except Exception: pass
    boots = np.array(boots)
    if len(boots) < 100: return pt, np.nan, np.nan, 'n/a'
    z0 = stats.norm.ppf(np.mean(boots < pt)); za = stats.norm.ppf(.975)
    lo = np.percentile(boots, 100*stats.norm.cdf(2*z0-za))
    hi = np.percentile(boots, 100*stats.norm.cdf(2*z0+za))
    return pt, lo, hi, ('SIG' if (lo > 0 or hi < 0) else 'ns')

# ------------------------------------------------------------------ load
print("="*72)
print("CGD (cumulative-gaze bias, food vs toy) — parallel mediator analysis")
print("="*72)
raw = pd.read_excel(DATA, sheet_name='Sheet 1')
num = lambda c: pd.to_numeric(raw[c], errors='coerce') if c in raw.columns else pd.Series(np.nan, index=raw.index)
df = pd.DataFrame({
    'redcapid': num('redcapid'), 'eah': num('eah_total_kcal'),
    'cgd': num('cgd_foodvtoy'), 'cgd_f': num('cgd_f'), 'cgd_t': num('cgd_t'),
    'age': num('conf_age'), 'gender': num('gender'),
    'bmi_percentile': num('bmip_bl'), 'income': num('income_imputed'),
})
df['female'] = (df['gender'] - 1).clip(0, 1)
media = pd.read_csv(SCREEN)
mid = [c for c in media.columns if 'redcap' in c.lower() or c.lower() in ('id','pid')][0]
msc = [c for c in media.columns if 'perweek' in c.lower() or 'total' in c.lower()][0]
m = media[[mid, msc]].copy(); m.columns = ['redcapid', 'mpw']
m['redcapid'] = pd.to_numeric(m['redcapid'], errors='coerce')
m['screentime'] = pd.to_numeric(m['mpw'], errors='coerce') / 60 / 7
df = df.merge(m[['redcapid', 'screentime']], on='redcapid', how='left')
df['eah_log'] = np.log(df['eah'].clip(lower=0) + OFFSET)

need = ['screentime', 'eah', 'age', 'female', 'bmi_percentile', 'income', 'cgd']
full = df.dropna(subset=need).reset_index(drop=True)
eaters = full[full['eah'] > 0].reset_index(drop=True)

def prep(d):
    d = d.copy()
    for v in ['age', 'bmi_percentile', 'income', 'screentime', 'cgd']:
        d['z_'+v] = zscore(d[v])
    d['z_eah_log'] = zscore(d['eah_log'])
    return d
full, eaters = prep(full), prep(eaters)
SAMPLES = [('Full sample (n=%d)' % len(full), full),
           ('Consumers only (n=%d)' % len(eaters), eaters)]
print(f"\nSamples: full n={len(full)}, consumers n={len(eaters)}")

# We report paths and indirect effects separately for the full sample and consumers.
# ------------------------------------------------------------------ MEDIATION
print("\n" + "="*72); print("CGD MEDIATION — a, b, c', indirect (both samples)"); print("="*72)
med_rows, ind_rows = [], []
for label, S in SAMPLES:
    Xc = COV + ['z_screentime']
    rc = hc3_ols(S['z_eah_log'].values, S[Xc]); c_b, c_lo, c_hi, c_p = hc3_ci(rc, Xc, 'z_screentime', len(S))
    ra = hc3_ols(S['z_cgd'].values, S[Xc]); a_b, a_lo, a_hi, a_p = hc3_ci(ra, Xc, 'z_screentime', len(S))
    Xb = COV + ['z_screentime', 'z_cgd']; rb = hc3_ols(S['z_eah_log'].values, S[Xb])
    b_b, b_lo, b_hi, b_p = hc3_ci(rb, Xb, 'z_cgd', len(S)); cp_b, cp_lo, cp_hi, cp_p = hc3_ci(rb, Xb, 'z_screentime', len(S))
    pt, lo, hi, s = bc_boot_indirect(S, 'z_cgd')
    prop = pt / c_b * 100 if c_b else np.nan
    print(f"\n  {label}")
    print(f"    total c:   beta={c_b:+.3f} [{c_lo:+.3f},{c_hi:+.3f}] p={c_p:.4f} {sig(c_p)}")
    print(f"    a (ST→CGD): beta={a_b:+.3f} [{a_lo:+.3f},{a_hi:+.3f}] p={a_p:.4f} {sig(a_p)}")
    print(f"    b (CGD→EAH):beta={b_b:+.3f} [{b_lo:+.3f},{b_hi:+.3f}] p={b_p:.4f} {sig(b_p)}")
    print(f"    c' (ST→EAH|CGD): beta={cp_b:+.3f} [{cp_lo:+.3f},{cp_hi:+.3f}] p={cp_p:.4f} {sig(cp_p)}")
    print(f"    indirect a×b={pt:+.4f} BC bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}] {s} ({prop:.1f}% mediated)")
    med_rows.append({'sample': label,
                     'a_beta': round(a_b,3),'a_ci_lo':round(a_lo,3),'a_ci_hi':round(a_hi,3),'a_p': round(a_p,4),
                     'b_beta': round(b_b,3),'b_ci_lo':round(b_lo,3),'b_ci_hi':round(b_hi,3),'b_p': round(b_p,4),
                     'cp_beta': round(cp_b,3),'cp_ci_lo':round(cp_lo,3),'cp_ci_hi':round(cp_hi,3),'cp_p': round(cp_p,4),
                     'c_beta': round(c_b,3),'c_ci_lo':round(c_lo,3),'c_ci_hi':round(c_hi,3),'c_p': round(c_p,4)})
    ind_rows.append({'sample': label, 'indirect_ab': round(pt,4), 'ci_lo': round(lo,4),
                     'ci_hi': round(hi,4), 'significant': s, 'prop_mediated_pct': round(prop,1)})
pd.DataFrame(med_rows).to_csv(os.path.join(OUTDIR, 'table_cgd_paths.csv'), index=False)
pd.DataFrame(ind_rows).to_csv(os.path.join(OUTDIR, 'table_cgd_indirect.csv'), index=False)

# We refit after excluding observations flagged by Cook’s distance to see how strongly individual
# cases affect the results.
# ------------------------------------------------------------------ INFLUENTIAL-CASE SENSITIVITY
print("\n" + "="*72); print("CGD influential-case sensitivity (Cook's D > 4/n removed)"); print("="*72)
sens_rows = []
for label, S in SAMPLES:
    Xb = COV + ['z_screentime', 'z_cgd']
    rb = hc3_ols(S['z_eah_log'].values, S[Xb])
    d = cooks_d(rb); thr = 4.0 / len(S); keep = d <= thr
    Sr = S[keep].reset_index(drop=True); n_inf = int((~keep).sum())
    pt, lo, hi, s = bc_boot_indirect(Sr, 'z_cgd') if len(Sr) > 10 else (np.nan,)*4
    print(f"  {label}: {n_inf} influential removed → n={len(Sr)}; "
          f"indirect={pt:+.4f} CI [{lo:+.4f},{hi:+.4f}] {s}" if len(Sr) > 10 else
          f"  {label}: too few after removal")
    sens_rows.append({'sample': label, 'n_influential_removed': n_inf, 'n_reduced': len(Sr),
                      'indirect_ab': round(pt,4) if pt==pt else np.nan,
                      'ci_lo': round(lo,4) if lo==lo else np.nan,
                      'ci_hi': round(hi,4) if hi==hi else np.nan, 'significant': s})
pd.DataFrame(sens_rows).to_csv(os.path.join(OUTDIR, 'table_cgd_influential.csv'), index=False)

# We generate exploratory plots for this script; the manuscript versions are generated by
# make_figures.py.
# ------------------------------------------------------------------ FIGURES
# Fig 1: CGD food vs toy bars by group (Overall / Eaters / Non-eaters)
grp = [('Overall\n(n=%d)'%len(full), full),
       ('Eaters\n(n=%d)'%len(eaters), eaters),
       ('Non-eaters\n(n=%d)'%(len(full)-len(eaters)), full[full['eah']==0])]
fm = [g[1]['cgd_f'].mean() for g in grp]; fe = [g[1]['cgd_f'].sem() for g in grp]
tm = [g[1]['cgd_t'].mean() for g in grp]; te = [g[1]['cgd_t'].sem() for g in grp]
x = np.arange(len(grp)); w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)
ax.bar(x-w/2, fm, w, yerr=fe, capsize=4, label='Food', color='#D55E00', edgecolor='white')
ax.bar(x+w/2, tm, w, yerr=te, capsize=4, label='Toy',  color='#0072B2', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in grp])
ax.set_ylabel('Cumulative gaze duration (ms)'); ax.set_title('Cumulative gaze: food vs toy distractors')
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'fig_bars_cgd.png')); plt.close(fig)

# Fig 2: CGD mediation 3-panel (a: ST→CGD, b: CGD→EAH resid, c': ST→EAH|CGD) on full sample
S = full
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=300)
# a
axes[0].scatter(S['screentime'], S['cgd'], color='#009E73')
axes[0].set_xlabel('Screen time (hr/day)'); axes[0].set_ylabel('CGD bias (ms)'); axes[0].set_title("Path a: ST → CGD")
# b (raw)
axes[1].scatter(S['cgd'], np.exp(S['eah_log'])-OFFSET, color='#D55E00')
axes[1].set_xlabel('CGD bias (ms)'); axes[1].set_ylabel('EAH (kcal)'); axes[1].set_title("Path b: CGD → EAH")
# c'
axes[2].scatter(S['screentime'], np.exp(S['eah_log'])-OFFSET, color='#0072B2')
axes[2].set_xlabel('Screen time (hr/day)'); axes[2].set_ylabel('EAH (kcal)'); axes[2].set_title("Path c′: ST → EAH | CGD")
fig.suptitle('CGD mediation (full sample, n=%d)' % len(full))
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'fig_med_cgd.png')); plt.close(fig)

print("\n" + "="*72)
print("DONE. CGD tables + figures written to:", OUTDIR)
print("  table_cgd_paths.csv, table_cgd_indirect.csv, table_cgd_influential.csv")
print("  fig_bars_cgd.png, fig_med_cgd.png")
print("="*72)
