#!/usr/bin/env python3
"""
We assemble the mediation summary from the primary model output. We report HC3 intervals for regression paths and bias corrected bootstrap intervals for indirect effects.

Run the full workflow with scripts/run_all.py.
"""
import os, pandas as pd

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
IN   = os.path.join(_root, 'main_analysis')
RES  = os.path.join(_root, 'tables')
os.makedirs(RES, exist_ok=True)

paths = pd.read_csv(os.path.join(IN, 'table_aim2_paths.csv'))
ind   = pd.read_csv(os.path.join(IN, 'table_aim2_indirect.csv'))

# We list the full sample first and focus this table on separate models for the two candidate
# mediators.
sample_order = sorted(paths['sample'].unique(), key=lambda s: (0 if 'Full' in s else 1))
med_order = ['FF (first-fixation bias)', 'CGD (cumulative-gaze bias)']

def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}".lstrip('0')

def bci(beta, lo, hi, p):
    return f"{beta:+.2f} [{lo:+.2f}, {hi:+.2f}], p = {fmt_p(p)}"

# We use two decimals for regression paths and three for indirect effects so small estimates
# remain visible.
rows = []
for samp in sample_order:
    for med in med_order:
        pr = paths[(paths['sample']==samp)&(paths['mediator']==med)&(paths['model']=='single')]
        ir = ind[(ind['sample']==samp)&(ind['mediator']==med)&(ind['model']=='single')]
        if pr.empty or ir.empty: continue
        pr = pr.iloc[0]; ir = ir.iloc[0]
        rows.append({
            'Mediator': med.split(' (')[0],
            'Sample': samp,
            'a (ST->M) beta [95% CI] p':      bci(pr['a_beta'],  pr['a_ci_lo'],  pr['a_ci_hi'],  pr['a_p']),
            'b (M->EAH) beta [95% CI] p':     bci(pr['b_beta'],  pr['b_ci_lo'],  pr['b_ci_hi'],  pr['b_p']),
            "c' (ST->EAH|M) beta [95% CI] p": bci(pr['cp_beta'], pr['cp_ci_lo'], pr['cp_ci_hi'], pr['cp_p']),
            'Indirect a×b [95% CI]':          f"{ir['indirect_ab']:+.3f} [{ir['ci_lo']:+.3f}, {ir['ci_hi']:+.3f}]",
            'Indirect significant':           'Yes' if ir['significant']=='SIG' else 'No',
            '% mediated':                     f"{ir['prop_mediated_pct']:.0f}%",
        })

tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(RES, 'table2_mediation.csv'), index=False)

print("Total effect (c) of screen time on EAH, by sample:")
for samp in sample_order:
    pr = paths[(paths['sample']==samp)&(paths['model']=='single')].iloc[0]
    print(f"  {samp}: beta = {pr['c_beta']:+.2f} [{pr['c_ci_lo']:+.2f}, {pr['c_ci_hi']:+.2f}], p = {fmt_p(pr['c_p'])}")
print()
print(tab.to_string(index=False))
print("\nStandardized beta from multivariable HC3 regression (age, sex, BMI percentile, income).")
print("Paths a, b, c' and total effect c: HC3 95% CI and p. Indirect a×b: bias-corrected")
print("bootstrap 95% CI (2,000 resamples); interval excluding zero = significant (no p,")
print("as the bootstrap distribution of a×b is non-normal). % mediated = indirect / total.")
print("\nSaved table2_mediation.csv")
