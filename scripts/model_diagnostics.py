#!/usr/bin/env python3
"""
We inspect fitted models for collinearity, heteroskedasticity and other patterns that may affect interpretation. We retain HC3 standard errors and bootstrap intervals for the primary inference.

Run the full workflow with scripts/run_all.py.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

DATA  = sys.argv[1] if len(sys.argv) > 1 else "k_ab_bmi.xlsx"
MEDIA = sys.argv[2] if len(sys.argv) > 2 else "total_media_use.csv"
SHEET = "Sheet 1"

# ---------- load & derive (mirrors k01_SSIB_analysis.py) ----------
raw = pd.read_excel(DATA, sheet_name=SHEET)
num = lambda c: pd.to_numeric(raw[c], errors="coerce") if c in raw.columns else pd.Series(np.nan, index=raw.index)
df = pd.DataFrame({
    "redcapid": num("redcapid"),
    "ff":  num("ff_foodvtoy"), "cgd": num("cgd_foodvtoy"),
    "age": num("conf_age"), "bmi_percentile": num("bmip_bl"), "income": num("income_imputed"),
    "female": (num("gender") - 1).clip(0, 1),
    "eah": num("eah_total_kcal"),
})
media = pd.read_csv(MEDIA)
if "redcap_event_name" in media.columns:
    media = media[media["redcap_event_name"].astype(str).str.contains("baseline", case=False, na=False)]
media["screentime"] = pd.to_numeric(media["mediatotalperweek"], errors="coerce") / 60 / 7
df["redcapid"] = pd.to_numeric(df["redcapid"], errors="coerce")
df = df.merge(media[["redcapid", "screentime"]].assign(redcapid=lambda d: pd.to_numeric(d["redcapid"], errors="coerce")),
              on="redcapid", how="left")
df["eah_log"] = np.log(df["eah"].clip(lower=0) + 30)

# z-score continuous on full available sample (as in pipeline)
for v in ["ff", "cgd", "age", "bmi_percentile", "income", "screentime", "eah_log"]:
    s = df[v]; df["z_" + v] = (s - s.mean()) / s.std(ddof=1)

COV = ["z_age", "female", "z_bmi_percentile", "z_income"]

full = df.dropna(subset=["z_screentime", "eah_log", "eah", "z_ff", "z_cgd"] + COV).reset_index(drop=True)  # n=42
cons = full[full["eah"] > 0].reset_index(drop=True)                                                        # n=37
for d in (full, cons):
    d["z_eah_log"] = (d["eah_log"] - d["eah_log"].mean()) / d["eah_log"].std(ddof=1)

# ---------- helpers ----------
def ols(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    XtXi = np.linalg.pinv(X1.T @ X1)
    beta = XtXi @ X1.T @ y
    fitted = X1 @ beta
    resid = y - fitted
    h = np.einsum("ij,jk,ik->i", X1, XtXi, X1)
    return beta, fitted, resid, h, X1

def breusch_pagan(resid, X1):
    # regress squared resid on predictors; LM = n * R^2 ~ chi2(k)
    u2 = resid**2 / (resid**2).mean()
    b = np.linalg.pinv(X1.T @ X1) @ X1.T @ u2
    ss_tot = np.sum((u2 - u2.mean())**2); ss_res = np.sum((u2 - X1 @ b)**2)
    r2 = 1 - ss_res/ss_tot
    lm = len(resid) * r2
    dfree = X1.shape[1] - 1
    return lm, stats.chi2.sf(lm, dfree), dfree

# We quantify collinearity with variance inflation factors to assess how closely predictors
# overlap.
def vif(Xdf):
    out = {}
    cols = list(Xdf.columns)
    for i, c in enumerate(cols):
        y = Xdf[c].values
        others = Xdf[[x for x in cols if x != c]].values
        _, fit, res, _, _ = ols(y, others)
        ss_tot = np.sum((y - y.mean())**2); r2 = 1 - np.sum(res**2)/ss_tot
        out[c] = np.inf if r2 >= 1 else 1/(1 - r2)
    return out

def flag(ok): return "PASS " if ok else "CHECK"

# We report diagnostics separately for each model because the predictor sets and residuals
# differ.
def diagnose(d, label, predictors):
    print("\n" + "="*70)
    print(f"MODEL: {label}   (n={len(d)})")
    print(f"  outcome = Z[log(EAH+30)];  predictors = {', '.join(predictors)}")
    print("="*70)
    y = d["z_eah_log"].values
    X = d[predictors].values
    beta, fitted, resid, h, X1 = ols(y, X)

    # 1. residual normality — assessed visually via Q-Q plot (see make_figures.py);
    #    no formal test (unreliable at this n; inference uses HC3 + bootstrap).

    # 2. heteroskedasticity
    lm, p_bp, dfree = breusch_pagan(resid, X1)
    print(f"  2. Heteroskedasticity (Breusch-Pagan): LM={lm:.2f}, df={dfree}, p={p_bp:.3f}   [{flag(p_bp>=.05)}]")
    print(f"       (HC3 SEs used regardless; test is informational)")

    # 3. linearity / residual-fitted independence
    r_rf, p_rf = stats.pearsonr(fitted, resid)
    print(f"  3. Residual vs fitted r={r_rf:+.3f} (should be ~0)   [{flag(abs(r_rf)<0.1)}]")

    # 4. influence
    p_params = X1.shape[1]
    mse = np.sum(resid**2)/(len(y)-p_params)
    cooks = (resid**2 * h) / (p_params * mse * (1-h)**2)
    thr = 4/len(y)
    n_infl = int((cooks > thr).sum())
    print(f"  4. Influence: {n_infl} obs with Cook's D > 4/n ({thr:.3f}); max D={cooks.max():.3f}, max leverage={h.max():.3f}   [{flag(n_infl<=2)}]")

    # 5. multicollinearity
    v = vif(d[predictors])
    vmax = max(v.values())
    print(f"  5. Multicollinearity (VIF): max={vmax:.2f}   [{flag(vmax<5)}]")
    for k, val in v.items(): print(f"        VIF[{k}] = {val:.2f}")

# We summarize how the log offset changes the distribution. The outcome Q–Q plots are produced by
# make_figures.py.
def outcome_transform_check():
    print("\n" + "="*70)
    print("OUTCOME TRANSFORM CHECK  (raw EAH vs log(EAH+30)) — skewness")
    print("="*70)
    for lbl, ser in [("raw EAH (consumers)", cons["eah"]),
                     ("log(EAH+30) (consumers)", cons["eah_log"]),
                     ("raw EAH (full)", full["eah"]),
                     ("log(EAH+30) (full)", full["eah_log"])]:
        sk = stats.skew(ser.dropna())
        print(f"  {lbl:28s}: skewness={sk:+.2f}  ({'right-skewed' if sk>0.5 else 'approx. symmetric' if abs(sk)<=0.5 else 'left-skewed'})")
    print("  (Normality assessed visually with residual Q-Q plots; no formal test used.)")

# ---------- run ----------
print("Samples:  full n=%d  |  consumers n=%d  |  non-consumers=%d"
      % (len(full), len(cons), int((full['eah']==0).sum())))
outcome_transform_check()

# Aim 1 total effect
diagnose(cons, "Aim 1 total  (consumers)", COV + ["z_screentime"])
diagnose(full, "Aim 1 total  (full sample)", COV + ["z_screentime"])

# Aim 2 path b / c' single-mediator (FF, then CGD)
diagnose(cons, "Aim 2  FF single-mediator (consumers)", COV + ["z_screentime", "z_ff"])
diagnose(cons, "Aim 2  CGD single-mediator (consumers)", COV + ["z_screentime", "z_cgd"])

# Aim 2 parallel FF + CGD  (the collinearity-sensitive one)
diagnose(cons, "Aim 2  FF+CGD parallel (consumers)", COV + ["z_screentime", "z_ff", "z_cgd"])
diagnose(full, "Aim 2  FF+CGD parallel (full sample)", COV + ["z_screentime", "z_ff", "z_cgd"])

print("\nDONE. PASS = assumption reasonably met; CHECK = inspect / note as limitation.")
