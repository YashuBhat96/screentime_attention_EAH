#!/usr/bin/env python3
"""
We examine how strongly individual observations affect the screen time and mediation estimates. We use these refits as sensitivity checks.

Run the full workflow with scripts/run_all.py.
"""
import os, sys, numpy as np, pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATADIR = os.path.join(_HERE, '..', 'data')
DATA  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_DATADIR, "k_ab_bmi.xlsx")
MEDIA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_DATADIR, "total_media_use.csv")
SHEET = "Sheet 1"; N_BOOT = 2000; SEED = 42

raw = pd.read_excel(DATA, sheet_name=SHEET)
num = lambda c: pd.to_numeric(raw[c], errors="coerce") if c in raw.columns else pd.Series(np.nan, index=raw.index)
df = pd.DataFrame({
    "redcapid": num("redcapid"),
    "ff": num("ff_foodvtoy"), "cgd": num("cgd_foodvtoy"),
    "age": num("conf_age"), "bmi_percentile": num("bmip_bl"), "income": num("income_imputed"),
    "female": (num("gender") - 1).clip(0, 1), "eah": num("eah_total_kcal"),
})
media = pd.read_csv(MEDIA)
if "redcap_event_name" in media.columns:
    media = media[media["redcap_event_name"].astype(str).str.contains("baseline", case=False, na=False)]
media["screentime"] = pd.to_numeric(media["mediatotalperweek"], errors="coerce") / 60 / 7
df["redcapid"] = pd.to_numeric(df["redcapid"], errors="coerce")
df = df.merge(media[["redcapid","screentime"]].assign(redcapid=lambda d: pd.to_numeric(d["redcapid"], errors="coerce")),
              on="redcapid", how="left")
df["eah_log"] = np.log(df["eah"].clip(lower=0) + 30)
for v in ["ff","cgd","age","bmi_percentile","income","screentime","eah_log"]:
    s = df[v]; df["z_"+v] = (s - s.mean())/s.std(ddof=1)
COV = ["z_age","female","z_bmi_percentile","z_income"]
full = df.dropna(subset=["z_screentime","eah_log","eah","z_ff","z_cgd"]+COV).reset_index(drop=True)
cons = full[full["eah"] > 0].reset_index(drop=True)
for d in (full, cons):
    d["z_eah_log"] = (d["eah_log"]-d["eah_log"].mean())/d["eah_log"].std(ddof=1)

def hc3(y, Xdf):
    X = np.column_stack([np.ones(len(y)), Xdf.values]); names=["int"]+list(Xdf.columns)
    XtXi = np.linalg.pinv(X.T@X); beta = XtXi@X.T@y; resid = y-X@beta
    h = np.einsum("ij,jk,ik->i",X,XtXi,X); e = resid/(1-h)
    V = XtXi @ ((X*e[:,None]).T@(X*e[:,None])) @ XtXi; se=np.sqrt(np.diag(V))
    t = beta/se; p = 2*stats.t.sf(np.abs(t), len(y)-X.shape[1])
    return names, beta, se, p
def ols_b(y, Xdf):
    X = np.column_stack([np.ones(len(y)), Xdf.values]); return np.linalg.lstsq(X, y, rcond=None)[0]
# We use Cook’s distance to flag observations whose exclusion may noticeably change the
# regression fit.
def cooks(y, Xdf):
    X = np.column_stack([np.ones(len(y)), Xdf.values]); n,pp=X.shape
    XtXi=np.linalg.pinv(X.T@X); beta=XtXi@X.T@y; resid=y-X@beta
    mse=np.sum(resid**2)/(n-pp); h=np.diag(X@XtXi@X.T)
    return (resid**2*h)/(pp*mse*(1-h)**2)
def foc(names,beta,se,p,col): i=names.index(col); return beta[i],se[i],p[i]
def total(d):
    nm,b,se,p = hc3(d["z_eah_log"].values, d[COV+["z_screentime"]]); return foc(nm,b,se,p,"z_screentime")
def indirect(d, med):
    a = ols_b(d[med].values, d[COV+["z_screentime"]])[1+(COV+["z_screentime"]).index("z_screentime")]
    nmb=COV+["z_screentime",med]; b = ols_b(d["z_eah_log"].values, d[nmb])[1+nmb.index(med)]
    ind=a*b; rng=np.random.default_rng(SEED); n=len(d); boots=[]
    for _ in range(N_BOOT):
        s=d.iloc[rng.integers(0,n,n)].reset_index(drop=True)
        try:
            zz=(s["eah_log"]-s["eah_log"].mean())/s["eah_log"].std(ddof=1)
            aa=ols_b(s[med].values,s[COV+["z_screentime"]])[1+(COV+["z_screentime"]).index("z_screentime")]
            bb=ols_b(zz.values,s[nmb])[1+nmb.index(med)]; boots.append(aa*bb)
        except: pass
    boots=np.array(boots)
    z0=stats.norm.ppf((boots<ind).mean()); za=stats.norm.ppf(.975)
    lo=np.percentile(boots,100*stats.norm.cdf(2*z0-za)); hi=np.percentile(boots,100*stats.norm.cdf(2*z0+za))
    return ind,lo,hi,("SIG" if (lo>0 or hi<0) else "ns")
# We identify influential rows separately by model because leverage and residual size depend on
# the predictors.
def infl_idx(d, predictors):
    c = cooks(d["z_eah_log"].values, d[predictors]); return np.where(c > 4/len(d))[0], c

MODELS = {
    "Aim1 total   [cov+ST]":        COV+["z_screentime"],
    "Aim2 FF      [cov+ST+FF]":     COV+["z_screentime","z_ff"],
    "Aim2 CGD     [cov+ST+CGD]":    COV+["z_screentime","z_cgd"],
    "Aim2 FF+CGD  [cov+ST+FF+CGD]": COV+["z_screentime","z_ff","z_cgd"],
}
nonc = full.loc[full["eah"]==0, "redcapid"].astype(int).tolist()

print("="*72); print("PART A — WHO drives the results"); print("="*72)
print("Non-consumers (EAH=0), the 5 kids distinguishing n=42 from n=37:")
print(f"   Count: {len(nonc)}")
for name, d in [("full sample n=42", full), ("consumers n=37", cons)]:
    print(f"\nInfluential (Cook's D > 4/n = {4/len(d):.3f}) — {name}:")
    for mlabel, preds in MODELS.items():
        idx,_ = infl_idx(d, preds)
        # We report counts rather than study IDs in the run log.
        n_nonconsumers = (d.loc[idx, "eah"] == 0).sum()
        print(f"   {mlabel:32s}: n={len(idx)} (nonconsumers: {n_nonconsumers})")

print("\n" + "="*72); print("PART B — FULL vs REDUCED (influential removed)"); print("="*72)
for name, d in [("full sample n=42", full), ("consumers n=37", cons)]:
    print(f"\n### SAMPLE: {name} (n={len(d)}) ###")
    idx,_ = infl_idx(d, COV+["z_screentime"]); dr = d.drop(index=idx).reset_index(drop=True)
    b,se,p = total(d); b2,se2,p2 = total(dr)
    print(f"  Aim1 ST→EAH:  FULL β={b:+.3f} p={p:.4f}  |  REDUCED(n={len(dr)},-{len(idx)}) β={b2:+.3f} p={p2:.4f}")
    for med,lbl in [("z_ff","FF"),("z_cgd","CGD")]:
        idx,_ = infl_idx(d, COV+["z_screentime",med]); dr = d.drop(index=idx).reset_index(drop=True)
        ind,lo,hi,sg = indirect(d, med); ind2,lo2,hi2,sg2 = indirect(dr, med)
        print(f"  Aim2 {lbl} a×b: FULL {ind:+.3f}[{lo:+.3f},{hi:+.3f}]{sg}  |  "
              f"REDUCED(n={len(dr)},-{len(idx)}) {ind2:+.3f}[{lo2:+.3f},{hi2:+.3f}]{sg2}")
print("\nRobust if REDUCED keeps direction & rough magnitude of FULL (p may weaken with n).")
