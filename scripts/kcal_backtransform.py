#!/usr/bin/env python3
"""
We translate estimates on the log scale into predicted kilocalories for a one hour per day screen time contrast at each sample’s mean.

Run the full workflow with scripts/run_all.py.
"""
import os, re, numpy as np, pandas as pd
from scipy import stats

BASE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DATA   = os.path.join(BASE,'k_ab_bmi.xlsx')
SCREEN = os.path.join(BASE,'total_media_use.csv')
MVPA_F = os.path.join(BASE,'K01_MVPA_BL_051425.csv')
SLEEP_F= os.path.join(BASE,'K01_SleepSummaries_2023-01-20.csv')
OFF, SEED, NB = 30, 42, 2000

def ols_beta(y, X):
    X1=np.column_stack([np.ones(len(y)),X]); return np.linalg.pinv(X1.T@X1)@X1.T@y

raw=pd.read_excel(DATA, sheet_name='Sheet 1')
for dup in ['income','age','bmi_percentile','eah']:
    if dup in raw.columns and dup not in ('income_imputed','conf_age','bmip_bl','eah_total_kcal'):
        raw=raw.drop(columns=[dup])
df=raw.rename(columns={'conf_age':'age','bmip_bl':'bmi_percentile','income_imputed':'income','eah_total_kcal':'eah'}).copy()
df=df.loc[:,~df.columns.duplicated()]
for c in ['age','bmi_percentile','income','eah','fr','sr']: df[c]=pd.to_numeric(df[c],errors='coerce')
df['female']=(pd.to_numeric(df['gender'],errors='coerce')-1).clip(0,1)
df['rid']=pd.to_numeric(df['redcapid'],errors='coerce')

media=pd.read_csv(SCREEN)
if 'redcap_event_name' in media.columns:
    media=media[media['redcap_event_name'].astype(str).str.contains('baseline',case=False,na=False)]
media['screentime']=pd.to_numeric(media['mediatotalperweek'],errors='coerce')/60/7
media['rid']=pd.to_numeric(media['redcapid'],errors='coerce')
df=df.merge(media[['rid','screentime']].dropna().drop_duplicates('rid'),on='rid',how='left')

try:
    mv=pd.read_csv(MVPA_F,skiprows=1); mv=mv[mv['Subject Name'].astype(str).str.startswith('K01M',na=False)].copy()
    mv['rid']=mv['Subject Name'].str.extract(r'K01M0*(\d+)').astype(float)
    def _hms(s):
        m=re.match(r'(\d+)H\s+(\d+)M\s+(\d+)S',str(s)); return int(m.group(1))*60+int(m.group(2))+int(m.group(3))/60 if m else np.nan
    mv['mvpa']=mv['Average MVPA Per Day'].apply(_hms)
    df=df.merge(mv[['rid','mvpa']].dropna().drop_duplicates('rid'),on='rid',how='left')
except Exception as e: print('WARN MVPA',e); df['mvpa']=np.nan
try:
    sl=pd.read_csv(SLEEP_F); sl['rid']=sl['PID'].astype(str).str.extract(r'K01M0*(\d+)').astype(float)
    slp=sl.groupby('rid')['total.sleep.time'].mean().reset_index().rename(columns={'total.sleep.time':'sleep'})
    df=df.merge(slp,on='rid',how='left')
except Exception as e: print('WARN sleep',e); df['sleep']=np.nan

df['eah_log']=np.log(df['eah'].clip(lower=0)+OFF)
CORE = ['screentime', 'eah', 'eah_log', 'age', 'female', 'bmi_percentile',
        'income', 'ff_foodvtoy', 'cgd_foodvtoy']
analytic = df.dropna(subset=CORE).copy()
eaters = analytic[analytic['eah'] > 0].copy()
st_sd = eaters['screentime'].std(ddof=1)

# We hold other predictors at their sample means, predict at the mean and one hour above it, and
# subtract the back transformed predictions. The kcal contrast depends on the starting screen
# time.
def kcal_effect(d, extra):
    """Return per-hour kcal effect + BC bootstrap CI for screen time in model with extra covars."""
    cols = ['screentime', 'age', 'female', 'bmi_percentile', 'income'] + extra
    d = d.dropna(subset=cols + ['eah_log']).copy()
    n = len(d)
    if n < len(cols) + 2:
        return None
    # Use model-sample scaling and the same bootstrap convention as Aim 1.
    for col in cols:
        if col != 'female':
            d[col] = (d[col] - d[col].mean()) / d[col].std(ddof=1)
    screen_hours = df.loc[d.index, 'screentime'].to_numpy()
    d['_screen_hours'] = screen_hours
    d['_z_eah'] = (d['eah_log'] - d['eah_log'].mean()) / d['eah_log'].std(ddof=1)

    def per_hour(dd):
        slope = ols_beta(dd['_z_eah'].values, dd[cols].values)[1]
        delta = slope * dd['eah_log'].std(ddof=1) / dd['_screen_hours'].std(ddof=1)
        return np.exp(dd['eah_log'].mean()) * np.expm1(delta)
    pt=per_hour(d)
    rng=np.random.default_rng(SEED); boots=[]
    for _ in range(NB):
        s=d.iloc[rng.integers(0,n,n)]
        try: boots.append(per_hour(s))
        except Exception: pass
    boots=np.array(boots)
    z0=stats.norm.ppf((boots<pt).mean()); za=stats.norm.ppf(.975)
    lo=np.percentile(boots,100*stats.norm.cdf(2*z0-za)); hi=np.percentile(boots,100*stats.norm.cdf(2*z0+za))
    return pt,lo,hi,n

MODELS=[
    ("Primary (eaters)",           eaters, []),
    ("+ MVPA",                     eaters, ['mvpa']),
    ("+ Sleep",                    eaters, ['sleep']),
    ("+ MVPA + Sleep",             eaters, ['mvpa','sleep']),
    ("Full sample",               analytic, []),
    ("Full + MVPA + Sleep",         analytic, ['mvpa', 'sleep']),
]
print(f"eaters n={len(eaters)}  |  screentime SD (eaters) = {st_sd:.2f} h/day")
print("="*74)
print(f"{'Model':<24}{'kcal / +1 hr/day':>18}{'  95% CI':>22}{'  n':>5}")
print("-"*74)
for lbl,d,ex in MODELS:
    r=kcal_effect(d,ex)
    if r is None: print(f"{lbl:<24}  (skipped)"); continue
    pt,lo,hi,n=r
    print(f"{lbl:<24}{pt:>+15.1f}   [{lo:+.1f}, {hi:+.1f}]{n:>7}")
print("="*74)
print("All effects are kcal of EAH per additional hour/day of screen time,")
print("back-transformed from log(EAH+30). BC bootstrap 95% CI, 2000 resamples.")
