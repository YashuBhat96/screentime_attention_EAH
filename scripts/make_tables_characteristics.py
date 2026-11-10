#!/usr/bin/env python3
"""
We describe the full analytic sample and the subset with positive EAH. We include denominators for accelerometer measures because availability differs by measure.

Run the full workflow with scripts/run_all.py.
"""
import os, numpy as np, pandas as pd
from scipy import stats

BASE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DATA   = os.path.join(BASE, 'k_ab_bmi.xlsx')
SCREEN = os.path.join(BASE, 'total_media_use.csv')
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'tables')
os.makedirs(OUTDIR, exist_ok=True)

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
    'education':num('education'),
})
df['female'] = (df['gender'] - 1).clip(0, 1)

media = pd.read_csv(SCREEN)
mid = [c for c in media.columns if 'redcap' in c.lower() or c.lower() in ('id','pid')][0]
msc = [c for c in media.columns if 'perweek' in c.lower() or 'total' in c.lower()][0]
m = media[[mid,msc]].copy(); m.columns=['redcapid','mpw']
m['redcapid']=pd.to_numeric(m['redcapid'],errors='coerce')
m['screentime']=pd.to_numeric(m['mpw'],errors='coerce')/60/7
df = df.merge(m[['redcapid','screentime']],on='redcapid',how='left')

need = ['screentime','eah','age','female','bmi_percentile','income','ff','cgd']
full   = df.dropna(subset=need).reset_index(drop=True)
eaters = full[full['eah']>0].reset_index(drop=True)
noncon = full[full['eah']==0].reset_index(drop=True)

def msd(s):
    s=s.dropna()
    return f"{s.mean():.2f} ± {s.std(ddof=1):.2f}" if len(s) else "—"
def npct(s):
    s=s.dropna()
    return f"{int(s.sum())} ({100*s.mean():.0f}%)" if len(s) else "—"

def row(label, kind, col):
    f = npct(full[col]) if kind=='pct' else msd(full[col])
    e = npct(eaters[col]) if kind=='pct' else msd(eaters[col])
    if kind == 'bmi_percentile':
        f, e = [f'{s.mean() * 100:.1f} ± {s.std(ddof=1) * 100:.1f}'
                for s in (full[col].dropna(), eaters[col].dropna())]
    return [label, f, e]

SHORT = [
    row('Age, years','msd','age'),
    row('Female, n (%)','pct','female'),
    row('BMI percentile','bmi_percentile','bmi_percentile'),
    row('Household income (index)','msd','income'),
    row('Usual screen time, hr/day','msd','screentime'),
    row('EAH, kcal','msd','eah'),
]
# We group categories as shown in the manuscript and calculate percentages within each displayed
# sample.
# Match the grouped category coding used in the manuscript table.
def category(label, column, codes):
    def cell(sample):
        values = sample[column].dropna()
        n = int(values.isin(codes).sum())
        return f"{n} ({100*n/len(values):.0f}%)" if len(values) else "—"
    return [label, cell(full), cell(eaters)]

def summary(label, column, digits=1, range_digits=None, include_n=False):
    def cell(sample):
        values = sample[column].dropna()
        if not len(values):
            return "—"
        value = f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"
        if range_digits is not None:
            value += f" ({values.min():.{range_digits}f}–{values.max():.{range_digits}f})"
        if include_n:
            value += f" (n = {len(values)})"
        return value
    return [label, cell(full), cell(eaters)]

# We use baseline accelerometry for MVPA and nighttime sleep, and summarize each available
# measure separately.
mvpa = pd.read_csv(os.path.join(BASE, 'K01_MVPA_BL_051425.csv'), skiprows=1)
mvpa['redcapid'] = pd.to_numeric(mvpa['Subject Name'].astype(str).str.extract(r'^K01M0*(\d+)')[0])
duration = mvpa['Average MVPA Per Day'].astype(str).str.extract(r'^(\d+)H\s+(\d+)M\s+(\d+)S').astype(float)
mvpa['mvpa'] = duration[0]*60 + duration[1] + duration[2]/60
mvpa = mvpa.dropna(subset=['redcapid','mvpa']).drop_duplicates('redcapid').set_index('redcapid')

sleep = pd.read_csv(os.path.join(BASE, 'K01_SleepSummaries_2023-01-20.csv'))
sleep['redcapid'] = pd.to_numeric(sleep['PID'].astype(str).str.extract(r'K01M0*(\d+)')[0])
sleep_hours = sleep.groupby('redcapid')['total.sleep.time'].mean()/60

for sample in (full, eaters):
    sample['mvpa'] = sample['redcapid'].map(mvpa['mvpa'])
    sample['sleep'] = sample['redcapid'].map(sleep_hours)
    sample['weight_status'] = pd.cut(sample['bmi_percentile'],
        [-np.inf,.05,.85,.95,np.inf], labels=[1,2,3,4], right=False)

DETAIL = [
    summary('Age, years','age',2,0),
    row('Female, n (%)','pct','female'),
    row('BMI percentile','bmi_percentile','bmi_percentile'),
    category('Healthy weight, n (%)','weight_status',[2]),
    category('Overweight, n (%)','weight_status',[3]),
    category('Obesity, n (%)','weight_status',[4]),
    category('Underweight, n (%)','weight_status',[1]),
    ['Household income, n (%)','',''],
    category('< $65,000','income',[1,2]),
    category('$65,000–145,000','income',[3]),
    category('> $145,000','income',[4,5]),
    ['Highest parent education, n (%)','',''],
    category('High School graduate or GED','education',[3]),
    category('Associate/vocational','education',[4]),
    category('Bachelor’s degree','education',[5]),
    category('Graduate/professional','education',[6]),
    summary('Screen time, hr/day','screentime',2,1),
    summary('EAH, kcal','eah',1,1),
    summary('First-fixation bias, ms','ff'),
    summary('Cumulative-gaze bias, ms','cgd'),
    summary('MVPA, min/day','mvpa',include_n=True),
    summary('Sleep, h/night','sleep',2,include_n=True),
]

hdr = ['Characteristic', f'Full sample (n={len(full)})', f'Consumers (n={len(eaters)})']

def write_csv(rows, path):
    pd.DataFrame(rows, columns=hdr).to_csv(path, index=False)

write_csv(SHORT,  os.path.join(OUTDIR,'table1_short.csv'))
write_csv(DETAIL, os.path.join(OUTDIR,'table1_detailed.csv'))

# We treat the comparisons with the five nonconsumers as descriptive given the small subgroup.
# footnote: which vars differed between consumers and non-consumers (exploratory)
print("Footnote check — consumers vs non-consumers (exploratory, non-consumer n=%d):" % len(noncon))
for label,col in [('Age','age'),('BMI percentile','bmi_percentile'),('Income','income'),
                  ('Screen time','screentime'),('First-fixation bias','ff'),
                  ('Cumulative-gaze bias','cgd')]:
    a=eaters[col].dropna(); b=noncon[col].dropna()
    if col == 'bmi_percentile':
        a, b = a * 100, b * 100
    if len(b)>=2 and len(a)>=2:
        try:
            _,p = stats.mannwhitneyu(a,b,alternative='two-sided')
            flag = '  <-- differs (p<.05)' if p<.05 else ''
            print(f"   {label:24s} consumers {a.mean():.2f} vs non-consumers {b.mean():.2f}  p={p:.3f}{flag}")
        except Exception as e:
            print(f"   {label}: {e}")

print("\nSHORT table:"); print(pd.DataFrame(SHORT,columns=hdr).to_string(index=False))
print("\nDETAILED table:"); print(pd.DataFrame(DETAIL,columns=hdr).to_string(index=False))
print("\nSaved table1_short.csv and table1_detailed.csv")
