#!/usr/bin/env python3
"""
We generate the analysis figures from the released input files and use saved model output for the mediation annotations and path diagrams. We show observed values alongside adjusted fits and retain the manuscript units and displayed scales.

Run the full workflow with scripts/run_all.py.
"""
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_style import use_manuscript_font

use_manuscript_font()

_HERE   = os.path.dirname(os.path.abspath(__file__))
BASE    = os.path.join(_HERE, '..', 'data')
FIGDIR  = os.path.join(_HERE, '..', 'results', 'figures')
os.makedirs(FIGDIR, exist_ok=True)
DATA    = os.path.join(BASE, 'k_ab_bmi.xlsx')
SCREEN  = os.path.join(BASE, 'total_media_use.csv')
OFF, DPI, SEED, NB = 30, 300, 42, 2000
# Okabe-Ito colorblind-safe palette
FOOD = "#D55E00"   # vermillion  (food)
TOY  = "#0072B2"   # blue        (toy)
PT   = "#009E73"   # bluish-green (consumers)
NON  = "#E69F00"   # orange       (non-consumers) — distinct from FOOD vermillion
FIT  = "#1F2A44"   # dark navy    (fit lines)
plt.rcParams.update({"figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BETA = "\u03b2"

raw = pd.read_excel(DATA, sheet_name='Sheet 1')
num = lambda c: pd.to_numeric(raw[c], errors='coerce') if c in raw.columns else pd.Series(np.nan, index=raw.index)
df = pd.DataFrame({'redcapid': num('redcapid'), 'eah': num('eah_total_kcal'),
    'ff': num('ff_foodvtoy'), 'ff_f': num('ff_f'), 'ff_t': num('ff_t'),
    'cgd': num('cgd_foodvtoy'), 'cgd_f': num('cgd_f'), 'cgd_t': num('cgd_t'),
    'age': num('conf_age'), 'gender': num('gender'), 'bmi_percentile': num('bmip_bl'), 'income': num('income_imputed')})
df['female'] = (df['gender'] - 1).clip(0, 1)
media = pd.read_csv(SCREEN)
if 'redcap_event_name' in media.columns:
    media = media[media['redcap_event_name'].astype(str).str.contains('baseline', case=False, na=False)]
media['screentime'] = pd.to_numeric(media['mediatotalperweek'], errors='coerce')/60/7
media['redcapid'] = pd.to_numeric(media['redcapid'], errors='coerce')
df['redcapid'] = pd.to_numeric(df['redcapid'], errors='coerce')
df = df.merge(media[['redcapid','screentime']].dropna().drop_duplicates('redcapid'), on='redcapid', how='left')
COVARIATES = ['age', 'female', 'bmi_percentile', 'income']
df = df.dropna(subset=['screentime', 'eah', 'ff', 'cgd'] + COVARIATES).reset_index(drop=True)
eat = df[df['eah'] > 0]; non = df[df['eah'] == 0]

# We hold the other predictors at their sample means. For EAH we fit on the log scale, then back
# transform predictions and confidence limits to kcal.
def adjusted_line(ax, sample, predictor, outcome, label, linestyle,
                  extra=(), band=False, width=2.2):
    """Plot the fitted mean with other predictors at their sample means."""
    columns = [predictor] + COVARIATES + list(extra)
    X = np.column_stack([np.ones(len(sample)), sample[columns].to_numpy()])
    y = sample[outcome].to_numpy()
    if outcome == 'eah':
        y = np.log(np.maximum(y, 0) + OFF)
    inverse = np.linalg.pinv(X)
    beta = inverse @ y
    residual = y - X @ beta
    leverage = np.einsum('ij,ji->i', X, inverse)
    weighted = inverse * (residual / np.maximum(1 - leverage, 1e-10))
    covariance = weighted @ weighted.T

    xs = np.linspace(sample[predictor].min(), sample[predictor].max(), 100)
    grid = np.column_stack([np.ones(len(xs)), xs,
                            np.tile(sample[columns[1:]].mean().to_numpy(), (len(xs), 1))])
    fitted = grid @ beta
    def display(values):
        return np.exp(values) - OFF if outcome == 'eah' else values

    ax.plot(xs, display(fitted), color=FIT, lw=width, ls=linestyle,
            zorder=4, label=f'{label} adjusted fit (n = {len(sample)})')
    if band:
        se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', grid, covariance, grid), 0))
        margin = stats.t.ppf(.975, len(sample) - X.shape[1]) * se
        ax.fill_between(xs, display(fitted - margin), display(fitted + margin),
                        color=FIT, alpha=.13, zorder=2)

def z(s): return (s - s.mean())/s.std(ddof=1)
def std_beta_ci(d):
    # HC3 beta + HC3-based 95% CI (matches manuscript; p and CI from same model)
    d = d.dropna(subset=['screentime','eah','age','female','bmi_percentile','income']).copy()
    for v in ['age','bmi_percentile','income','screentime']: d['z_'+v]=z(d[v])
    d['zy']=z(np.log(d['eah'].clip(lower=0)+OFF))
    cols=['z_age','female','z_bmi_percentile','z_income','z_screentime']
    X=np.column_stack([np.ones(len(d))]+[d[c].values for c in cols])
    XtXi=np.linalg.pinv(X.T@X); beta=XtXi@X.T@d['zy'].values
    resid=d['zy'].values-X@beta; h=np.einsum('ij,jk,ik->i',X,XtXi,X)
    omega=resid**2/(1-h)**2; cov=XtXi@(X.T*omega)@X@XtXi; se=np.sqrt(np.diag(cov))
    dfree=len(d)-X.shape[1]; tc=stats.t.ppf(.975,dfree)
    bt=beta[-1]; s=se[-1]; pval=2*stats.t.sf(abs(bt/s),dfree)
    return bt, bt-tc*s, bt+tc*s, pval
b_full=std_beta_ci(df); b_eat=std_beta_ci(eat)

# We plot individual EAH intake and overlay the two adjusted fits. The shaded band is the HC3
# confidence interval for the full sample fit.
# Fig 1
fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=DPI)
ax.scatter(eat['screentime'], eat['eah'], s=55, color=PT, edgecolor='white', lw=.8, label=f'Consumers (n = {len(eat)})', zorder=3)
if len(non): ax.scatter(non['screentime'], non['eah'], s=95, marker='D', color=NON, edgecolor='white', lw=1, label=f'Non-consumers (n = {len(non)})', zorder=5)
adjusted_line(ax, df, 'screentime', 'eah', 'Full-sample', '-', band=True, width=2.6)
adjusted_line(ax, eat, 'screentime', 'eah', 'Consumers-only', ':')
ann=(f"Full sample:  {BETA} = {b_full[0]:.2f}  [{b_full[1]:.2f}, {b_full[2]:.2f}]\n"
     f"Consumers:  {BETA} = {b_eat[0]:.2f}  [{b_eat[1]:.2f}, {b_eat[2]:.2f}]")
ax.text(0.03,0.97,ann,transform=ax.transAxes,va='top',ha='left',fontsize=9.5,
        bbox=dict(boxstyle='round',facecolor='white',edgecolor='#CBD6CE'))
ax.set_xlabel('Usual screen time (hours/day)'); ax.set_ylabel('EAH intake (kcal)')
ax.set_title('Screen time and Eating in the Absence of Hunger')
ax.set_ylim(top=375)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR,'fig_aim1_screen_eah.png'),dpi=DPI,bbox_inches='tight'); plt.close(fig)

# We show each food and toy condition together with individual values so the reader can see group
# summaries and variation.
def bars(fcol,tcol,ylab,title,fname):
    grp=[('Overall\n(n = %d)'%len(df),df),('Consumers\n(n = %d)'%len(eat),eat),('Non-consumers\n(n = %d)'%len(non),non)]
    xp=np.arange(len(grp)); w=.36
    fig,ax=plt.subplots(figsize=(7.6,5.6),dpi=DPI)
    rng=np.random.default_rng(3)
    diffs=[]
    for i,(gl,g) in enumerate(grp):
        fvals=g[fcol].dropna(); tvals=g[tcol].dropna()
        fm,fsd=fvals.mean(),fvals.std(ddof=1); tm,tsd=tvals.mean(),tvals.std(ddof=1)
        fse=fvals.sem(); tse=tvals.sem()
        ax.bar(i-w/2,fm,w,yerr=fse,capsize=4,color=FOOD,edgecolor='white',zorder=2,label='Food' if i==0 else None)
        ax.bar(i+w/2,tm,w,yerr=tse,capsize=4,color=TOY,edgecolor='white',zorder=2,label='Toy' if i==0 else None)
        ax.scatter(i-w/2+rng.uniform(-.08,.08,len(fvals)),fvals,facecolors='none',edgecolors='#333',s=22,lw=.7,zorder=3)
        ax.scatter(i+w/2+rng.uniform(-.08,.08,len(tvals)),tvals,facecolors='none',edgecolors='#333',s=22,lw=.7,zorder=3)
        d=(g[fcol]-g[tcol]).dropna(); n=len(d)
        if n>1:
            m=d.mean(); se=d.std(ddof=1)/np.sqrt(n); tcr=stats.t.ppf(.975,n-1)
            diffs.append((i,m,m-tcr*se,m+tcr*se))
    ax.axhline(0,color='#999',lw=.8,zorder=1)
    # add headroom at top, then place food-toy difference boxes in that band
    ymax=max([ (g[fcol].max() if g[fcol].notna().any() else 0) for _,g in grp]+
             [ (g[tcol].max() if g[tcol].notna().any() else 0) for _,g in grp])
    ax.set_ylim(top=ymax*1.28)
    ytxt=ymax*1.16
    for (i,m,lo,hi) in diffs:
        ax.text(i, ytxt, f"food\u2212toy\n{m:+.0f} [{lo:+.0f}, {hi:+.0f}]",
                ha='center', va='center', fontsize=8, color='#1F2A44',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CBD6CE', alpha=0.95), zorder=6)
    ax.set_xticks(xp); ax.set_xticklabels([g[0] for g in grp])
    ax.set_ylabel(ylab); ax.set_title(title); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR,fname),dpi=DPI,bbox_inches='tight'); plt.close(fig)
bars('ff_f','ff_t','First-fixation duration (ms)','First fixation: food vs. toy distractors','fig_bars_ff.png')
bars('cgd_f','cgd_t','Cumulative gaze duration (ms)','Cumulative gaze: food vs. toy distractors','fig_bars_cgd.png')

# We display the three component associations separately. Path b holds screen time fixed and path
# c prime holds the mediator fixed in addition to the core covariates.
def med3(med,mlab,fname):
    S=df.dropna(subset=[med,'screentime','eah'])
    Sc=S[S['eah']>0]                 # consumers
    Sn=S[S['eah']==0]                # non-consumers
    _pcsv=os.path.join(_HERE,'..','results','main_analysis','table_aim2_paths.csv')
    A={'a':'','b':'','cp':''}
    if os.path.exists(_pcsv):
        _pp=pd.read_csv(_pcsv); _key='first-fixation' if med=='ff' else 'cumulative-gaze'
        def _row(samp): 
            r=_pp[(_pp['sample'].str.contains(samp))&(_pp['mediator'].str.contains(_key))&(_pp['model']=='single')]
            return r.iloc[0] if len(r) else None
        rf=_row('Full'); rc=_row('Consumers')
        def _pair(path):
            out=[]
            if rf is not None: out.append(f"Full: {BETA} = {rf[path+'_beta']:+.2f} [{rf[path+'_ci_lo']:+.2f}, {rf[path+'_ci_hi']:+.2f}]")
            if rc is not None: out.append(f"Cons: {BETA} = {rc[path+'_beta']:+.2f} [{rc[path+'_ci_lo']:+.2f}, {rc[path+'_ci_hi']:+.2f}]")
            return "\n".join(out)
        A['a']=_pair('a'); A['b']=_pair('b'); A['cp']=_pair('cp')
    def scatter_pts(ax,xcol_c,ycol_c,xcol_n,ycol_n):
        ax.scatter(xcol_c,ycol_c,color=PT,s=45,edgecolor='white',zorder=3,label='Consumers (n = %d)'%len(Sc))
        if len(Sn): ax.scatter(xcol_n,ycol_n,color=NON,marker='D',s=60,edgecolor='white',zorder=4,label='Non-consumers (n = %d)'%len(Sn))
    def fitline(ax, predictor, outcome, extra=()):
        adjusted_line(ax, S, predictor, outcome, 'Full-sample', '-', extra=extra)
        adjusted_line(ax, Sc, predictor, outcome, 'Consumers-only', ':', extra=extra)
    def annotate(ax,txt):
        if txt: ax.text(0.03,0.97,txt,transform=ax.transAxes,va='top',ha='left',fontsize=8,
                        bbox=dict(boxstyle='round',facecolor='white',edgecolor='#CBD6CE'))
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),dpi=DPI)
    # (a) screen -> mediator  (non-consumers have real mediator values)
    scatter_pts(axes[0],Sc['screentime'],Sc[med],Sn['screentime'],Sn[med])
    fitline(axes[0], 'screentime', med)
    axes[0].set_xlabel('Screen time (hours/day)'); axes[0].set_ylabel(mlab+' (ms)'); axes[0].set_title('Path a:  screen time \u2192 '+mlab); annotate(axes[0],A['a'])
    # (b) mediator -> EAH
    scatter_pts(axes[1],Sc[med],Sc['eah'],Sn[med],Sn['eah'])
    fitline(axes[1], med, 'eah', ('screentime',))
    axes[1].set_xlabel(mlab+' (ms)'); axes[1].set_ylabel('EAH intake (kcal)'); axes[1].set_title('Path b:  '+mlab+' \u2192 EAH'); annotate(axes[1],A['b'])
    # (c') screen -> EAH
    scatter_pts(axes[2],Sc['screentime'],Sc['eah'],Sn['screentime'],Sn['eah'])
    fitline(axes[2], 'screentime', 'eah', (med,))
    axes[2].set_xlabel('Screen time (hours/day)'); axes[2].set_ylabel('EAH intake (kcal)'); axes[2].set_title("Path c\u2032:  screen time \u2192 EAH | "+mlab); annotate(axes[2],A['cp'])
    for ax in axes[1:]:
        ax.set_ylim(top=375)
    # single combined legend (markers + line styles) below the panels
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=4,frameon=True,fontsize=8.5,bbox_to_anchor=(0.5,-0.02))
    fig.suptitle(mlab+' mediation: component scatter plots', fontsize=11)
    fig.tight_layout(rect=[0,0.06,1,0.96]); fig.savefig(os.path.join(FIGDIR,fname),dpi=DPI,bbox_inches='tight'); plt.close(fig)
med3('ff','First-fixation bias','fig_med_ff.png')
med3('cgd','Cumulative-gaze bias','fig_med_cgd.png')

# ---------- Q-Q plots: raw EAH vs ln(EAH+30), full sample AND consumers ----------
# We compare raw and log transformed EAH visually in each sample to see how the transformation
# changes the distribution.
def qq_transform(fname):
    fig,axes=plt.subplots(2,2,figsize=(10,8.4),dpi=DPI)
    panels=[('Full sample (n = %d)'%len(df),df),('Consumers (n = %d)'%len(eat),eat)]
    for row,(slabel,d) in enumerate(panels):
        dd=d.dropna(subset=['eah'])
        raw=dd['eah'].values; logv=np.log(dd['eah'].clip(lower=0).values+OFF)
        for col,(vals,vlabel) in enumerate([(raw,'Raw EAH (kcal)'),(logv,'ln(EAH + 30)')]):
            ax=axes[row,col]; stats.probplot(vals,dist='norm',plot=ax)
            ax.set_title(f'{vlabel} \u2014 {slabel}'); ax.get_lines()[0].set_color(PT); ax.get_lines()[1].set_color(FIT)
            ax.set_xlabel('Theoretical quantiles'); ax.set_ylabel('Ordered values')
    fig.suptitle('Distribution of EAH before and after transformation', fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(os.path.join(FIGDIR,fname),dpi=DPI,bbox_inches='tight'); plt.close(fig)
qq_transform('fig_qq_eah_distributions.png')

# ---------- path-diagram figures (schematic box-and-arrow mediation model) ----------
def path_diagram(med_label, a, b, cp, c, indirect_ci, sig_indirect, fname):
    """Box-and-arrow mediation diagram. a,b,cp,c are (beta,p) tuples for FULL sample;
    indirect_ci is (est,lo,hi). Labels use HC3 p for paths, bootstrap CI for indirect."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=DPI); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,6)
    def box(x,y,txt):
        ax.add_patch(plt.Rectangle((x-1.4,y-.55),2.8,1.1,fill=True,facecolor='#F4F4F2',edgecolor='#333',lw=1.2))
        ax.text(x,y,txt,ha='center',va='center',fontsize=10)
    box(2,1.2,'Screen time'); box(8,1.2,'EAH'); box(5,4.6,med_label)
    def arrow(x1,y1,x2,y2,lbl):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=1.6,color='#333'))
        mid_x = (x1 + x2) / 2
        # We place path labels beside diagonal arrows so the line stays legible.
        x_shift = -1.05 if y2 > y1 else (1.05 if y2 < y1 else 0)
        ax.text(mid_x + x_shift, (y1+y2)/2+0.25, lbl, ha='center', fontsize=9, color='#1F2A44')
    arrow(3.0,1.7,4.0,4.1, f"a: {BETA}={a[0]:+.2f} [{a[2]:+.2f}, {a[3]:+.2f}]")
    arrow(6.0,4.1,7.0,1.7, f"b: {BETA}={b[0]:+.2f} [{b[2]:+.2f}, {b[3]:+.2f}]")
    arrow(3.4,1.2,6.6,1.2, f"c\u2032: {BETA}={cp[0]:+.2f} [{cp[2]:+.2f}, {cp[3]:+.2f}]")
    ax.text(5,0.3,f"total c: {BETA}={c[0]:+.2f} [{c[2]:+.2f}, {c[3]:+.2f}]    |    indirect a\u00d7b = {indirect_ci[0]:.3f} [{indirect_ci[1]:.2f}, {indirect_ci[2]:.2f}] ({sig_indirect})",
            ha='center',fontsize=8.5,color='#333')
    ax.set_title(med_label+f' mediation model (full sample, n = {len(df)})',fontsize=11)
    fig.savefig(os.path.join(FIGDIR,fname),dpi=DPI,bbox_inches='tight'); plt.close(fig)

# FF and CGD path diagrams read real values from the analysis output (no hardcoding)
_PATHS_CSV = os.path.join(_HERE, '..', 'results', 'main_analysis', 'table_aim2_paths.csv')
_IND_CSV   = os.path.join(_HERE, '..', 'results', 'main_analysis', 'table_aim2_indirect.csv')
if os.path.exists(_PATHS_CSV) and os.path.exists(_IND_CSV):
    _p = pd.read_csv(_PATHS_CSV); _i = pd.read_csv(_IND_CSV)
    def _diag_for(med_key, med_label, fname):
        pr = _p[(_p['sample'].str.contains('Full'))&(_p['mediator'].str.contains(med_key))&(_p['model']=='single')].iloc[0]
        ir = _i[(_i['sample'].str.contains('Full'))&(_i['mediator'].str.contains(med_key))&(_i['model']=='single')].iloc[0]
        sig = 'significant' if ir['significant']=='SIG' else 'not significant'
        path_diagram(med_label,(pr['a_beta'],pr['a_p'],pr['a_ci_lo'],pr['a_ci_hi']),
                     (pr['b_beta'],pr['b_p'],pr['b_ci_lo'],pr['b_ci_hi']),
                     (pr['cp_beta'],pr['cp_p'],pr['cp_ci_lo'],pr['cp_ci_hi']),
                     (pr['c_beta'],pr['c_p'],pr['c_ci_lo'],pr['c_ci_hi']),
                     (ir['indirect_ab'],ir['ci_lo'],ir['ci_hi']), sig, fname)
    _diag_for('first-fixation','First-fixation bias','fig_path_ff.png')
    _diag_for('cumulative-gaze','Cumulative-gaze bias','fig_path_cgd.png')
else:
    print("  path diagrams SKIPPED — run k01_analysis_corrected.py first")

# We stack the two panel exports for the manuscript version of Figure 4.
if os.path.exists(_PATHS_CSV) and os.path.exists(_IND_CSV):
    combined, panels = plt.subplots(2, 1, figsize=(8.2, 9.2), dpi=DPI)
    for ax, panel, filename in zip(panels, ("A", "B"), ("fig_path_ff.png", "fig_path_cgd.png")):
        ax.imshow(plt.imread(os.path.join(FIGDIR, filename)))
        ax.set_axis_off()
        ax.text(0.01, 0.98, f"({panel})", transform=ax.transAxes, va="top", fontsize=15, weight="bold")
    combined.tight_layout(pad=0.25)
    combined.savefig(os.path.join(FIGDIR, "fig_4_mediation.png"), dpi=DPI, bbox_inches="tight")
    plt.close(combined)

# ---------- Fig: preload specificity check (screen time vs preload intake) ----------
# We use preload intake as a specificity check because it was measured before satiety. We join
# baseline values by study ID.
# Preload intake is NOT in k_ab_bmi.xlsx; it comes from the full REDCap export.
# Reads preload_total_kcal, merges to the analytic sample by ID at baseline.
PRELOAD_FILE = os.path.join(BASE, '5K01MediaUseFullStud_DATA_2020-06-22_1116.csv')
if os.path.exists(PRELOAD_FILE):
    _pl = pd.read_csv(PRELOAD_FILE, low_memory=False)
    _idc = [c for c in _pl.columns if c.lower() in ('redcapid','record_id','study_id')][0]
    _keep = [_idc, 'preload_total_kcal'] + (['redcap_event_name'] if 'redcap_event_name' in _pl.columns else [])
    _plc = _pl[_keep].copy()
    _plc['redcapid'] = pd.to_numeric(_plc[_idc], errors='coerce')
    _plc['preload']  = pd.to_numeric(_plc['preload_total_kcal'], errors='coerce')
    if 'redcap_event_name' in _plc.columns:
        _plc = _plc[_plc['redcap_event_name'].astype(str).str.contains('baseline', case=False, na=False)]
    _plc = _plc.dropna(subset=['preload']).drop_duplicates('redcapid')
    _pf = df.merge(_plc[['redcapid','preload']], on='redcapid', how='left').dropna(subset=['preload','screentime'])

    figp, axp = plt.subplots(figsize=(6.4, 5.0), dpi=DPI)
    _pfc = _pf[_pf['eah'] > 0]   # preload rows that are also consumers
    # HC3 beta + CI + p for screen time -> preload (full and consumers)
    def _pl_beta(dd):
        dd=dd.dropna(subset=['preload','screentime','age','female','bmi_percentile','income'])
        if len(dd)<8: return None
        zz=lambda s:(s-s.mean())/s.std(ddof=1)
        y=zz(dd['preload']).values
        X=np.column_stack([np.ones(len(dd)),zz(dd['screentime']),zz(dd['age']),dd['female'],zz(dd['bmi_percentile']),zz(dd['income'])])
        XtXi=np.linalg.pinv(X.T@X); bb=XtXi@X.T@y; r=y-X@bb; h=np.einsum('ij,jk,ik->i',X,XtXi,X)
        om=r**2/(1-h)**2; cov=XtXi@(X.T*om)@X@XtXi; se=np.sqrt(np.diag(cov)); dfree=len(dd)-6; tc=stats.t.ppf(.975,dfree)
        return bb[1], bb[1]-tc*se[1], bb[1]+tc*se[1], 2*stats.t.sf(abs(bb[1]/se[1]),dfree)
    pf_b=_pl_beta(_pf); pfc_b=_pl_beta(_pfc)
    _pfn = _pf[_pf['eah'] == 0]
    axp.scatter(_pfc['screentime'], _pfc['preload'], s=55, color=PT,
                edgecolor='white', lw=.8, zorder=3, label=f'EAH consumers (n = {len(_pfc)})')
    axp.scatter(_pfn['screentime'], _pfn['preload'], s=65, color=NON, marker='D',
                edgecolor='white', lw=.8, zorder=5, label=f'EAH nonconsumers (n = {len(_pfn)})')
    if pf_b is not None:
        adjusted_line(axp, _pf, 'screentime', 'preload', 'Full preload sample',
                      '-', band=True, width=2.6)
    if pfc_b is not None:
        adjusted_line(axp, _pfc, 'screentime', 'preload', 'Consumers-only', ':')
    axp.set_ylim(top=435)
    axp.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8.5)
    txt = 'Screen time \u2192 preload (specificity check):'
    if pf_b: txt += f"\nFull sample:  {BETA} = {pf_b[0]:+.2f} [{pf_b[1]:+.2f}, {pf_b[2]:+.2f}]"
    if pfc_b: txt += f"\nConsumers:  {BETA} = {pfc_b[0]:+.2f} [{pfc_b[1]:+.2f}, {pfc_b[2]:+.2f}]"
    axp.text(0.03, 0.97, txt, transform=axp.transAxes, va='top', ha='left', fontsize=8.5,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='#CBD6CE'))
    axp.set_xlabel('Usual screen time (hours/day)'); axp.set_ylabel('Preload intake (kcal)')
    axp.set_title('Screen time and preload intake (before satiety)')
    figp.tight_layout(); figp.savefig(os.path.join(FIGDIR, 'fig_preload.png'), dpi=DPI, bbox_inches='tight'); plt.close(figp)
    print("  fig_preload.png  (full n=%d, consumers n=%d)" % (len(_pf), len(_pfc)))
else:
    print("  fig_preload.png SKIPPED — %s not found in data/" % os.path.basename(PRELOAD_FILE))

print("Saved figures. betas: full %.2f [%.2f,%.2f] p=%.3f | eaters %.2f [%.2f,%.2f] p=%.3f"%(b_full+b_eat))
