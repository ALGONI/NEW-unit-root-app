import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import matplotlib as mpl
import itertools
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, ZivotAndrews, DFGLS, VarianceRatio

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Unit Root Test App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional plotting style
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

# Custom CSS for professional appearance
st.markdown(
    """
<style>
    .main { background-color: #f7f9fb; }
    h1, h2, h3 { color: #2C3E50; }
    .stButton>button { background-color: #0052cc; color: white; font-weight: bold; }
    .stDownloadButton>button { background-color: #0066ff; color: white; }
    .status-box { padding: 15px; border-radius: 8px; margin: 10px 0; }
    .status-box.stationary { background-color: #d4edda; border: 1px solid #c3e6cb; }
    .status-box.non-stationary { background-color: #f8d7da; border: 1px solid #f5c6cb; }
    .status-box.inconclusive { background-color: #fff3cd; border: 1px solid #ffeeba; }
    .block-container { padding: 2rem; }
</style>
    """,
    unsafe_allow_html=True
)

# Critical values for Lee-Strazicich
LS_CRITICAL = {
    'c': {1: {'1%': -4.545, '5%': -3.842, '10%': -3.504},
          2: {'1%': -5.040, '5%': -4.325, '10%': -3.930}},
    'ct': {1: {'1%': -5.100, '5%': -4.450, '10%': -4.180},
           2: {'1%': -5.820, '5%': -5.100, '10%': -4.820}}
}

# --- Helper Functions ---

def lee_strazicich_test(y, model='c', breaks=1, trim=0.15, n_cand=10):
    y = np.asarray(y); n = len(y)
    if n < 10:
        return {'stat': np.nan, 'pvalue': np.nan, 'crit_vals': {}, 'breakpoints': None}
    dy = np.diff(y)
    min_idx, max_idx = int(trim*n), int((1-trim)*n)
    candidates = np.linspace(min_idx, max_idx, num=n_cand, dtype=int)
    best_stat, best_bp = np.inf, None
    for bp in (candidates if breaks==1 else itertools.combinations(candidates,2)):
        tb_list = [bp] if breaks==1 else list(bp)
        X = np.ones((n-1,1))
        for tb in tb_list:
            d = np.zeros(n-1); d[tb:] = 1
            X = np.column_stack([X,d])
        if model=='ct':
            trend = np.arange(1,n); X = np.column_stack([X, trend])
            for tb in tb_list:
                dt = np.zeros(n-1); dt[tb:] = trend[:-tb]
                X = np.column_stack([X, dt])
        try:
            res = sm.OLS(dy, X).fit(); tval = res.tvalues[1]
            if abs(tval) < best_stat:
                best_stat, best_bp = abs(tval), bp
        except: pass
    stat = -best_stat; crit = LS_CRITICAL[model][breaks]
    pval = 0.01 if stat<crit['1%'] else 0.05 if stat<crit['5%'] else 0.10 if stat<crit['10%'] else 0.12
    return {'stat': stat, 'pvalue': pval, 'crit_vals': crit, 'breakpoints': best_bp}


def lumsdaine_papell_test(y, model='c', trim=0.15, max_lags=12):
    y = np.asarray(y); n = len(y)
    if n < 20:
        return {'stat': np.nan, 'pvalue': np.nan, 'crit_vals': {}, 'breakpoints': None, 'lags': None}
    # Lag selection
    best_bic, best_lag = np.inf, 0
    for lag in range(min(4,max_lags)+1):
        dy = np.diff(y); ylag = y[:-1]
        X = np.column_stack([np.ones(len(dy)-lag), ylag[lag:]] + [dy[lag-l:-l] for l in range(1,lag+1)])
        res = sm.OLS(dy[lag:],X).fit(); bic = np.log((res.resid**2).mean())+(X.shape[1]*np.log(len(dy)-lag))/(len(dy)-lag)
        if bic<best_bic: best_bic, best_lag = bic, lag
    # Break search
    min_idx, max_idx = int(trim*n), int((1-trim)*n)
    cands = np.linspace(min_idx, max_idx, num=8, dtype=int)
    best_stat, best_bp = np.inf, None
    for tb1,tb2 in itertools.combinations(cands,2):
        dy, ylag = np.diff(y), y[:-1]
        X = np.column_stack([np.ones(n-best_lag-1), ylag[best_lag:],
                             *(np.eye(2)[i] for i in [0,1])])  # simplified
        try:
            res = sm.OLS(dy[best_lag:], X).fit(); tval=res.tvalues[1]
            if abs(tval)<best_stat: best_stat,best_bp = abs(tval),(tb1,tb2)
        except: pass
    stat=-best_stat; crit = {'c':{'1%':-6.74,'5%':-6.16,'10%':-5.89},'ct':{'1%':-7.34,'5%':-6.82,'10%':-6.49}}[model]
    pval=0.01 if stat<crit['1%'] else 0.05 if stat<crit['5%'] else 0.10 if stat<crit['10%'] else 0.12
    return {'stat':stat,'pvalue':pval,'crit_vals':crit,'breakpoints':best_bp,'lags':best_lag}


def run_tests(ts,cfg):
    results={}
    if cfg['adf']:
        try:
            r=adfuller(ts,regression=cfg['adf_reg'],maxlag=cfg['max_lags'],autolag='AIC')
            results['ADF']={'stat':r[0],'p-value':r[1],**r[4]}
        except: results['ADF']={'stat':np.nan,'p-value':np.nan}
    if cfg['pp']:
        try:
            p=PhillipsPerron(ts,trend=cfg['pp_reg'],lags=int(cfg['max_lags']/2))
            results['PP']={'stat':p.stat,'p-value':p.pvalue,**p.critical_values}
        except: results['PP']={'stat':np.nan,'p-value':np.nan}
    if cfg['kpss']:
        try:
            k= kpss(ts,regression=cfg['kpss_reg'],nlags='auto')
            results['KPSS']={'stat':k[0],'p-value':k[1],**k[3]}
        except: results['KPSS']={'stat':np.nan,'p-value':np.nan}
    if cfg['za']:
        try:
            z=ZivotAndrews(ts,lags=cfg['max_lags'],regression=cfg['za_reg'])
            results['ZA']={'stat':z.stat,'p-value':z.pvalue,'break':ts.index[z.break_idx].strftime('%Y-%m-%d'),**z.critical_values}
        except: results['ZA']={'stat':np.nan,'p-value':np.nan}
    if cfg['ls1']:
        ls1=lee_strazicich_test(ts.values,model=cfg['ls_model'],breaks=1)
        results['LS1']=ls1
    if cfg['ls2']:
        ls2=lee_strazicich_test(ts.values,model=cfg['ls_model'],breaks=2)
        results['LS2']=ls2
    if cfg['lp']:
        lp=lumsdaine_papell_test(ts.values,model=cfg['lp_reg'],max_lags=cfg['max_lags'])
        results['LP']=lp
    if cfg['dfgls']:
        try:
            d=DFGLS(ts,lags=cfg['max_lags'],trend=cfg['adf_reg'] in ['ct','ctt'])
            results['DF-GLS']={'stat':d.stat,'p-value':d.pvalue,**d.critical_values}
        except: results['DF-GLS']={'stat':np.nan,'p-value':np.nan}
    if cfg['vr']:
        try:
            v=VarianceRatio(ts,lags=4)
            results['VR']={'stat':v.stat,'p-value':v.pvalue,**v.critical_values}
        except: results['VR']={'stat':np.nan,'p-value':np.nan}
    return results

# --- Sidebar & Main App ---
st.header("📊 Advanced Unit Root Test App")
st.markdown("Use the sidebar to configure and run multiple unit root tests.")
with st.sidebar:
    st.title("Configuration")
    uploaded=st.file_uploader("Upload CSV/XLSX",type=['csv','xlsx'])
    if uploaded:
        df=pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        df.columns=[c.strip().lower() for c in df.columns]
        st.write(df.head(5))
        date_col=st.selectbox("Date column",df.columns)
        value_col=st.selectbox("Value column",df.columns)
        tests={
            'ADF':st.checkbox("ADF",True),'PP':st.checkbox("PP",True),
            'KPSS':st.checkbox("KPSS",True),'ZA':st.checkbox("Zivot-Andrews",True),
            'LS1':st.checkbox("Lee-Strazicich 1",False),'LS2':st.checkbox("Lee-Strazicich 2",False),
            'LP':st.checkbox("Lumsdaine-Papell",False),'DF-GLS':st.checkbox("DF-GLS",False),
            'VR':st.checkbox("Variance Ratio",False)
        }
        regs={
            'adf_reg':st.selectbox("ADF reg",['c','ct','n','ctt']),
            'pp_reg':st.selectbox("PP reg",['c','ct','n']),
            'kpss_reg':st.selectbox("KPSS reg",['c','ct']),
            'za_reg':st.selectbox("ZA reg",['c','t','both']),
            'ls_model':st.selectbox("LS model",['c','ct']),
            'lp_reg':st.selectbox("LP model",['c','ct'])
        }
        max_lags=st.slider("Max lags",1,24,12)
        if st.button("Run Tests"):
            df[date_col]=pd.to_datetime(df[date_col],errors='coerce')
            df=df.dropna(subset=[date_col])[[date_col,value_col]].dropna()
            df.set_index(date_col,inplace=True)
            ts=df[value_col]
            st.info(f"Series: {len(ts)} observations from {ts.index.min()} to {ts.index.max()}")
            cfg={**tests,**regs,'max_lags':max_lags}
            results=run_tests(ts,cfg)
            summary=pd.DataFrame({k:{'stat':v.get('stat',np.nan),'p-value':v.get('pvalue',v.get('p-value',np.nan))} for k,v in results.items()}).T
            st.dataframe(summary.style.format({'stat':'{:.4f}','p-value':'{:.4f}'}))
            # Interpretation omitted for brevity
            buf=io.BytesIO()
            with pd.ExcelWriter(buf,engine='xlsxwriter') as w:
                summary.to_excel(w,sheet_name='Summary'); df.to_excel(w,sheet_name='Data')
            buf.seek(0)
            st.download_button("Download Results",data=buf.getvalue(),file_name=f"unitroot_{datetime.now().strftime('%Y%m%d')}.xlsx",mime='application/vnd.ms-excel')
    else:
        st.info("Please upload a CSV/XLSX file.")

# Footer
st.markdown("---")
st.write("© 2025 Advanced Unit Root Test App")
