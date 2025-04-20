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
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9

# Custom CSS
st.markdown(
    """
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2C3E50; }
    .stButton>button { background-color: #4CAF50; color: white; font-weight: bold; }
    .stDownloadButton>button { background-color: #3498DB; color: white; }
    .status-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .status-box.stationary { background-color: #d4edda; border: 1px solid #c3e6cb; }
    .status-box.non-stationary { background-color: #f8d7da; border: 1px solid #f5c6cb; }
    .status-box.inconclusive { background-color: #fff3cd; border: 1px solid #ffeeba; }
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
    y = np.asarray(y)
    n = len(y)
    dy = np.diff(y)
    min_idx = int(trim * n)
    max_idx = int((1 - trim) * n)
    candidates = np.linspace(min_idx, max_idx, num=n_cand, dtype=int)
    best_stat, best_bp = np.inf, None

    if breaks == 1:
        for tb in candidates:
            d1 = np.zeros(n-1); d1[tb:] = 1
            X = np.column_stack([np.ones(n-1), d1])
            if model == 'ct':
                trend = np.arange(1, n)
                dt = np.zeros(n-1); dt[tb:] = trend[:-tb]
                X = np.column_stack([X, trend, dt])
            try:
                res = sm.OLS(dy, X).fit()
                tval = res.tvalues[1]
                if abs(tval) < best_stat:
                    best_stat, best_bp = abs(tval), tb
            except:
                continue
    else:
        for tb1, tb2 in itertools.combinations(candidates, 2):
            d1 = np.zeros(n-1); d2 = np.zeros(n-1)
            d1[tb1:], d2[tb2:] = 1, 1
            X = np.column_stack([np.ones(n-1), d1, d2])
            if model == 'ct':
                trend = np.arange(1, n)
                dt1 = np.zeros(n-1); dt2 = np.zeros(n-1)
                dt1[tb1:], dt2[tb2:] = trend[:-tb1], trend[:-tb2]
                X = np.column_stack([X, trend, dt1, dt2])
            try:
                res = sm.OLS(dy, X).fit()
                tval = res.tvalues[1]
                if abs(tval) < best_stat:
                    best_stat, best_bp = abs(tval), (tb1, tb2)
            except:
                continue

    stat = -best_stat
    crit = LS_CRITICAL[model][breaks]
    if stat < crit['1%']:
        pval = 0.01
    elif stat < crit['5%']:
        pval = 0.05
    elif stat < crit['10%']:
        pval = 0.10
    else:
        pval = 0.12

    return {'stat': stat, 'pvalue': pval, 'crit_vals': crit, 'breakpoints': best_bp}


def lumsdaine_papell_test(y, model='c', trim=0.15, max_lags=12):
    y = np.asarray(y)
    n = len(y)
    # Determine optimal lag by BIC
    best_bic, best_lag = np.inf, 0
    for lag in range(min(max_lags, 4) + 1):
        dy = np.diff(y)
        ylag = y[:-1]
        X = np.column_stack([np.ones(len(dy)-lag), ylag[lag:]])
        for l in range(1, lag+1):
            X = np.column_stack([X, dy[lag-l:-l]])
        res = sm.OLS(dy[lag:], X).fit()
        bic = np.log((res.resid**2).mean()) + (X.shape[1] * np.log(len(dy)-lag)) / (len(dy)-lag)
        if bic < best_bic:
            best_bic, best_lag = bic, lag

    # Grid search for two breaks
    min_idx, max_idx = int(trim*n), int((1-trim)*n)
    candidates = np.linspace(min_idx, max_idx, num=8, dtype=int)
    best_stat, best_bp = np.inf, None

    for tb1, tb2 in itertools.combinations(candidates, 2):
        dy = np.diff(y); ylag = y[:-1]
        d1 = np.zeros(n); d2 = np.zeros(n)
        d1[tb1:], d2[tb2:] = 1, 1
        X = np.column_stack([np.ones(n-best_lag-1), ylag[best_lag:], d1[best_lag:-1], d2[best_lag:-1]])
        if model == 'ct':
            trend = np.arange(1, n-best_lag)
            dt1 = np.zeros(n); dt2 = np.zeros(n)
            dt1[tb1:], dt2[tb2:] = np.arange(1, n-tb1+1), np.arange(1, n-tb2+1)
            X = np.column_stack([X, trend, dt1[best_lag:-1], dt2[best_lag:-1]])
        try:
            res = sm.OLS(dy[best_lag:], X).fit()
            tval = res.tvalues[1]
            if abs(tval) < best_stat:
                best_stat, best_bp = abs(tval), (tb1, tb2)
        except:
            continue

    stat = -best_stat
    crit = {'c': {'1%': -6.74, '5%': -6.16, '10%': -5.89},
            'ct': {'1%': -7.34, '5%': -6.82, '10%': -6.49}}[model]
    if stat < crit['1%']:
        pval = 0.01
    elif stat < crit['5%']:
        pval = 0.05
    elif stat < crit['10%']:
        pval = 0.10
    else:
        pval = 0.12

    return {'stat': stat, 'pvalue': pval, 'crit_vals': crit, 'breakpoints': best_bp, 'lags': best_lag}


def run_tests(ts, cfg):
    results = {}
    if cfg['adf']:
        adf_res = adfuller(ts, regression=cfg['adf_reg'], maxlag=cfg['max_lags'], autolag='AIC')
        results['ADF'] = {'stat': adf_res[0], 'p-value': adf_res[1], **adf_res[4]}
    if cfg['pp']:
        pp = PhillipsPerron(ts, trend=cfg['pp_reg'], lags=int(cfg['max_lags']/2))
        results['PP'] = {'stat': pp.stat, 'p-value': pp.pvalue, **pp.critical_values}
    if cfg['kpss']:
        k_stat, k_p, k_lag, k_crit = kpss(ts, regression=cfg['kpss_reg'], nlags='auto')
        results['KPSS'] = {'stat': k_stat, 'p-value': k_p, **k_crit}
    if cfg['za']:
        za = ZivotAndrews(ts, lags=cfg['max_lags'], regression=cfg['za_reg'])
        results['ZA'] = {'stat': za.stat, 'p-value': za.pvalue, 'break': ts.index[za.break_idx].strftime('%Y-%m-%d'), **za.critical_values}
    if cfg['ls1']:
        ls1 = lee_strazicich_test(ts.values, model=cfg['ls_model'], breaks=1)
        bp = ls1['breakpoints']
        results['LS1'] = {'stat': ls1['stat'], 'p-value': ls1['pvalue'], 'break': ts.index[bp].strftime('%Y-%m-%d'), **ls1['crit_vals']}
    if cfg['ls2']:
        ls2 = lee_strazicich_test(ts.values, model=cfg['ls_model'], breaks=2)
        bps = ls2['breakpoints']
        results['LS2'] = {'stat': ls2['stat'], 'p-value': ls2['pvalue'], 'breaks': ','.join([ts.index[i].strftime('%Y-%m-%d') for i in bps]), **ls2['crit_vals']}
    if cfg['lp']:
        lp = lumsdaine_papell_test(ts.values, model=cfg['lp_reg'], max_lags=cfg['max_lags'])
        results['LP'] = {'stat': lp['stat'], 'p-value': lp['pvalue'], 'breaks': str(lp['breakpoints']), **lp['crit_vals']}
    if cfg['dfgls']:
        trend = cfg['adf_reg'] in ['ct','ctt']
        dfg = DFGLS(ts, lags=cfg['max_lags'], trend=trend)
        results['DF-GLS'] = {'stat': dfg.stat, 'p-value': dfg.pvalue, **dfg.critical_values}
    if cfg['vr']:
        vr = VarianceRatio(ts, lags=4)
        results['VR'] = {'stat': vr.stat, 'p-value': vr.pvalue, **vr.critical_values}
    return results

# --- Sidebar & Main App ---
st.sidebar.title("Test Configuration")
uploaded = st.sidebar.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])
if uploaded:
    # Load data
    df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
    df.columns = [c.strip().lower() for c in df.columns]
    st.success(f"Loaded {uploaded.name}")
    with st.expander("Data Preview", expanded=True):
        st.dataframe(df.head(10))

    date_col = st.sidebar.selectbox("Date column", df.columns)
    value_col = st.sidebar.selectbox("Value column", df.columns)
    run_adf = st.sidebar.checkbox("ADF", True)
    adf_reg = st.sidebar.selectbox("ADF Regression", ["c","ct","n","ctt"], index=1)
    run_pp = st.sidebar.checkbox("PP", True)
    pp_reg = st.sidebar.selectbox("PP Regression", ["c","ct","n"], index=1)
    run_kpss = st.sidebar.checkbox("KPSS", True)
    kpss_reg = st.sidebar.selectbox("KPSS Regression", ["c","ct"], index=0)
    run_za = st.sidebar.checkbox("Zivot-Andrews", True)
    za_reg = st.sidebar.selectbox("ZA Regression", ["c","t","both"], index=2)
    run_ls1 = st.sidebar.checkbox("Lee-Strazicich 1 break", False)
    run_ls2 = st.sidebar.checkbox("Lee-Strazicich 2 breaks", False)
    ls_model = st.sidebar.selectbox("LS Model", ["c","ct"], index=0)
    run_lp = st.sidebar.checkbox("Lumsdaine-Papell", False)
    lp_reg = st.sidebar.selectbox("LP Regression", ["c","ct"], index=0)
    run_dfgls = st.sidebar.checkbox("DF-GLS", False)
    run_vr = st.sidebar.checkbox("Variance Ratio", False)
    max_lags = st.sidebar.slider("Max lags", 1, 24, 12)
    
    if st.sidebar.button("Run Analysis"):
        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df[[date_col, value_col]].dropna()
        df.set_index(date_col, inplace=True)
        ts = df[value_col]
        if len(ts) < 20:
            st.error("Too few observations (<20)")
        else:
            st.success(f"Series from {ts.index.min()} to {ts.index.max()} ({len(ts)} obs)")
            # Plot
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(ts.index, ts.values)
            ax.set_title("Time Series")
            ax.grid(True)
            st.pyplot(fig)
            
            # Run tests
            cfg = {
                'adf': run_adf, 'adf_reg': adf_reg,
                'pp': run_pp, 'pp_reg': pp_reg,
                'kpss': run_kpss, 'kpss_reg': kpss_reg,
                'za': run_za, 'za_reg': za_reg,
                'ls1': run_ls1, 'ls2': run_ls2, 'ls_model': ls_model,
                'lp': run_lp, 'lp_reg': lp_reg,
                'dfgls': run_dfgls, 'vr': run_vr,
                'max_lags': max_lags
            }
            results = run_tests(ts, cfg)

            # Summary Table
            summary = pd.DataFrame({k: {'stat': v['stat'], 'p-value': v['p-value']} for k,v in results.items()}).T
            st.subheader("Test Summary")
            st.dataframe(summary.style.format({'stat':'{:.4f}','p-value':'{:.4f}'}))

            # Interpretation
            stationary = nonstat = 0
            lines = []
            for k, v in results.items():
                p = v['p-value']
                if (k=='KPSS' and p>=0.05) or (k!='KPSS' and p<0.05):
                    lines.append(f"• {k}: Stationary (p={p:.3f})"); stationary+=1
                else:
                    lines.append(f"• {k}: Non-stationary (p={p:.3f})"); nonstat+=1
            status = "Stationary" if stationary>nonstat else "Non-Stationary" if nonstat>stationary else "Inconclusive"
            cls = status.lower().replace(' ', '-')
            st.markdown(f"<div class='status-box {cls}'><h3>{status}</h3><p>{'<br>'.join(lines)}</p></div>", unsafe_allow_html=True)

            # Heatmap
            st.subheader("P-value Heatmap")
            fig, ax = plt.subplots(figsize=(4, len(results)*0.4+1))
            data = np.array([v['p-value'] for v in results.values()]).reshape(-1,1)
            im = ax.imshow(data, aspect='auto', vmin=0, vmax=0.1, cmap='RdYlGn_r')
            ax.set_yticks(range(len(results))); ax.set_yticklabels(results.keys())
            ax.set_xticks([0]); ax.set_xticklabels(['p-value'])
            fig.colorbar(im, ax=ax, label='p-value')
            st.pyplot(fig)

            # Downloads
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                summary.to_excel(writer, sheet_name='Summary')
                pd.DataFrame(results).T.to_excel(reader, sheet_name='Detailed')
                df.to_excel(writer, sheet_name='Data')
            buffer.seek(0)
            st.download_button("Download Excel Report", data=buffer.getvalue(), file_name=f"unitroot_{datetime.now().strftime('%Y%m%d')}.xlsx", mime='application/vnd.ms-excel')

else:
    st.info("Upload a CSV/XLSX or load sample below.")
    if st.button("Load Sample"):
        dates = pd.date_range('2010-01', periods=120, freq='M')
        trend = np.arange(120)*0.1; trend[60:] += 5
        season = 2*np.sin(np.arange(120)*2*np.pi/12); noise = np.random.normal(0,1,120)
        rw = np.cumsum(np.random.normal(0,0.5,120))
        sample = pd.DataFrame({'date': dates, 'value': trend+season+noise+rw})
        st.dataframe(sample.head(10))
        buf = io.BytesIO(); sample.to_csv(buf, index=False); buf.seek(0)
        st.download_button("Download Sample CSV", data=buf.getvalue(), file_name='sample.csv', mime='text/csv')

# Footer
st.markdown("---")
st.markdown("© 2025 Advanced Unit Root Test App")

