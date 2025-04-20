import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib as mpl
from datetime import datetime
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, ZivotAndrews, DFGLS, VarianceRatio

# Set professional plotting style
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9

# --- Page Settings ---
st.set_page_config(
    page_title="Advanced Unit Root Test App", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #3498DB;
        color: white;
    }
    .css-18e3th9 {
        padding-top: 1rem;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .st-cb {
        border-color: #4CAF50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #4CAF50;
    }
    div.block-container {
        padding-top: 2rem;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .status-box.stationary {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .status-box.non-stationary {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .status-box.inconclusive {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for unit root tests

# Function for Lee-Strazicich test
def lee_strazicich_test(y, model='c', breaks=1, trim=0.15):
    """
    Implements a simplified version of Lee-Strazicich test
    
    Parameters:
    -----------
    y : array_like
        The time series data
    model : str
        'c' for crash model (level shift), 'ct' for trend shift
    breaks : int
        Number of breaks (1 or 2)
    trim : float
        Trimming parameter
    
    Returns:
    --------
    dict : Dictionary with test results
    """
    y = np.asarray(y)
    nobs = y.shape[0]
    
    # First differences
    dy = np.diff(y)
    
    # Potential break dates
    min_idx = int(trim * nobs)
    max_idx = int((1 - trim) * nobs)
    break_range = range(min_idx, max_idx)
    
    if breaks == 1:
        best_stat = np.inf
        best_break = None
        
        # Grid search for single break
        for tb1 in break_range:
            # Create dummy variables
            d1 = np.zeros(nobs - 1)
            d1[tb1:] = 1  # Level shift
            
            dt1 = np.zeros(nobs - 1)
            if model == 'ct':
                dt1[tb1:] = np.arange(1, nobs - tb1)  # Trend shift
            
            # Construct augmented regression
            X = np.ones((nobs - 1, 1))
            if model == 'c':
                X = np.column_stack((X, d1))
            else:  # model == 'ct'
                X = np.column_stack((X, d1, np.arange(1, nobs), dt1))
            
            # Run regression
            model_fit = sm.OLS(dy, X).fit()
            t_stat = model_fit.tvalues[0]  # Get t-stat for lagged level
            
            if abs(t_stat) < best_stat:
                best_stat = abs(t_stat)
                best_break = tb1
        
        breakpoints = [best_break]
        stat = -best_stat
        
    else:  # breaks == 2
        best_stat = np.inf
        best_breaks = None
        
        # Grid search for two breaks (simplified, not exhaustive)
        for tb1 in break_range[:-20]:  # Ensure gap between breaks
            for tb2 in range(tb1 + 20, max_idx):
                # Create dummy variables
                d1 = np.zeros(nobs - 1)
                d1[tb1:] = 1
                
                d2 = np.zeros(nobs - 1)
                d2[tb2:] = 1
                
                dt1 = np.zeros(nobs - 1)
                dt2 = np.zeros(nobs - 1)
                
                if model == 'ct':
                    dt1[tb1:] = np.arange(1, nobs - tb1)
                    dt2[tb2:] = np.arange(1, nobs - tb2)
                
                # Construct augmented regression
                X = np.ones((nobs - 1, 1))
                if model == 'c':
                    X = np.column_stack((X, d1, d2))
                else:  # model == 'ct'
                    X = np.column_stack((X, d1, d2, np.arange(1, nobs), dt1, dt2))
                
                # Run regression
                model_fit = sm.OLS(dy, X).fit()
                t_stat = model_fit.tvalues[0]
                
                if abs(t_stat) < best_stat:
                    best_stat = abs(t_stat)
                    best_breaks = (tb1, tb2)
        
        breakpoints = list(best_breaks) if best_breaks else [None, None]
        stat = -best_stat
    
    # Define critical values (approximations)
    if model == 'c' and breaks == 1:
        crit_vals = {'1%': -4.545, '5%': -3.842, '10%': -3.504}
    elif model == 'c' and breaks == 2:
        crit_vals = {'1%': -5.040, '5%': -4.325, '10%': -3.930}
    elif model == 'ct' and breaks == 1:
        crit_vals = {'1%': -5.100, '5%': -4.450, '10%': -4.180}
    else:  # model == 'ct' and breaks == 2
        crit_vals = {'1%': -5.820, '5%': -5.100, '10%': -4.820}
    
    # Determine p-value based on critical values (approximation)
    if stat < crit_vals['1%']:
        pvalue = 0.01
    elif stat < crit_vals['5%']:
        pvalue = 0.05
    elif stat < crit_vals['10%']:
        pvalue = 0.10
    else:
        pvalue = 0.12  # Above 10%
    
    return {
        'stat': stat,
        'pvalue': pvalue,
        'crit_vals': crit_vals,
        'breakpoints': breakpoints,
        'model': model,
        'nbreaks': breaks
    }

# Function for Lumsdaine-Papell test with 2 breaks
def lumsdaine_papell_test(y, model='c', trim=0.15, max_lags=12):
    """
    Implements a simplified version of Lumsdaine-Papell test with two structural breaks
    
    Parameters:
    -----------
    y : array_like
        The time series data
    model : str
        'c' for crash model (level shift), 'ct' for trend shift
    trim : float
        Trimming parameter
    max_lags : int
        Maximum number of lagged differences to include
    
    Returns:
    --------
    dict : Dictionary with test results
    """
    y = np.asarray(y)
    nobs = y.shape[0]
    
    # Determine optimal lag length (simplified)
    best_bic = np.inf
    best_lag = 0
    
    for lag in range(min(4, max_lags)):
        # Simple differencing
        dy = np.diff(y)
        y_lag = y[:-1]
        
        # Create lag matrix
        X = np.ones((nobs - lag - 1, 1))
        X = np.column_stack((X, y_lag[lag:], np.arange(1, nobs - lag)))
        
        # Add lagged differences
        for l in range(1, lag+1):
            X = np.column_stack((X, dy[lag-l:-l]))
        
        # Fit model
        model_fit = sm.OLS(dy[lag:], X).fit()
        bic = np.log(np.sum(model_fit.resid**2) / (nobs - lag - 1)) + (X.shape[1] * np.log(nobs - lag - 1)) / (nobs - lag - 1)
        
        if bic < best_bic:
            best_bic = bic
            best_lag = lag
    
    # Grid search for breaks
    min_idx = int(trim * nobs)
    max_idx = int((1 - trim) * nobs)
    
    best_stat = np.inf
    best_breaks = None
    
    # Simplified grid search for computational efficiency
    step = max(1, (max_idx - min_idx) // 10)  # Sample 10 potential points for faster computation
    
    for tb1 in range(min_idx, max_idx - min_idx, step):
        for tb2 in range(tb1 + min_idx, max_idx, step):
            # Create dummy variables
            d1 = np.zeros(nobs)
            d1[tb1:] = 1
            
            d2 = np.zeros(nobs)
            d2[tb2:] = 1
            
            dt1 = np.zeros(nobs)
            dt2 = np.zeros(nobs)
            
            if model == 'ct':
                dt1[tb1:] = np.arange(1, nobs - tb1 + 1)
                dt2[tb2:] = np.arange(1, nobs - tb2 + 1)
            
            # Construct regression variables
            dy = np.diff(y)
            y_lag = y[:-1]
            
            # Create model matrix
            X = np.ones((nobs - best_lag - 1, 1))
            X = np.column_stack((X, y_lag[best_lag:], d1[best_lag:-1], d2[best_lag:-1]))
            
            if model == 'ct':
                trend = np.arange(1, nobs - best_lag)
                X = np.column_stack((X, trend, dt1[best_lag:-1], dt2[best_lag:-1]))
            
            # Add lagged differences
            for l in range(1, best_lag+1):
                X = np.column_stack((X, dy[best_lag-l:-l]))
            
            # Fit model
            try:
                model_fit = sm.OLS(dy[best_lag:], X).fit()
                t_stat = model_fit.tvalues[1]  # t-stat on the lagged level
                
                if abs(t_stat) < best_stat:
                    best_stat = abs(t_stat)
                    best_breaks = (tb1, tb2)
            except:
                continue  # Skip in case of singular matrix
    
    # Define critical values (approximations)
    if model == 'c':
        crit_vals = {'1%': -6.74, '5%': -6.16, '10%': -5.89}
    else:  # model == 'ct'
        crit_vals = {'1%': -7.34, '5%': -6.82, '10%': -6.49}
    
    # Calculate stat and p-value
    stat = -best_stat
    
    # Determine p-value (approximation)
    if stat < crit_vals['1%']:
        pvalue = 0.01
    elif stat < crit_vals['5%']:
        pvalue = 0.05
    elif stat < crit_vals['10%']:
        pvalue = 0.10
    else:
        pvalue = 0.12  # Above 10%
    
    breakpoints = list(best_breaks) if best_breaks else [None, None]
    
    return {
        'stat': stat,
        'pvalue': pvalue,
        'crit_vals': crit_vals,
        'breakpoints': breakpoints,
        'lags': best_lag,
        'model': model
    }

# Sidebar for configuration
st.sidebar.image("https://raw.githubusercontent.com/statsmodels/statsmodels/main/docs/source/_static/statsmodels-logo-v2-horizontal.svg", width=200)
st.sidebar.title("Test Configuration")

# Main content
st.title("📊 Advanced Unit Root Test Application")
st.markdown("#### Analyze time series stationarity with detection of multiple structural breaks")

# File upload section
uploaded_file = st.file_uploader("📁 Upload Time Series Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Display success message
    st.success(f"Successfully loaded: {uploaded_file.name}")
    
    # STEP 1: Load CSV or Excel
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # STEP 2: Clean column names (lowercase, strip spaces)
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Show data preview
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(10))
            st.text(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # STEP 3: Let user select which columns to use
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("📅 Select Date Column", options=df.columns)
        with col2:
            value_col = st.selectbox("📈 Select Value Column", options=df.columns)
        
        # Test selection
        st.sidebar.subheader("Select Tests to Run")
        run_adf = st.sidebar.checkbox("Augmented Dickey-Fuller (ADF)", value=True)
        run_pp = st.sidebar.checkbox("Phillips-Perron (PP)", value=True)
        run_kpss = st.sidebar.checkbox("KPSS", value=True)
        run_za = st.sidebar.checkbox("Zivot-Andrews (1 break)", value=True)
        run_ls_1 = st.sidebar.checkbox("Lee-Strazicich (1 break)", value=False)
        run_ls_2 = st.sidebar.checkbox("Lee-Strazicich (2 breaks)", value=False)
        run_lp = st.sidebar.checkbox("Lumsdaine-Papell (2 breaks)", value=False)
        run_dfgls = st.sidebar.checkbox("DF-GLS", value=False)
        run_vr = st.sidebar.checkbox("Variance Ratio", value=False)
        
        # Test parameters
        st.sidebar.subheader("Test Parameters")
        
        # ADF parameters
        adf_regression = st.sidebar.selectbox(
            "ADF Regression Type", 
            options=["c", "ct", "n", "ctt"],
            format_func=lambda x: {
                "c": "Constant",
                "ct": "Constant & Trend",
                "n": "No Constant or Trend",
                "ctt": "Constant & Quadratic Trend"
            }.get(x),
            index=1
        )
        
        # PP parameters
        pp_regression = st.sidebar.selectbox(
            "PP Regression Type", 
            options=["c", "ct", "n"],
            format_func=lambda x: {
                "c": "Constant",
                "ct": "Constant & Trend",
                "n": "No Constant or Trend"
            }.get(x),
            index=1
        )
        
        # KPSS parameters
        kpss_regression = st.sidebar.selectbox(
            "KPSS Regression Type", 
            options=["c", "ct"],
            format_func=lambda x: {
                "c": "Constant",
                "ct": "Constant & Trend"
            }.get(x),
            index=0
        )
        
        # ZA parameters
        za_regression = st.sidebar.selectbox(
            "Zivot-Andrews Model", 
            options=["c", "t", "both"],
            format_func=lambda x: {
                "c": "Break in Constant",
                "t": "Break in Trend",
                "both": "Break in Constant & Trend"
            }.get(x),
            index=2
        )
        
        # LS parameters
        ls_model = st.sidebar.selectbox(
            "Lee-Strazicich Model",
            options=["c", "ct"],
            format_func=lambda x: {
                "c": "Crash Model (Level Shift)", 
                "ct": "Trend Shift Model"
            }.get(x),
            index=0
        )
        
        # LP parameters
        lp_model = st.sidebar.selectbox(
            "Lumsdaine-Papell Model",
            options=["c", "ct"],
            format_func=lambda x: {
                "c": "Crash Model (Level Shift)", 
                "ct": "Trend Shift Model"
            }.get(x),
            index=0
        )
        
        # Max lags for tests
        max_lags = st.sidebar.slider("Maximum Lags", min_value=1, max_value=24, value=12)
        
        # Process Button
        process_button = st.sidebar.button("▶️ Run Analysis", use_container_width=True)
        
        if process_button:
            try:
                # Data processing with robust date parsing
                try:
                    # First attempt standard parsing
                    df[date_col] = pd.to_datetime(df[date_col])
                except ValueError as e:
                    # Check for monthly format like '2013M01'
                    if any(str(x).endswith('M01') or str(x).endswith('M12') for x in df[date_col].head()):
                        st.info("Detected monthly format data. Attempting special parsing...")
                        # Custom parse for monthly format 'YYYYMM' or 'YYYYMDD'
                        df[date_col] = pd.to_datetime(df[date_col].astype(str).str.replace('M', '-'), format='%Y-%m', errors='coerce')
                    else:
                        # Try other common formats
                        formats_to_try = ['%Y-%m', '%Y%m', '%Y/%m', '%Y.%m', '%Y-%m-%d', '%Y%m%d', '%m/%d/%Y', '%d/%m/%Y']
                        for fmt in formats_to_try:
                            try:
                                df[date_col] = pd.to_datetime(df[date_col], format=fmt)
                                st.success(f"Successfully parsed dates using format: {fmt}")
                                break
                            except:
                                continue
                        else:
                            # If all parsing attempts fail, use pandas' flexible parser with error handling
                            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                            # Check for NaT and warn user
                            if df[date_col].isna().any():
                                st.warning(f"Some dates could not be parsed. {df[date_col].isna().sum()} rows will be dropped.")
                
                # Drop NaN values and set index
                df = df[[date_col, value_col]].dropna()
                df.set_index(date_col, inplace=True)
                ts = df[value_col]
                
                if len(ts) < 20:
                    st.error("Time series is too short (less than 20 observations). Please use a longer time series.")
                    st.stop()
                
                st.success(f"Successfully processed {len(ts)} time series observations from {ts.index.min()} to {ts.index.max()}")
                
                st.subheader("📉 Time Series Visualization")
                
                # Create tabs for different visualizations
                tab1, tab2 = st.tabs(["Line Chart", "Interactive Chart"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(ts.index, ts.values, linewidth=2)
                    ax.set_title(f"Time Series: {value_col}", fontsize=14)
                    ax.set_xlabel("Date")
                    ax.set_ylabel(value_col)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with tab2:
                    st.line_chart(ts, use_container_width=True)
                
                # Run the selected tests
                results = {}
                break_dates = {}  # Store break dates for each test
                
                with st.spinner("Running unit root tests..."):
                    # ADF Test
                    if run_adf:
                        try:
                            adf_result = adfuller(ts, regression=adf_regression, maxlag=max_lags, autolag='AIC')
                            results['ADF'] = {
                                'Test Statistic': adf_result[0],
                                'p-value': adf_result[1],
                                'Critical Values (1%)': adf_result[4]['1%'],
                                'Critical Values (5%)': adf_result[4]['5%'],
                                'Critical Values (10%)': adf_result[4]['10%'],
                                'Lags': adf_result[2],
                                'Observations': adf_result[3],
                                'Regression Type': adf_regression,
                                'Breakpoint': 'N/A'
                            }
                        except Exception as e:
                            st.error(f"Error in ADF test: {str(e)}")
                    
                    # PP Test
                    if run_pp:
                        try:
                            pp = PhillipsPerron(ts, trend=pp_regression, lags=int(max_lags/2))
                            results['Phillips-Perron'] = {
                                'Test Statistic': pp.stat,
                                'p-value': pp.pvalue,
                                'Critical Values (1%)': pp.critical_values['1%'],
                                'Critical Values (5%)': pp.critical_values['5%'],
                                'Critical Values (10%)': pp.critical_values['10%'],
                                'Lags': pp.lags,
                                'Regression Type': pp_regression,
                                'Breakpoint': 'N/A'
                            }
                        except Exception as e:
                            st.error(f"Error in Phillips-Perron test: {str(e)}")
                    
                    # KPSS Test
                    if run_kpss:
                        try:
                            kpss_lags = int(4 * (len(ts)/100)**(1/4)) if max_lags is None else max_lags
                            kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(ts, regression=kpss_regression, nlags=kpss_lags)
                            results['KPSS'] = {
                                'Test Statistic': kpss_stat,
                                'p-value': kpss_pval,
                                'Critical Values (1%)': kpss_crit['1%'],
                                'Critical Values (5%)': kpss_crit['5%'],
                                'Critical Values (10%)': kpss_crit['10%'],
                                'Lags': kpss_lags,
                                'Regression Type': kpss_regression,
                                'Breakpoint': 'N/A'
                            }
                        except Exception as e:
                            st.error(f"Error in KPSS test: {str(e)}")
                    
                    # Zivot-Andrews Test
                    if run_za:
                        try:
                            za = ZivotAndrews(ts, lags=max_lags)
                            
                            # Get break date
                            breakpoint_date = ts.index[za.break_idx]
                            break_dates['ZA'] = breakpoint_date
                            
                            results['Zivot-Andrews'] = {
                                'Test Statistic': za.stat,
                                'p-value': za.pvalue,
                                'Critical Values (1%)': za.critical_values['1%'],
                                'Critical Values (5%)': za.critical_values['5%'],
                                'Critical Values (10%)': za.critical_values['10%'],
                                'Lags': za.lags,
                                'Regression Type': za_regression,
                                'Breakpoint': breakpoint_date.strftime('%Y-%m-%d')
                            }
                        except Exception as e:
                            st.error(f"Error in Zivot-Andrews test: {str(e)}")
                    
                    # Lee-Strazicich with 1 break
                    if run_ls_1:
                        try:
                            ls1_result = lee_strazicich_test(ts.values, model=ls_model, breaks=1)
                            
                            # Get break date
                            if ls1_result['breakpoints'][0] is not None:
                                ls1_break = ts.index[ls1_result['breakpoints'][0]]
                                break_dates['LS1'] = ls1_break
                                breakpoint_str = ls1_break.strftime('%Y-%m-%d')
                            else:
                                breakpoint_str = 'Not found'
                            
                            results['Lee-Strazicich (1)'] = {
                                'Test Statistic': ls1_result['stat'],
                                'p-value': ls1_result['pvalue'],
                                'Critical Values (1%)': ls1_result['crit_vals']['1%'],
                                'Critical Values (5%)': ls1_result['crit_vals']['5%'],
                                'Critical Values (10%)': ls1_result['crit_vals']['10%'],
                                'Lags': 'Auto',  # Could be improved with actual lag calculation
                                'Regression Type': ls1_result['model'],
                                'Breakpoint': breakpoint_str
                            }
                        except Exception as e:
                            st.error(f"Error in Lee-Strazicich (1 break) test: {str(e)}")
                    
                    # Lee-Strazicich with 2 breaks
                    if run_ls_2:
                        try:
                            ls2_result = lee_strazicich_test(ts.values, model=ls_model, breaks=2)
                            
                            # Get break dates
                            breakpoint_str = []
                            for i, bp in enumerate(ls2_result['breakpoints']):
                                if bp is not None:
                                    break_date = ts.index[bp]
                                    break_dates[f'LS2_{i+1}'] = break_date
                                    breakpoint_str.append(break_date.strftime('%Y-%m-%d'))
                                else:
                                    breakpoint_str.append('Not found')
                            
                            results['Lee-Strazicich (2)'] = {
                                'Test Statistic': ls2_result['stat'],
                                'p-value': ls2_result['pvalue'],
                                'Critical Values (1%)': ls2_result['crit_vals']['1%'],
                                'Critical Values (5%)': ls2_result['crit_vals']['5%'],
                                'Critical Values (10%)': ls2_result['crit_vals']['10%'],
                                'Lags': 'Auto',
                                'Regression Type': ls2_result['model'],
                                'Breakpoint': ', '.join(breakpoint_str)
                            }
                        except Exception as e:
                            st.error(f"Error in Lee-Strazicich (2 breaks) test: {str(e)}")
                    
                    # Lumsdaine-Papell test
                    if run_lp:
                        try:
                            lp_result = lumsdaine_papell_test(ts.values, model=lp_model, max_lags=max_lags)
                            
                            # Get break dates
                            breakpoint_str = []
                            for i, bp in enumerate(lp_result['breakpoints']):
                                if bp is not None:
                                    break_date = ts.index[bp]
                                    break_dates[f'LP_{i+1}'] = break_date
                                    breakpoint_str.append(break_date.strftime('%Y-%m-%d'))
                                else:
                                    breakpoint_str.append('Not found')
                            
                            results['Lumsdaine-Papell'] = {
                                'Test Statistic': lp_result['stat'],
                                'p-value': lp_result['pvalue'],
                                'Critical Values (1%)': lp_result['crit_vals']['1%'],
                                'Critical Values (5%)': lp_result['crit_vals']['5%'],
                                'Critical Values (10%)': lp_result['crit_vals']['10%'],
                                'Lags': lp_result['lags'],
                                'Regression Type': lp_result['model'],
                                'Breakpoint': ', '.join(breakpoint_str)
                            }
                        except Exception as e:
                            st.error(f"Error in Lumsdaine-Papell test: {str(e)}")
                    
                    # DFGLS Test
                    if run_dfgls:
                        try:
                            # Map regression type
                            trend_bool = adf_regression in ['ct', 'ctt']
                            
                            dfgls = DFGLS(ts, lags=max_lags, trend=trend_bool)
                            results['DF-GLS'] = {
                                'Test Statistic': dfgls.stat,
                                'p-value': dfgls.pvalue,
                                'Critical Values (1%)': dfgls.critical_values['1%'],
                                'Critical Values (5%)': dfgls.critical_values['5%'],
                                'Critical Values (10%)': dfgls.critical_values['10%'],
                                'Lags': dfgls.lags,
                                'Regression Type': 'trend' if trend_bool else 'constant',
                                'Breakpoint': 'N/A'
                            }
                        except Exception as e:
                            st.error(f"Error in DF-GLS test: {str(e)}")
                    
                    # Variance Ratio Test
                    if run_vr:
                        try:
                            vr = VarianceRatio(ts, lags=4)
                            results['Variance Ratio'] = {
                                'Test Statistic': vr.stat,
                                'p-value': vr.pvalue,
                                'Critical Values (1%)': vr.critical_values['1%'],
                                'Critical Values (5%)': vr.critical_values['5%'],
                                'Critical Values (10%)': vr.critical_values['10%'],
                                'Lags': 4,
                                'Regression Type': 'N/A',
                                'Breakpoint': 'N/A'
                            }
                        except Exception as e:
                            st.error(f"Error in Variance Ratio test: {str(e)}")
                
                st.success("✅ Unit root tests completed!")
                
                # Extract key results for summary table
                summary_cols = ['Test Statistic', 'p-value', 'Critical Values (5%)', 'Lags', 'Breakpoint']
                results_df = pd.DataFrame({k: {col: v[col] for col in summary_cols if col in v} 
                                          for k, v in results.items()}).T
                
                # Create detailed results dataframe
                detailed_results = pd.DataFrame(results).T
                
                # Display results in tabs
                tab1, tab2 = st.tabs(["Summary Results", "Detailed Results"])
                
                with tab1:
                    st.subheader("📋 Unit Root Test Summary")
                    
                    # Format summary table
                    st.dataframe(results_df.style.format({
                        'Test Statistic': '{:.4f}',
                        'p-value': '{:.4f}',
                        'Critical Values (5%)': '{:.4f}'
                    }), use_container_width=True)
                    
                    # Quick interpretation
                    st.subheader("🔍 Test Interpretation")
                    
                    # Create columns for test interpretations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Unit Root Test Results")
                        # Create interpretation
                        interpretations = []
                        overall_stationary = 0
                        overall_nonstationary = 0
                        
                        for test, values in results.items():
                            p_value = values['p-value']
                            stat = values['Test Statistic']
                            crit_val = values.get('Critical Values (5%)', None)
                            
                            if test == 'KPSS':
                                # For KPSS, null hypothesis is stationarity
                                if p_value < 0.05:
                                    interpretations.append(f"• {test}: **Non-stationary** (p-value = {p_value:.4f} < 0.05)")
                                    overall_nonstationary += 1
                                else:
                                    interpretations.append(f"• {test}: **Stationary** (p-value = {p_value:.4f} ≥ 0.05)")
                                    overall_stationary += 1
                            else:
                                # For other tests, null hypothesis is non-stationarity
                                if p_value < 0.05:
                                    interpretations.append(f"• {test}: **Stationary** (p-value = {p_value:.4f} < 0.05)")
                                    overall_stationary += 1
                                else:
                                    interpretations.append(f"• {test}: **Non-stationary** (p-value = {p_value:.4f} ≥ 0.05)")
                                    overall_nonstationary += 1
                        
                        # Display interpretations
                        for interp in interpretations:
                            st.markdown(interp)
                    
                    with col2:
                        st.markdown("##### Overall Assessment")
                        
                        # Determine overall consensus
                        if overall_stationary > overall_nonstationary:
                            status_class = "stationary"
                            status_text = "Stationary"
                            description = f"The time series appears to be **stationary** based on {overall_stationary} out of {overall_stationary + overall_nonstationary} tests."
                        elif overall_nonstationary > overall_stationary:
                            status_class = "non-stationary"
                            status_text = "Non-Stationary"
                            description = f"The time series appears to be **non-stationary** based on {overall_nonstationary} out of {overall_stationary + overall_nonstationary} tests."
                        else:
                            status_class = "inconclusive"
                            status_text = "Inconclusive"
                            description = f"The results are **inconclusive** with {overall_stationary} tests indicating stationarity and {overall_nonstationary} tests indicating non-stationarity."
                        
                        # Display status box
                        st.markdown(f"""
                        <div class="status-box {status_class}">
                            <h3 style="margin-top:0;">{status_text}</h3>
                            <p>{description}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Explain implications
                        st.markdown("##### What This Means")
                        if status_class == "stationary":
                            st.markdown("""
                            - The series has constant mean and variance over time
                            - Shocks to the system are temporary
                            - The series can be modeled using ARMA processes
                            - Forecasting and statistical inference are straightforward
                            """)
                        elif status_class == "non-stationary":
                            st.markdown("""
                            - The series may have unit roots, trends, or changing variance
                            - Shocks to the system have permanent effects
                            - Differencing may be required before modeling
                            - Consider ARIMA, SARIMA, or cointegration analysis
                            """)
                        else:
                            st.markdown("""
                            - Different tests provide conflicting evidence
                            - Consider examining the structural breaks identified
                            - Try differencing and testing again
                            - Visual inspection of ACF/PACF plots may help
                            """)
                
                with tab2:
                    st.subheader("📊 Detailed Test Results")
                    st.dataframe(detailed_results.style.format({
                        'Test Statistic': '{:.4f}',
                        'p-value': '{:.4f}',
                        'Critical Values (1%)': '{:.4f}',
                        'Critical Values (5%)': '{:.4f}',
                        'Critical Values (10%)': '{:.4f}'
                    }), use_container_width=True)
                    
                    # Display test descriptions
                    with st.expander("ℹ️ Unit Root Test Descriptions"):
                        st.markdown("""
                        **Augmented Dickey-Fuller (ADF)** - Tests the null hypothesis that a unit root is present. A low p-value rejects the null and suggests stationarity.
                        
                        **Phillips-Perron (PP)** - A non-parametric test that is robust to serial correlation. Like ADF, it tests for unit roots.
                        
                        **KPSS** - Unlike ADF and PP, the null hypothesis is that the series is stationary. A low p-value suggests non-stationarity.
                        
                        **Zivot-Andrews** - Tests for unit root allowing for a single structural break. Particularly useful when structural breaks might lead to false non-rejection of the unit root hypothesis.
                        
                        **Lee-Strazicich** - A Lagrange Multiplier unit root test that allows for one or two structural breaks under both the null and alternative hypotheses.
                        
                        **Lumsdaine-Papell** - An extension of Zivot-Andrews that allows for two structural breaks. Helps identify multiple regime changes.
                        
                        **DF-GLS** - A more powerful variant of the ADF test that uses generalized least squares detrending.
                        
                        **Variance Ratio** - Tests the random walk hypothesis based on the property that the variance of random walk increments is linear in the sampling interval.
                        """)
                
                # Visualizations
                st.subheader("📊 Visual Analysis")
                
                # Create visualization tabs
                viz_tab1, viz_tab2, viz_tab3 = st.tabs(["P-value Heatmap", "Time Series with Breaks", "Stationarity Analysis"])
                
                with viz_tab1:
                    # P-value Heatmap
                    fig1, ax1 = plt.subplots(figsize=(8, len(results)/2 + 1))
                    sns.heatmap(
                        results_df[["p-value"]].astype(float), 
                        annot=True, 
                        cmap='RdYlGn_r', 
                        fmt=".4f", 
                        ax=ax1,
                        vmin=0, 
                        vmax=0.1,
                        cbar_kws={'label': 'p-value'}
                    )
                    ax1.set_title("P-values from Unit Root Tests")
                    plt.tight_layout()
                    st.pyplot(fig1)
                
                with viz_tab2:
                    if break_dates:
                        # Time Series Plot with Breaks
                        fig2, ax2 = plt.subplots(figsize=(10, 5))
                        ax2.plot(ts.index, ts.values, label='Time Series', linewidth=2)
                        
                        # Color map for different breaks
                        colors = ['red', 'blue', 'green', 'purple', 'orange']
                        color_idx = 0
                        
                        # Add breakpoint lines
                        for test, break_date in break_dates.items():
                            color = colors[color_idx % len(colors)]
                            linestyle = '--' if 'ZA' in test or 'LS1' in test else '-.'
                            ax2.axvline(break_date, color=color, linestyle=linestyle, linewidth=2, 
                                        label=f'{test} Break: {break_date.strftime("%Y-%m-%d")}')
                            color_idx += 1
                        
                        ax2.set_title("Time Series with Structural Break Detection")
                        ax2.set_xlabel("Date")
                        ax2.set_ylabel(value_col)
                        ax2.legend(loc='best')
                        ax2.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig2)
                    else:
                        st.warning("No structural breaks detected or structural break tests not selected.")
                
                with viz_tab3:
                    # Create differentiated series
                    ts_diff = ts.diff().dropna()
                    
                    # Create subplot
                    fig3, axs = plt.subplots(3, 1, figsize=(10, 12))
                    
                    # Original series
                    axs[0].plot(ts.index, ts.values, label='Original Series', color='blue')
                    axs[0].set_title("Original Time Series")
                    axs[0].grid(True, alpha=0.3)
                    
                    # Differenced series
                    axs[1].plot(ts_diff.index, ts_diff.values, label='First Differenced Series', color='green')
                    axs[1].set_title("First Differenced Series")
                    axs[1].grid(True, alpha=0.3)
                    
                    # ACF/PACF plots to check stationarity
                    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                    
                    plot_acf(ts, lags=min(30, len(ts)//4), ax=axs[2], alpha=0.05)
                    axs[2].set_title("Autocorrelation Function (ACF)")
                    
                    plt.tight_layout()
                    st.pyplot(fig3)
                    
                    # Create another plot for PACF
                    fig4, ax4 = plt.subplots(figsize=(10, 4))
                    plot_pacf(ts, lags=min(30, len(ts)//4), ax=ax4, alpha=0.05)
                    ax4.set_title("Partial Autocorrelation Function (PACF)")
                    plt.tight_layout()
                    st.pyplot(fig4)
                    
                    # Add explanatory text
                    st.markdown("""
                    **Stationarity Analysis:**
                    
                    The charts above show the original time series, its first difference, and the ACF/PACF plots.
                    
                    **For a stationary series:**
                    
                    1. The ACF should decay quickly to zero
                    2. The PACF should have a sharp cutoff
                    3. The differenced series should fluctuate around a constant mean
                    
                    **For a non-stationary series:**
                    
                    1. The ACF decreases slowly
                    2. There may be significant autocorrelation at high lags
                    3. The original series shows trends or changing variance
                    """)
                
                # Download options
                st.subheader("📥 Download Results")
                
                # Create Excel file with multiple sheets
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    # Summary sheet
                    results_df.to_excel(writer, sheet_name='Summary Results')
                    workbook = writer.book
                    worksheet = writer.sheets['Summary Results']
                    
                    # Add formatting
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#D7E4BC',
                        'border': 1
                    })
                    
                    for col_num, value in enumerate(results_df.columns.values):
                        worksheet.write(0, col_num + 1, value, header_format)
                    
                    # Detailed results sheet
                    detailed_results.to_excel(writer, sheet_name='Detailed Results')
                    
                    # Original data sheet
                    ts.to_frame().to_excel(writer, sheet_name='Original Data')
                    
                    # Add metadata sheet
                    meta_data = {
                        'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'File Analyzed': uploaded_file.name,
                        'Date Column': date_col,
                        'Value Column': value_col,
                        'Number of Observations': len(ts),
                        'Start Date': ts.index.min().strftime('%Y-%m-%d'),
                        'End Date': ts.index.max().strftime('%Y-%m-%d'),
                        'Tests Run': ', '.join(results.keys()),
                        'Overall Assessment': status_text if 'status_text' in locals() else 'Not determined'
                    }
                    pd.DataFrame(list(meta_data.items()), columns=['Metadata', 'Value']).to_excel(writer, sheet_name='Metadata')
                    
                    # Add interpretation sheet
                    if 'interpretations' in locals():
                        interp_data = {
                            'Test': [i.split(':')[0].replace('• ', '') for i in interpretations],
                            'Result': [i.split(':')[1].strip() for i in interpretations]
                        }
                        pd.DataFrame(interp_data).to_excel(writer, sheet_name='Interpretation')
                
                excel_buffer.seek(0)
                
                # Create download buttons in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        "📊 Download Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"unit_root_results_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel",
                        use_container_width=True
                    )
                
                # Create PDF for plots
                with col2:
                    # Create a comprehensive PDF with all plots
                    pdf_buffer = io.BytesIO()
                    
                    # Save multiple figures to a single PDF
                    from matplotlib.backends.backend_pdf import PdfPages
                    
                    with PdfPages(pdf_buffer) as pdf:
                        # Add all figures
                        if 'fig' in locals(): pdf.savefig(fig)
                        if 'fig1' in locals(): pdf.savefig(fig1)
                        if break_dates and 'fig2' in locals(): pdf.savefig(fig2)
                        if 'fig3' in locals(): pdf.savefig(fig3)
                        if 'fig4' in locals(): pdf.savefig(fig4)
                        
                        # Create a metadata page
                        plt.figure(figsize=(8.5, 11))
                        plt.axis('off')
                        plt.text(0.5, 0.95, "Unit Root Test Analysis Report", ha='center', fontsize=16, fontweight='bold')
                        plt.text(0.5, 0.9, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha='center')
                        plt.text(0.5, 0.85, f"File: {uploaded_file.name}", ha='center')
                        
                        y_pos = 0.8
                        plt.text(0.1, y_pos, "Test Results Summary:", fontweight='bold')
                        y_pos -= 0.05
                        
                        for interp in interpretations:
                            plt.text(0.1, y_pos, interp)
                            y_pos -= 0.05
                        
                        if 'status_text' in locals():
                            y_pos -= 0.05
                            plt.text(0.1, y_pos, f"Overall Assessment: {status_text}", fontweight='bold')
                        
                        pdf.savefig()
                        plt.close()
                    
                    pdf_buffer.seek(0)
                    
                    st.download_button(
                        "📈 Download All Plots (PDF)",
                        data=pdf_buffer.getvalue(),
                        file_name=f"unit_root_plots_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.error("Please ensure you've selected the correct date and value columns.")
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.error("Please ensure your file is in the correct format (CSV or Excel).")

else:
    # Display sample data option
    st.info("Please upload a time series file (CSV or Excel) or use the sample data below.")
    
    if st.button("Load Sample Data"):
        # Generate sample time series
        start_date = pd.Timestamp('2010-01-01')
        periods = 120
        dates = pd.date_range(start=start_date, periods=periods, freq='M')
        
        # Create trend with structural break
        trend = np.arange(periods) * 0.1
        trend[60:] = trend[60:] + 5  # Add structural break
        
        # Add seasonality, noise and random walk component
        seasonality = 2 * np.sin(np.arange(periods) * 2 * np.pi / 12)
        noise = np.random.normal(0, 1, periods)
        random_walk = np.cumsum(np.random.normal(0, 0.5, periods))
        
        values = trend + seasonality + noise + random_walk
        
        # Create DataFrame
        sample_df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        # Cache the sample data
        st.session_state['sample_data'] = sample_df
        
        # Display success message
        st.success("Sample data loaded successfully!")
        
        # Show preview
        st.dataframe(sample_df.head(10))
        
        # Save to CSV for download
        csv_buffer = io.BytesIO()
        sample_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        st.download_button(
            "Download Sample Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name="sample_time_series.csv",
            mime="text/csv"
        )
        
    # Instructions
    with st.expander("📚 Instructions & Information"):
        st.markdown("""
        ### How to Use This App
        
        1. **Upload a time series file** in CSV or Excel format
        2. **Select the date and value columns** from your data
        3. **Choose which unit root tests** to run in the sidebar
        4. **Configure test parameters** in the sidebar
        5. **Run the analysis** and interpret the results
        6. **Download the results** as an Excel report or PDF plots
        
        ### About Unit Root Tests
        
        Unit root tests check whether a time series is stationary or not:
        
        - **Augmented Dickey-Fuller (ADF)**: Tests the null hypothesis that a unit root is present
        - **Phillips-Perron (PP)**: Non-parametric test that is robust to heteroskedasticity
        - **KPSS**: Tests the null hypothesis that the series is stationary
        - **Zivot-Andrews**: Tests for unit root with a single structural break
        - **Lee-Strazicich**: Tests for unit roots with one or two structural breaks using LM test
        - **Lumsdaine-Papell**: Tests for unit roots with two structural breaks
        - **DF-GLS**: A more powerful variant of the ADF test
        - **Variance Ratio**: Tests the random walk hypothesis
        
        For ADF, PP, DF-GLS, Zivot-Andrews, Lee-Strazicich, and Lumsdaine-Papell tests, a p-value < 0.05 suggests stationarity.  
        For KPSS test, a p-value < 0.05 suggests non-stationarity.
        """)
        
    # Footer
    st.markdown("---")
    st.markdown(
        "Created with ❤️ | "
        "© 2025 Advanced Unit Root Testing Application | "
        "Version 2.1.0"
    )
stat'],
                                'p-value': lp_result['
