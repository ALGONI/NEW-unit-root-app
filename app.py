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
    
    for lag in range(min(12, max_lags)):
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
    step = max(1, (max_idx - min_idx) // 20)  # Sample 20 potential points
    
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
            model_fit = sm.OLS(dy[best_lag:], X).fit()
            t_stat = model_fit.tvalues[1]  # t-stat on the lagged level
            
            if abs(t_stat) < best_stat:
                best_stat = abs(t_stat)
                best_breaks = (tb1, tb2)
    
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

# Regression type mapping
regression_types = {
    'adf_pp': {
        'c': 'Constant',
        'ct': 'Constant & Trend',
        'n': 'No Constant or Trend',
        'ctt': 'Constant, Linear & Quadratic Trend'
    },
    'kpss': {
        'c': 'Constant',
        'ct': 'Constant & Trend'
    },
    'za': {
        'c': 'Break in Constant',
        't': 'Break in Trend',
        'both': 'Break in Constant & Trend'
    }
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
        
        # ADF and PP parameters
        adf_regression = st.sidebar.selectbox(
            "ADF/PP Regression Type", 
            options=["c", "ct", "n", "ctt"],
            format_func=lambda x: regression_types['adf_pp'].get(x, x),
            index=1
        )
        
        # KPSS parameters
        kpss_regression = st.sidebar.selectbox(
            "KPSS Regression Type", 
            options=["c", "ct"],
            format_func=lambda x: regression_types['kpss'].get(x, x),
            index=0
        )
        
        # ZA parameters
        za_regression = st.sidebar.selectbox(
            "Zivot-Andrews Model", 
            options=["c", "t", "both"],
            format_func=lambda x: regression_types['za'].get(x, x),
            index=2
        )
        
        # LS parameters
        ls_model = st.sidebar.selectbox(
            "Lee-Strazicich Model",
            options=["c", "ct"],
            format_func=lambda x: {"c": "Crash Model (Level Shift)", "ct": "Trend Shift Model"}.get(x, x),
            index=0
        )
        
        # LP parameters
        lp_model = st.sidebar.selectbox(
            "Lumsdaine-Papell Model",
            options=["c", "ct"],
            format_func=lambda x: {"c": "Crash Model (Level Shift)", "ct": "Trend Shift Model"}.get(x, x),
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
                            # Convert regression type for PP test
                            pp_trend_map = {'c': 'c', 'ct': 'ct', 'n': 'n', 'ctt': 'ct'}
                            pp_trend = pp_trend_map.get(adf_regression, 'c')
                            
                            pp = PhillipsPerron(ts, trend=pp_trend, lags=max_lags)
                            results['Phillips-Perron'] = {
                                'Test Statistic': pp.stat,
                                'p-value': pp.pvalue,
                                'Critical Values (1%)': pp.critical_values['1%'],
                                'Critical Values (5%)': pp.critical_values['5%'],
                                'Critical Values (10%)': pp.critical_values['10%'],
                                'Lags': pp.lags,
                                'Regression Type': pp_trend,
                                'Breakpoint': 'N/A'
                            }
                        except Exception as e:
                            st.error(f"Error in Phillips-Perron test: {str(e)}")
                    
                    # KPSS Test
                    if run_kpss:
                        try:
                            kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(ts, regression=kpss_regression, nlags="auto")
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
                            # Convert regression type for ZA test
                            za_trend_map = {'c': 'c', 't': 't', 'both': 'both'}
                            za_trend = za_trend_map.get(za_regression, 'both')
                            
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
                                'Regression Type': za_trend,
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
                                'Test Statistic
