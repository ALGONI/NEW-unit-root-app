import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib as mpl
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import skew, kurtosis
import warnings

# Import arch and check version
try:
    import arch
    from arch.unitroot import PhillipsPerron, DFGLS, VarianceRatio
except:
    st.error("Please install the 'arch' package: pip install arch")
    st.stop()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9

# Page configuration
st.set_page_config(
    page_title="Unit Root Test App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2C3E50; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #3498DB;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("📊 Test Configuration")
st.sidebar.subheader("Select Tests")
test_options = {
    'ADF': st.sidebar.checkbox("Augmented Dickey-Fuller (ADF)", value=True),
    'PP': st.sidebar.checkbox("Phillips-Perron (PP)", value=True),
    'KPSS': st.sidebar.checkbox("KPSS", value=True),
    'DFGLS': st.sidebar.checkbox("DFGLS", value=True),
    'VR': st.sidebar.checkbox("Variance Ratio", value=False)
}

# Data differencing option
st.sidebar.subheader("Differencing Level")
diff_options = ["Level (No Differencing)", "First Difference", "Second Difference"]
diff_selection = st.sidebar.radio("Select Differencing Level", diff_options)

# Test parameters
st.sidebar.subheader("Test Parameters")

# Define lag selection criteria options
lag_criteria_options = {
    "Fixed": "Fixed Value",
    "AIC": "Akaike Information Criterion (AIC)",
    "BIC": "Bayesian Information Criterion (BIC)",
    "t-stat": "t-statistic significance",
    "None": "Let test decide (default)"
}

# Lag selection method
lag_selection_method = st.sidebar.selectbox(
    "Lag Selection Method",
    options=list(lag_criteria_options.keys()),
    format_func=lambda x: lag_criteria_options[x],
    index=0
)

# Only show max lag parameter if using Fixed or other methods (except None)
if lag_selection_method != "None":
    max_lags = st.sidebar.number_input(
        "Maximum Lags" if lag_selection_method != "Fixed" else "Fixed Lags",
        min_value=0,
        max_value=30,
        value=4
    )
else:
    max_lags = None

# Define regression type mapping (for display)
regression_type_display = {
    "c": "Constant Only",
    "ct": "Constant & Trend",
    "n": "No Constant or Trend",
    "ctt": "Constant, Linear & Quadratic Trend"
}

# Standardized regression type options for most tests
regression_options = ["c", "ct", "n", "ctt"]
adf_regression = st.sidebar.selectbox(
    "ADF Regression Type",
    options=regression_options,
    format_func=lambda x: regression_type_display.get(x, x),
    index=1  # Default to constant & trend
)

# For KPSS: Only c and ct are supported
kpss_regression = st.sidebar.selectbox(
    "KPSS Regression Type",
    options=["c", "ct"],
    format_func=lambda x: regression_type_display.get(x, x),
    index=0  # Default to constant only
)

# Phillips-Perron: Same options as ADF
pp_regression = st.sidebar.selectbox(
    "Phillips-Perron Regression Type",
    options=regression_options,
    format_func=lambda x: regression_type_display.get(x, x),
    index=1  # Default to constant & trend
)

# For DFGLS: Only c and ct are supported
dfgls_regression = st.sidebar.selectbox(
    "DFGLS Regression Type",
    options=["c", "ct"],
    format_func=lambda x: regression_type_display.get(x, x),
    index=1  # Default to constant & trend
)

# Main content
st.title("📊 Advanced Unit Root Test Application")
st.markdown("Analyze time series stationarity with multiple test methods")

# File upload
uploaded_file = st.file_uploader("📁 Upload CSV/Excel", type=["csv", "xlsx"])

@st.cache_data
def load_data(file):
    """Load and cache file data"""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file, parse_dates=True)
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"File loading failed: {str(e)}")
        return None

@st.cache_data
def parse_dates(series, sample_size=10):
    """Robust date parsing for yearly, monthly, daily, and quarterly data"""
    formats = [
        None, '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y%m%d',  # Daily
        '%Y-%m', '%Y/%m', '%Y%m', '%YM%m',  # Monthly
        '%YQ%q', '%Y-Q%q', '%Y',  # Quarterly and Yearly
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'  # Daily with time
    ]
    
    for fmt in formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors='coerce')
            if parsed.notna().sum() > len(series) * 0.8:
                return parsed
        except:
            continue
    
    # Custom parsing for Year-Month (2013M01)
    try:
        def parse_year_month(date_str):
            if isinstance(date_str, str) and 'M' in date_str:
                year, month = date_str.split('M')
                return pd.Timestamp(year=int(year), month=int(month), day=1)
            return pd.NaT
        parsed = series.apply(parse_year_month)
        if parsed.notna().sum() > len(series) * 0.8:
            return parsed
    except:
        pass
    
    # Try pandas default parsing as fallback
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.notna().sum() > len(series) * 0.5:
        return parsed
    
    st.warning("Could not parse all dates. Some data may be excluded.")
    return parsed

def apply_differencing(series, diff_level):
    """Apply differencing based on the selected level"""
    if diff_level == "First Difference":
        return series.diff().dropna()
    elif diff_level == "Second Difference":
        return series.diff().diff().dropna()
    else:  # Level (No Differencing)
        return series

def get_differencing_suffix(diff_level):
    """Get a suffix for display based on differencing level"""
    if diff_level == "First Difference":
        return " (Δ)"
    elif diff_level == "Second Difference":
        return " (Δ²)"
    else:
        return ""

if uploaded_file:
    with st.spinner("Loading data..."):
        df = load_data(uploaded_file)
    
    if df is not None:
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Data preview
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(10))
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Column selection
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("📅 Date Column", options=df.columns)
        
        # Modified to allow selecting multiple value columns
        with col2:
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != date_col]
            value_cols = st.multiselect("📈 Value Column(s)", options=numeric_cols, default=numeric_cols[0] if numeric_cols else None)
        
        if not value_cols:
            st.warning("Please select at least one value column for analysis.")
        else:
            # Parse dates
            try:
                df[date_col] = parse_dates(df[date_col])
                df = df[[date_col] + value_cols].dropna()
                df.set_index(date_col, inplace=True)
                
                # Select which variable to analyze (if multiple selected)
                if len(value_cols) > 1:
                    selected_var = st.selectbox("Select Variable for Unit Root Test", options=value_cols)
                else:
                    selected_var = value_cols[0]
                
                # Apply differencing based on selection
                diff_suffix = get_differencing_suffix(diff_selection)
                display_title = f"{selected_var}{diff_suffix}"
                
                # Compute descriptive statistics for all selected columns
                desc_stats_all = {}
                
                for col in value_cols:
                    ts = df[col]
                    
                    # Apply differencing based on selection
                    if diff_selection == "First Difference":
                        ts_diff = ts.diff().dropna()
                        suffix = " (Δ)"
                    elif diff_selection == "Second Difference":
                        ts_diff = ts.diff().diff().dropna()
                        suffix = " (Δ²)"
                    else:  # Level (No Differencing)
                        ts_diff = ts
                        suffix = ""
                    
                    # Calculate stats for the (possibly differenced) series
                    desc_stats_all[col + suffix] = {
                        'Count': ts_diff.count(),
                        'Mean': ts_diff.mean(),
                        'Std': ts_diff.std(),
                        'Min': ts_diff.min(),
                        '25%': ts_diff.quantile(0.25),
                        '50%': ts_diff.quantile(0.50),
                        '75%': ts_diff.quantile(0.75),
                        'Max': ts_diff.max(),
                        'Skewness': skew(ts_diff),
                        'Kurtosis': kurtosis(ts_diff, fisher=True)
                    }
                
                # Create a DataFrame from the descriptive statistics
                desc_stats_df = pd.DataFrame(desc_stats_all)
                
                # Display descriptive statistics for all variables
                with st.expander("📊 Descriptive Statistics", expanded=True):
                    st.dataframe(desc_stats_df.style.format("{:.4f}"))
                    
                    # Download descriptive statistics
                    stats_buffer = io.BytesIO()
                    desc_stats_df.to_csv(stats_buffer)
                    st.download_button(
                        "📊 Download Descriptive Statistics",
                        stats_buffer.getvalue(),
                        f"descriptive_stats_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                # Select the TS for the selected variable and apply differencing
                ts = apply_differencing(df[selected_var], diff_selection)
                
                if st.sidebar.button("▶️ Run Analysis", use_container_width=True):
                    with st.spinner("Processing data..."):
                        try:
                            # Validate data
                            if len(ts) < 20:
                                st.error(f"Insufficient data points ({len(ts)}). Need at least 20 observations for reliable results.")
                                st.stop()
                            if ts.isna().any():
                                st.error("Time series contains missing values. Please clean the data.")
                                st.stop()
                            if not np.issubdtype(ts.dtype, np.number):
                                st.error("Value column must contain numeric data.")
                                st.stop()
                            
                            # Determine appropriate lag selection parameters based on method
                            if lag_selection_method == "None":
                                autolag = 'AIC'  # Default method
                                max_lag = None
                            elif lag_selection_method == "Fixed":
                                autolag = None
                                max_lag = max_lags
                            else:
                                autolag = lag_selection_method.lower()
                                max_lag = max_lags
                            
                            # Visualizations
                            st.subheader(f"📉 Time Series Visualization: {display_title}")
                            tab1, tab2 = st.tabs(["Line Chart", "Interactive Chart"])
                            
                            with tab1:
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.plot(ts.index, ts.values, linewidth=2)
                                ax.set_title(f"Time Series: {display_title}")
                                ax.grid(True, alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            with tab2:
                                st.line_chart(ts, use_container_width=True)
                            
                            # Run tests
                            results = {}
                            with st.spinner("Running unit root tests..."):
                                if test_options['ADF']:
                                    adf_result = adfuller(
                                        ts, 
                                        regression=adf_regression, 
                                        maxlag=max_lag, 
                                        autolag=autolag if autolag != "None" else None
                                    )
                                    
                                    # Get the actual lag used
                                    actual_lag = adf_result[2] if autolag else max_lag
                                    
                                    results['ADF'] = {
                                        'Test Statistic': adf_result[0],
                                        'p-value': adf_result[1],
                                        'Critical Values (5%)': adf_result[4]['5%'],
                                        'Lags': actual_lag,
                                        'Lag Method': lag_selection_method,
                                        'Regression Type': regression_type_display.get(adf_regression, adf_regression)
                                    }
                                
                                if test_options['PP']:
                                    pp = PhillipsPerron(
                                        ts, 
                                        trend=pp_regression, 
                                        lags=max_lag if lag_selection_method == "Fixed" else None
                                    )
                                    results['PP'] = {
                                        'Test Statistic': pp.stat,
                                        'p-value': pp.pvalue,
                                        'Critical Values (5%)': pp.critical_values['5%'],
                                        'Lags': pp.lags,
                                        'Lag Method': "Fixed" if lag_selection_method == "Fixed" else "Newey-West",
                                        'Regression Type': regression_type_display.get(pp_regression, pp_regression)
                                    }
                                
                                if test_options['KPSS']:
                                    # For KPSS, adapt lag method
                                    if lag_selection_method == "Fixed":
                                        kpss_nlags = max_lags
                                    else:
                                        kpss_nlags = "auto"  # KPSS uses Newey-West for automatic lag selection
                                        
                                    kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(
                                        ts, 
                                        regression=kpss_regression, 
                                        nlags=kpss_nlags
                                    )
                                    results['KPSS'] = {
                                        'Test Statistic': kpss_stat,
                                        'p-value': kpss_pval,
                                        'Critical Values (5%)': kpss_crit['5%'],
                                        'Lags': kpss_lags,
                                        'Lag Method': "Fixed" if lag_selection_method == "Fixed" else "Newey-West",
                                        'Regression Type': regression_type_display.get(kpss_regression, kpss_regression)
                                    }
                                
                                if test_options['DFGLS']:
                                    try:
                                        # For DFGLS, adapt lag selection method
                                        if lag_selection_method == "Fixed":
                                            dfgls_method = None  # Use fixed lags
                                            dfgls_max_lags = max_lags
                                        else:
                                            # Map lag selection method to valid DFGLS options
                                            method_mapping = {
                                                "AIC": "aic",
                                                "BIC": "bic",
                                                "t-stat": "t-stat",
                                                "None": "aic"  # Default to AIC if None is selected
                                            }
                                            dfgls_method = method_mapping.get(lag_selection_method, "aic")  # Default to 'aic'
                                            dfgls_max_lags = max_lags if max_lags is not None else 12
                                        
                                        # Ensure max_lags is valid
                                        if dfgls_max_lags is not None and (dfgls_max_lags < 0 or dfgls_max_lags >= len(ts) // 2):
                                            st.warning(f"Invalid max_lags ({dfgls_max_lags}) for DFGLS. Using default value (12).")
                                            dfgls_max_lags = 12
                                        
                                        # Run DFGLS test
                                        dfgls = DFGLS(
                                            ts,
                                            trend=dfgls_regression,
                                            max_lags=dfgls_max_lags,
                                            method=dfgls_method
                                        )
                                        results['DFGLS'] = {
                                            'Test Statistic': dfgls.stat,
                                            'p-value': dfgls.pvalue,
                                            'Critical Values (5%)': dfgls.critical_values['5%'],
                                            'Lags': dfgls.lags,
                                            'Lag Method': "Fixed" if lag_selection_method == "Fixed" else lag_selection_method,
                                            'Regression Type': regression_type_display.get(dfgls_regression, dfgls_regression)
                                        }
                                    except Exception as e:
                                        st.warning(f"DFGLS test failed: {str(e)}. Possible causes: incompatible parameters or insufficient data.")
                                        results['DFGLS'] = {
                                            'Test Statistic': None,
                                            'p-value': None,
                                            'Critical Values (5%)': None,
                                            'Lags': None,
                                            'Lag Method': lag_selection_method,
                                            'Regression Type': dfgls_regression
                                        }
                                
                                if test_options['VR']:
                                    # Variance Ratio test
                                    vr_lags = max_lags if lag_selection_method == "Fixed" else None
                                    vr = VarianceRatio(ts, lags=vr_lags)
                                    results['VR'] = {
                                        'Test Statistic': vr.stat,
                                        'p-value': vr.pvalue,
                                        'Critical Values (5%)': vr.critical_values['5%'],
                                        'Lags': vr.lags,
                                        'Lag Method': "Fixed" if lag_selection_method == "Fixed" else "Default",
                                        'Regression Type': 'N/A (Not Applicable)'
                                    }
                            
                            if not results:
                                st.warning("No tests selected. Please select at least one test.")
                                st.stop()
                            
                            # Display results
                            results_df = pd.DataFrame(results).T
                            st.subheader(f"📋 Test Results: {display_title}")
                            st.dataframe(results_df.style.format({
                                'Test Statistic': lambda x: '{:.4f}'.format(x) if isinstance(x, (int, float)) else x,
                                'p-value': lambda x: '{:.4f}'.format(x) if isinstance(x, (int, float)) else x,
                                'Critical Values (5%)': lambda x: '{:.4f}'.format(x) if isinstance(x, (int, float)) else x,
                                'Lags': lambda x: '{:.0f}'.format(x) if isinstance(x, (int, float)) else x
                            }))
                            
                            # Interpretation
                            st.subheader("🔍 Interpretation")
                            st.markdown(f"**{selected_var}** {diff_suffix}")
                            for test, values in results.items():
                                p_value = values['p-value']
                                if not isinstance(p_value, (int, float)):
                                    continue
                                interpretation = "Stationary" if (test == 'KPSS' and p_value >= 0.05) or \
                                    (test != 'KPSS' and p_value < 0.05) else "Non-stationary"
                                st.markdown(f"• {test}: {interpretation} (p-value = {p_value:.4f})")
                            
                            # Visualizations
                            st.subheader("📊 Visual Analysis")
                            
                            # P-value heatmap
                            p_value_df = results_df[results_df['p-value'].apply(lambda x: isinstance(x, (int, float)))]
                            if not p_value_df.empty:
                                fig1, ax1 = plt.subplots(figsize=(8, len(p_value_df)/2))
                                sns.heatmap(
                                    p_value_df[["p-value"]].astype(float),
                                    annot=True,
                                    cmap='RdYlGn_r',
                                    fmt=".4f",
                                    vmin=0,
                                    vmax=0.1
                                )
                                plt.title(f"P-values: {display_title}")
                                st.pyplot(fig1)
                            else:
                                st.info("No valid p-values for heatmap.")
                            
                            # Download results
                            st.subheader("📥 Download Results")

                            # CSV download option (doesn't require xlsxwriter)
                            csv_buffer = io.BytesIO()
                            results_df.to_csv(csv_buffer)
                            st.download_button(
                                "📊 Download Results CSV",
                                csv_buffer.getvalue(),
                                f"unit_root_results_{selected_var}_{diff_selection.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )

                            # Also offer time series data download
                            data_buffer = io.BytesIO()
                            ts.to_frame().to_csv(data_buffer)
                            st.download_button(
                                "📈 Download Time Series Data",
                                data_buffer.getvalue(),
                                f"time_series_{selected_var}_{diff_selection.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                            st.info("Please check your data format and column selections.")
            
            except Exception as e:
                st.warning(f"Could not compute descriptive statistics: {str(e)}")

else:
    st.info("Upload a time series file or try sample data.")
    
    if st.button("Load Sample Data"):
        dates = pd.date_range('2010-01-01', periods=120, freq='M')
        trend = np.arange(120) * 0.1
        trend[60:] += 5
        values1 = trend + 2 * np.sin(np.arange(120) * 2 * np.pi / 12) + \
                 np.random.normal(0, 1, 120) + np.cumsum(np.random.normal(0, 0.5, 120))
        values2 = values1 * 0.5 + np.random.normal(0, 2, 120) + np.arange(120) * 0.2
        
        sample_df = pd.DataFrame({
            'date': dates, 
            'variable1': values1, 
            'variable2': values2
        })
        st.session_state['sample_data'] = sample_df
        st.dataframe(sample_df.head(10))
        
        csv_buffer = io.BytesIO()
        sample_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download Sample Data",
            csv_buffer.getvalue(),
            "sample_time_series.csv",
            mime="text/csv"
        )

# Instructions
with st.expander("📚 Instructions"):
    st.markdown("""
    1. Upload a CSV/Excel file with time series data
    2. Select date and value column(s)
    3. Choose the differencing level (Level, First Difference, Second Difference)
    4. Select tests and lag selection methods
    5. Run analysis and download results
    
    **Differencing Levels**:
    - **Level (No Differencing)**: Original data without transformation
    - **First Difference**: Changes between consecutive observations (Δyt = yt - yt-1)
    - **Second Difference**: Differences of differences (Δ²yt = Δyt - Δyt-1)
    
    **Lag Selection Methods**:
    - **Fixed Value**: Uses the exact number of lags you specify
    - **AIC**: Akaike Information Criterion - minimizes information loss
    - **BIC**: Bayesian Information Criterion - favors simpler models (fewer lags)
    - **t-stat**: Selects lags based on significance of the t-statistics
    - **Let test decide**: Uses the default method for each test
    
    **Supported Date Formats**:
    - Daily: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY
    - Monthly: YYYY-MM, YYYYMM, YYYY-MM, 2013M01
    - Quarterly: YYYYQ1, YYYY-Q1
    - Yearly: YYYY
    
    **DFGLS Test Notes**:
    - Requires at least 20 observations for reliable results.
    - Supported lag selection methods: AIC, BIC, t-stat, or Fixed.
    - Only 'Constant' (c) or 'Constant & Trend' (ct) regression types are supported.
    
    **Dependencies**:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    statsmodels
    arch
    scipy
    ```
    """)

st.markdown("© 2025 Unit Root Test App | v3.2.0")
