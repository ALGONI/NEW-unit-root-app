import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib as mpl
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

# Import arch and check version
try:
    import arch
    arch_version = tuple(int(x) for x in arch.__version__.split('.'))
    ZA_REGRESSION_SUPPORTED = arch_version >= (5, 0, 0)
except:
    ZA_REGRESSION_SUPPORTED = False
    arch_version = (0, 0, 0)
    
# Then import unit root tests from arch
from arch.unitroot import PhillipsPerron, ZivotAndrews, DFGLS, VarianceRatio

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
    'ZA': st.sidebar.checkbox("Zivot-Andrews", value=True),
    'DFGLS': st.sidebar.checkbox("DFGLS", value=True),
    'VR': st.sidebar.checkbox("Variance Ratio", value=False)
}

# Test parameters
st.sidebar.subheader("Test Parameters")
lags = st.sidebar.number_input("Number of Lags for All Tests", min_value=1, max_value=20, value=4, step=1)

# Define regression type mapping (for display)
regression_type_display = {
    "c": "Constant Only",
    "ct": "Constant & Trend",
    "n": "No Constant or Trend",
    "ctt": "Constant, Linear & Quadratic Trend"
}

# Display current arch version
st.sidebar.info(f"Current arch version: {'.'.join(str(x) for x in arch_version)}")
if not ZA_REGRESSION_SUPPORTED:
    st.sidebar.warning("Zivot-Andrews break type selection requires 'arch' version 5.0.0 or later. Using default model. Install with 'pip install arch>=5.0.0'.")

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

# For Zivot-Andrews: Only if supported version
if ZA_REGRESSION_SUPPORTED:
    za_regression = st.sidebar.selectbox(
        "Zivot-Andrews Break Type",
        options=["c", "t", "ct"],
        format_func=lambda x: {
            "c": "Break in Constant", 
            "t": "Break in Trend", 
            "ct": "Break in Constant & Trend"
        }.get(x, x),
        index=2  # Default to breaks in both constant & trend
    )
else:
    za_regression = "ct"  # Default value for older versions
    st.sidebar.info("Zivot-Andrews will use default break model (Constant & Trend)")

# Main content
st.title("📊 Advanced Unit Root Test Application")
st.markdown("Analyze time series stationarity with structural break detection")

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
        with col2:
            value_col = st.selectbox("📈 Value Column", options=[col for col in df.columns if col != date_col])
        
        if st.sidebar.button("▶️ Run Analysis", use_container_width=True):
            with st.spinner("Processing data..."):
                try:
                    # Parse dates
                    df[date_col] = parse_dates(df[date_col])
                    df = df[[date_col, value_col]].dropna()
                    df.set_index(date_col, inplace=True)
                    ts = df[value_col]
                    
                    if len(ts) < lags + 2:
                        st.error(f"Insufficient data points ({len(ts)}). Need at least {lags + 2} observations.")
                        st.stop()
                    
                    # Visualizations
                    st.subheader("📉 Time Series Visualization")
                    tab1, tab2 = st.tabs(["Line Chart", "Interactive Chart"])
                    
                    with tab1:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(ts.index, ts.values, linewidth=2)
                        ax.set_title(f"Time Series: {value_col}")
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with tab2:
                        st.line_chart(ts, use_container_width=True)
                    
                    # Run tests
                    results = {}
                    breakpoints = []
                    with st.spinner("Running unit root tests..."):
                        if test_options['ADF']:
                            adf_result = adfuller(ts, regression=adf_regression, maxlag=lags, autolag=None)
                            results['ADF'] = {
                                'Test Statistic': adf_result[0],
                                'p-value': adf_result[1],
                                'Critical Values (5%)': adf_result[4]['5%'],
                                'Lags': lags,
                                'Regression Type': regression_type_display.get(adf_regression, adf_regression),
                                'Breakpoint': 'N/A'
                            }
                        
                        if test_options['PP']:
                            pp = PhillipsPerron(ts, trend=pp_regression, lags=lags)
                            results['PP'] = {
                                'Test Statistic': pp.stat,
                                'p-value': pp.pvalue,
                                'Critical Values (5%)': pp.critical_values['5%'],
                                'Lags': lags,
                                'Regression Type': regression_type_display.get(pp_regression, pp_regression),
                                'Breakpoint': 'N/A'
                            }
                        
                        if test_options['KPSS']:
                            kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(ts, regression=kpss_regression, nlags=lags)
                            results['KPSS'] = {
                                'Test Statistic': kpss_stat,
                                'p-value': kpss_pval,
                                'Critical Values (5%)': kpss_crit['5%'],
                                'Lags': lags,
                                'Regression Type': regression_type_display.get(kpss_regression, kpss_regression),
                                'Breakpoint': 'N/A'
                            }
                        
                        if test_options['ZA']:
                            try:
                                # Apply correct implementation based on version
                                if ZA_REGRESSION_SUPPORTED:
                                    # New API with regression parameter (version 5.0.0+)
                                    za = ZivotAndrews(ts, regression=za_regression, lags=lags)
                                    regression_type = za_regression
                                else:
                                    # Old API without regression parameter (version < 5.0.0)
                                    za = ZivotAndrews(ts, lags=lags)
                                    regression_type = 'Default (Constant & Trend Break)'
                                
                                # Handle breakpoint detection for different versions
                                try:
                                    # New versions have break_idx attribute
                                    break_idx = za.break_idx
                                except AttributeError:
                                    # Older versions - need to compute from stats
                                    if hasattr(za, 'stats') and len(za.stats) > 0:
                                        break_idx = np.argmin(za.stats)
                                        # Adjust for trimming (standard 15% trim in ZA)
                                        trim = int(len(ts) * 0.15)
                                        break_idx += trim  # Adjust index to account for trimming
                                        if break_idx >= len(ts):  # Safety check
                                            break_idx = len(ts) - 1
                                    else:
                                        break_idx = None
                                
                                # Get the date corresponding to the breakpoint
                                breakpoint_date = ts.index[break_idx] if break_idx is not None and break_idx < len(ts) else None
                                
                                # Store results
                                za_display_type = {
                                    "c": "Break in Constant", 
                                    "t": "Break in Trend", 
                                    "ct": "Break in Constant & Trend"
                                }.get(regression_type, regression_type)
                                
                                results['ZA'] = {
                                    'Test Statistic': za.stat,
                                    'p-value': za.pvalue,
                                    'Critical Values (5%)': za.critical_values['5%'],
                                    'Lags': lags,
                                    'Regression Type': za_display_type,
                                    'Breakpoint': breakpoint_date.strftime('%Y-%m-%d') if breakpoint_date else 'N/A'
                                }
                                
                                # Add breakpoint to the list for visualization
                                if breakpoint_date:
                                    breakpoints.append(('ZA', breakpoint_date))
                            
                            except Exception as e:
                                st.warning(f"Zivot-Andrews test failed: {str(e)}. Try upgrading to arch>=5.0.0.")
                                results['ZA'] = {
                                    'Test Statistic': None,
                                    'p-value': None,
                                    'Critical Values (5%)': None,
                                    'Lags': lags,
                                    'Regression Type': za_regression if ZA_REGRESSION_SUPPORTED else 'N/A',
                                    'Breakpoint': 'N/A'
                                }
                        
                        if test_options['DFGLS']:
                            try:
                                dfgls = DFGLS(ts, trend=dfgls_regression, lags=lags)
                                results['DFGLS'] = {
                                    'Test Statistic': dfgls.stat,
                                    'p-value': dfgls.pvalue,
                                    'Critical Values (5%)': dfgls.critical_values['5%'],
                                    'Lags': dfgls.lags,
                                    'Regression Type': regression_type_display.get(dfgls_regression, dfgls_regression),
                                    'Breakpoint': 'N/A'
                                }
                            except Exception as e:
                                st.warning(f"DFGLS test failed: {str(e)}. Check parameter compatibility.")
                                results['DFGLS'] = {
                                    'Test Statistic': None,
                                    'p-value': None,
                                    'Critical Values (5%)': None,
                                    'Lags': lags,
                                    'Regression Type': dfgls_regression,
                                    'Breakpoint': 'N/A'
                                }
                        
                        if test_options['VR']:
                            vr = VarianceRatio(ts, lags=lags)
                            results['VR'] = {
                                'Test Statistic': vr.stat,
                                'p-value': vr.pvalue,
                                'Critical Values (5%)': vr.critical_values['5%'],
                                'Lags': lags,
                                'Regression Type': 'N/A (Not Applicable)',
                                'Breakpoint': 'N/A'
                            }
                    
                    if not results:
                        st.warning("No tests selected. Please select at least one test.")
                        st.stop()
                    
                    # Display results
                    results_df = pd.DataFrame(results).T
                    st.subheader("📋 Test Results")
                    st.dataframe(results_df.style.format({
                        'Test Statistic': lambda x: '{:.4f}'.format(x) if isinstance(x, (int, float)) else x,
                        'p-value': lambda x: '{:.4f}'.format(x) if isinstance(x, (int, float)) else x,
                        'Critical Values (5%)': lambda x: '{:.4f}'.format(x) if isinstance(x, (int, float)) else x,
                        'Lags': lambda x: '{:.0f}'.format(x) if isinstance(x, (int, float)) else x
                    }))
                    
                    # Interpretation
                    st.subheader("🔍 Interpretation")
                    for test, values in results.items():
                        p_value = values['p-value']
                        if not isinstance(p_value, (int, float)):
                            continue
                        interpretation = "Stationary" if (test == 'KPSS' and p_value >= 0.05) or \
                            (test != 'KPSS' and p_value < 0.05) else "Non-stationary"
                        st.markdown(f"• {test}: {interpretation} (p-value = {p_value:.4f})")
                    
                    # Visualizations
                    st.subheader("📊 Visual Analysis")
                    viz_tab1, viz_tab2 = st.tabs(["P-value Heatmap", "Time Series with Breaks"])
                    
                    with viz_tab1:
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
                            plt.title("P-values")
                            st.pyplot(fig1)
                        else:
                            st.info("No valid p-values for heatmap.")
                    
                    with viz_tab2:
                        if breakpoints:
                            fig2, ax2 = plt.subplots(figsize=(10, 5))
                            ax2.plot(ts.index, ts.values, label='Time Series')
                            for test_name, bp_date in breakpoints:
                                ax2.axvline(bp_date, color='red', linestyle='--', 
                                           label=f'ZA Break: {bp_date.strftime("%Y-%m-%d")}')
                            ax2.legend()
                            plt.title("Time Series with Structural Breaks")
                            st.pyplot(fig2)
                        else:
                            st.info("No structural breaks detected. Run ZA test.")
                    
                    # Download results
                    st.subheader("📥 Download Results")

                    # CSV download option (doesn't require xlsxwriter)
                    csv_buffer = io.BytesIO()
                    results_df.to_csv(csv_buffer)
                    st.download_button(
                        "📊 Download Results CSV",
                        csv_buffer.getvalue(),
                        f"unit_root_results_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

                    # Also offer time series data download
                    data_buffer = io.BytesIO()
                    ts.to_frame().to_csv(data_buffer)
                    st.download_button(
                        "📈 Download Time Series Data",
                        data_buffer.getvalue(),
                        f"time_series_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Please check your data format and column selections.")

else:
    st.info("Upload a time series file or try sample data.")
    
    if st.button("Load Sample Data"):
        dates = pd.date_range('2010-01-01', periods=120, freq='M')
        trend = np.arange(120) * 0.1
        trend[60:] += 5
        values = trend + 2 * np.sin(np.arange(120) * 2 * np.pi / 12) + \
                 np.random.normal(0, 1, 120) + np.cumsum(np.random.normal(0, 0.5, 120))
        
        sample_df = pd.DataFrame({'date': dates, 'value': values})
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
    st.markdown(f"""
    1. Upload a CSV/Excel file with time series data
    2. Select date and value columns
    3. Choose tests and set number of lags
    4. Run analysis and download results
    
    **Important Note About Zivot-Andrews Test**:
    - Your current arch version is: {'.'.join(str(x) for x in arch_version)}
    - For full Zivot-Andrews functionality with break type selection, you need arch version 5.0.0 or higher
    - To upgrade: `pip install arch>=5.0.0`
    
    **Supported Date Formats**:
    - Daily: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY
    - Monthly: YYYY-MM, YYYYMM, YYYY-MM, 2013M01
    - Quarterly: YYYYQ1, YYYY-Q1
    - Yearly: YYYY
    
    **Dependencies**:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    statsmodels
    arch>=5.0.0
    ```
    """)

st.markdown("© 2025 Unit Root Test App | v2.11.0")
