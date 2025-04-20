import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib as mpl
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, ZivotAndrews, DFGLS, VarianceRatio
import warnings

# Try to import LeeStrazicich, with fallback
try:
    from arch.unitroot import LeeStrazicich
    LEE_STRAZICICH_AVAILABLE = True
except ImportError:
    LEE_STRAZICICH_AVAILABLE = False
    LeeStrazicich = None

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
    'LS': st.sidebar.checkbox("Lee-Strazicich (Structural Breaks)", value=True, disabled=not LEE_STRAZICICH_AVAILABLE),
    'DFGLS': st.sidebar.checkbox("DFGLS", value=True),
    'VR': st.sidebar.checkbox("Variance Ratio", value=False)
}

# Warn if Lee-Strazicich is unavailable
if not LEE_STRAZICICH_AVAILABLE:
    st.sidebar.warning("Lee-Strazicich test requires 'arch' version 5.0.0 or later. Install it with 'pip install arch>=5.0.0' or deselect the test.")

# Test parameters
st.sidebar.subheader("Test Parameters")
lags = st.sidebar.number_input("Number of Lags for All Tests", min_value=1, max_value=20, value=4, step=1)

adf_regression = st.sidebar.selectbox(
    "ADF Regression Type",
    options=["c", "ct", "n", "ctt"],
    format_func=lambda x: {"c": "Constant", "ct": "Constant & Trend", 
                          "n": "No Constant or Trend", "ctt": "Constant, Linear & Quadratic Trend"}.get(x),
    index=1
)
kpss_regression = st.sidebar.selectbox(
    "KPSS Regression Type",
    options=["c", "ct"],
    format_func=lambda x: {"c": "Constant", "ct": "Constant & Trend"}.get(x),
    index=0
)
za_regression = st.sidebar.selectbox(
    "Zivot-Andrews Regression",
    options=["c", "t", "ct"],
    format_func=lambda x: {"c": "Break in Constant", "t": "Break in Trend", 
                          "ct": "Break in Constant & Trend"}.get(x),
    index=2
)
ls_model = st.sidebar.selectbox(
    "Lee-Strazicich Model",
    options=["crash", "break"],
    format_func=lambda x: {"crash": "Crash Model (Level Shift)", "break": "Break Model (Level & Trend Shift)"}.get(x),
    index=1,
    disabled=not LEE_STRAZICICH_AVAILABLE
)

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
            # Check for Lee-Strazicich availability before starting analysis
            if test_options['LS'] and not LEE_STRAZICICH_AVAILABLE:
                st.error("Lee-Strazicich test requires 'arch' version 5.0.0 or later. Please upgrade using 'pip install --upgrade arch' or deselect the Lee-Strazicich test.")
                st.stop()
            
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
                                'Regression Type': adf_regression,
                                'Breakpoint': 'N/A'
                            }
                        
                        if test_options['PP']:
                            pp = PhillipsPerron(ts, trend=adf_regression, lags=lags)
                            results['PP'] = {
                                'Test Statistic': pp.stat,
                                'p-value': pp.pvalue,
                                'Critical Values (5%)': pp.critical_values['5%'],
                                'Lags': lags,
                                'Regression Type': adf_regression,
                                'Breakpoint': 'N/A'
                            }
                        
                        if test_options['KPSS']:
                            kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(ts, regression=kpss_regression, nlags=lags)
                            results['KPSS'] = {
                                'Test Statistic': kpss_stat,
                                'p-value': kpss_pval,
                                'Critical Values (5%)': kpss_crit['5%'],
                                'Lags': lags,
                                'Regression Type': kpss_regression,
                                'Breakpoint': 'N/A'
                            }
                        
                        if test_options['ZA']:
                            za = ZivotAndrews(ts, regression=za_regression, lags=lags)
                            breakpoint_date = ts.index[za.break_idx] if za.break_idx is not None else None
                            results['ZA'] = {
                                'Test Statistic': za.stat,
                                'p-value': za.pvalue,
                                'Critical Values (5%)': za.critical_values['5%'],
                                'Lags': lags,
                                'Regression Type': za_regression,
                                'Breakpoint': breakpoint_date.strftime('%Y-%m-%d') if breakpoint_date else 'N/A'
                            }
                            if breakpoint_date:
                                breakpoints.append(('ZA', breakpoint_date))
                        
                        if test_options['LS'] and LEE_STRAZICICH_AVAILABLE:
                            ls = LeeStrazicich(ts, model=ls_model, lags=lags)
                            break_dates = []
                            if ls.break_idx1 is not None:
                                break_dates.append(ts.index[ls.break_idx1])
                            if ls.break_idx2 is not None:
                                break_dates.append(ts.index[ls.break_idx2])
                            break_dates_str = ', '.join([d.strftime('%Y-%m-%d') for d in break_dates]) if break_dates else 'N/A'
                            results['LS'] = {
                                'Test Statistic': ls.stat,
                                'p-value': ls.pvalue,
                                'Critical Values (5%)': ls.critical_values['5%'],
                                'Lags': ls.lags,
                                'Model': ls_model,
                                'Breakpoint': break_dates_str
                            }
                            for break_date in break_dates:
                                breakpoints.append(('LS', break_date))
                        
                        if test_options['DFGLS']:
                            trend = 'ct' if adf_regression in ['ct', 'ctt'] else 'c'
                            dfgls = DFGLS(ts, trend=trend, max_lags=lags, method='aic')
                            results['DFGLS'] = {
                                'Test Statistic': dfgls.stat,
                                'p-value': dfgls.pvalue,
                                'Critical Values (5%)': dfgls.critical_values['5%'],
                                'Lags': dfgls.lags,
                                'Regression Type': trend,
                                'Breakpoint': 'N/A'
                            }
                        
                        if test_options['VR']:
                            vr = VarianceRatio(ts, lags=lags)
                            results['VR'] = {
                                'Test Statistic': vr.stat,
                                'p-value': vr.pvalue,
                                'Critical Values (5%)': vr.critical_values['5%'],
                                'Lags': lags,
                                'Regression Type': 'N/A',
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
                        if test == 'LS' and values['Breakpoint'] != 'N/A':
                            st.markdown(f"• {test}: Detected breakpoints at {values['Breakpoint']} (p-value = {values['p-value']:.4f})")
                            continue
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
                                ax2.axvline(bp_date, color='red' if test_name == 'ZA' else 'blue', 
                                           linestyle='--', label=f'{test_name} Break: {bp_date.strftime("%Y-%m-%d")}')
                            ax2.legend()
                            plt.title("Time Series with Structural Breaks")
                            st.pyplot(fig2)
                        else:
                            st.info("No structural breaks detected. Run ZA or LS tests.")
                    
                    # Download results
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        results_df.to_excel(writer, sheet_name='Results')
                        ts.to_frame().to_excel(writer, sheet_name='Data')
                    
                    st.download_button(
                        "📊 Download Excel Report",
                        excel_buffer.getvalue(),
                        f"unit_root_results_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel"
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
    st.markdown("""
    1. Upload a CSV/Excel file with time series data
    2. Select date and value columns
    3. Choose tests and set number of lags
    4. Run analysis and download results
    
    **Supported Date Formats**:
    - Daily: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY
    - Monthly: YYYY-MM, YYYYMM, YYYY-MM, 2013M01
    - Quarterly: YYYYQ1, YYYY-Q1
    - Yearly: YYYY
    
    **Dependencies**:
    - Install required packages: `pip install streamlit pandas numpy matplotlib seaborn statsmodels arch xlsxwriter`
    - For Lee-Strazicich test, ensure `arch` version 5.0.0 or later: `pip install arch>=5.0.0`
    - For Streamlit Cloud, add these to a `requirements.txt` file in your GitHub repository:
      ```
      streamlit
      pandas
      numpy
      matplotlib
      seaborn
      statsmodels
      arch>=5.0.0
      xlsxwriter
      ```
    - If Lee-Strazicich is unavailable, deselect the test to avoid errors.
    """)

st.markdown("© 2025 Unit Root Test App | v2.7.0")
