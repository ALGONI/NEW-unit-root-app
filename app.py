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
from matplotlib.backends.backend_pdf import PdfPages
import warnings

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
    'DFGLS': st.sidebar.checkbox("DFGLS", value=False),
    'VR': st.sidebar.checkbox("Variance Ratio", value=False)
}

# Test parameters
st.sidebar.subheader("Test Parameters")
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
    "Zivot-Andrews Model",
    options=["c", "t", "ct"],
    format_func=lambda x: {"c": "Break in Constant", "t": "Break in Trend", 
                          "ct": "Break in Constant & Trend"}.get(x),
    index=2
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
    """Robust date parsing with multiple format attempts"""
    formats = [
        None, '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y%m%d', 
        '%YM%m', '%Y-%m', '%Y/%m', '%Y%m', '%YQ%q', '%Y-Q%q'
    ]
    
    for fmt in formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors='coerce')
            if parsed.notna().sum() > len(series) * 0.8:  # At least 80% successful
                return parsed
        except:
            continue
    
    # Try Year-Month format (2013M01)
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
    
    return pd.to_datetime(series, errors='coerce')

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
                    
                    if len(ts) < 10:
                        st.error("Insufficient data points after cleaning. Need at least 10 observations.")
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
                    with st.spinner("Running unit root tests..."):
                        if test_options['ADF']:
                            adf_result = adfuller(ts, regression=adf_regression, autolag='AIC')
                            results['ADF'] = {
                                'Test Statistic': adf_result[0],
                                'p-value': adf_result[1],
                                'Critical Values (5%)': adf_result[4]['5%'],
                                'Lags': adf_result[2],
                                'Regression Type': adf_regression,
                                'Breakpoint': 'N/A'
                            }
                        
                        if test_options['PP']:
                            pp = PhillipsPerron(ts, trend=adf_regression)
                            results['PP'] = {
                                'Test Statistic': pp.stat,
                                'p-value': pp.pvalue,
                                'Critical Values (5%)': pp.critical_values['5%'],
                                'Lags': pp.lags,
                                'Regression Type': adf_regression,
                                'Breakpoint': 'N/A'
                            }
                        
                        if test_options['KPSS']:
                            kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(ts, regression=kpss_regression, nlags="auto")
                            results['KPSS'] = {
                                'Test Statistic': kpss_stat,
                                'p-value': kpss_pval,
                                'Critical Values (5%)': kpss_crit['5%'],
                                'Lags': kpss_lags,
                                'regression Type': kpss_regression,
                                'Breakpoint': 'N/A'
                            }
                        
                        if test_options['ZA']:
                            za = ZivotAndrews(ts, model=za_regression)
                            breakpoint_date = ts.index[za.break_idx] if za.break_idx is not None else None
                            results['ZA'] = {
                                'Test Statistic': za.stat,
                                'p-value': za.pvalue,
                                'Critical Values (5%)': za.critical_values['5%'],
                                'Lags': za.lags,
                                'Regression Type': za_regression,
                                'Breakpoint': breakpoint_date.strftime('%Y-%m-%d') if breakpoint_date else 'N/A'
                            }
                        
                        if test_options['DFGLS']:
                            dfgls = DFGLS(ts, trend=adf_regression in ['ct', 'ctt'])
                            results['DFGLS'] = {
                                'Test Statistic': dfgls.stat,
                                'p-value': dfgls.pvalue,
                                'Critical Values (5%)': dfgls.critical_values['5%'],
                                'Lags': dfgls.lags,
                                'Regression Type': 'trend' if adf_regression in ['ct', 'ctt'] else 'constant',
                                'Breakpoint': 'N/A'
                            }
                        
                        if test_options['VR']:
                            vr = VarianceRatio(ts, lags=4)
                            results['VR'] = {
                                'Test Statistic': vr.stat,
                                'p-value': vr.pvalue,
                                'Critical Values (5%)': vr.critical_values['5%'],
                                'Lags': 4,
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
                        'Test Statistic': '{:.4f}',
                        'p-value': '{:.4f}',
                        'Critical Values (5%)': '{:.4f}',
                        'Lags': '{:.0f}'
                    }))
                    
                    # Interpretation
                    st.subheader("🔍 Interpretation")
                    for test, values in results.items():
                        p_value = values['p-value']
                        interpretation = "Stationary" if (test == 'KPSS' and p_value >= 0.05) or \
                            (test != 'KPSS' and p_value < 0.05) else "Non-stationary"
                        st.markdown(f"• {test}: {interpretation} (p-value = {p_value:.4f})")
                    
                    # Visualizations
                    st.subheader("📊 Visual Analysis")
                    viz_tab1, viz_tab2 = st.tabs(["P-value Heatmap", "Time Series with Breaks"])
                    
                    with viz_tab1:
                        fig1, ax1 = plt.subplots(figsize=(8, len(results)/2))
                        sns.heatmap(
                            results_df[["p-value"]].astype(float),
                            annot=True,
                            cmap='RdYlGn_r',
                            fmt=".4f",
                            vmin=0,
                            vmax=0.1
                        )
                        plt.title("P-values")
                        st.pyplot(fig1)
                    
                    with viz_tab2:
                        if test_options['ZA'] and 'ZA' in results and results['ZA']['Breakpoint'] != 'N/A':
                            fig2, ax2 = plt.subplots(figsize=(10, 5))
                            ax2.plot(ts.index, ts.values)
                            ax2.axvline(pd.to_datetime(results['ZA']['Breakpoint']), 
                                      color='red', linestyle='--', 
                                      label=f'Break: {results["ZA"]["Breakpoint"]}')
                            ax2.legend()
                            plt.title("Time Series with Structural Break")
                            st.pyplot(fig2)
                        else:
                            st.info("Run Zivot-Andrews test to see structural breaks.")
                    
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
    3. Choose tests and parameters
    4. Run analysis and download results
    
    **Note**: Data should have consistent date formats and numeric values.
    """)

st.markdown("© 2025 Unit Root Test App | v2.1.0")
