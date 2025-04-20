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
# Import PdfPages at the module level for better code organization
from matplotlib.backends.backend_pdf import PdfPages

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
    page_title="Unit Root Test App", 
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
</style>
""", unsafe_allow_html=True)

# Sidebar for configuration - use a placeholder image that doesn't require external URLs
st.sidebar.title("📊 Test Configuration")

# Main content
st.title("📊 Advanced Unit Root Test Application")
st.markdown("#### Analyze time series stationarity with detection of structural breaks")

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
            value_col = st.selectbox("📈 Select Value Column", options=[col for col in df.columns if col != date_col])
        
        # Test selection
        st.sidebar.subheader("Select Tests to Run")
        run_adf = st.sidebar.checkbox("Augmented Dickey-Fuller (ADF)", value=True)
        run_pp = st.sidebar.checkbox("Phillips-Perron (PP)", value=True)
        run_kpss = st.sidebar.checkbox("KPSS", value=True)
        run_za = st.sidebar.checkbox("Zivot-Andrews (with structural breaks)", value=True)
        run_dfgls = st.sidebar.checkbox("DFGLS", value=False)
        run_vr = st.sidebar.checkbox("Variance Ratio", value=False)
        
        # Test parameters
        st.sidebar.subheader("Test Parameters")
        
        # Add information about date format detection
        st.sidebar.markdown("""
        ### Date Format Detection
        The app will attempt to automatically detect and handle various date formats, including:
        - Standard formats (YYYY-MM-DD)
        - Year-Month formats (e.g., 2013M01)
        - Quarterly formats (e.g., 2013Q1)
        """)
        
        adf_regression = st.sidebar.selectbox("ADF Regression Type", 
                                        options=["c", "ct", "n", "ctt"], 
                                        format_func=lambda x: {
                                            "c": "Constant",
                                            "ct": "Constant & Trend",
                                            "n": "No Constant or Trend",
                                            "ctt": "Constant, Linear & Quadratic Trend"
                                        }.get(x),
                                        index=1)
        
        kpss_regression = st.sidebar.selectbox("KPSS Regression Type", 
                                        options=["c", "ct"], 
                                        format_func=lambda x: {
                                            "c": "Constant",
                                            "ct": "Constant & Trend"
                                        }.get(x),
                                        index=0)
        
        za_regression = st.sidebar.selectbox("Zivot-Andrews Model", 
                                    options=["c", "t", "ct"], 
                                    format_func=lambda x: {
                                        "c": "Break in Constant",
                                        "t": "Break in Trend",
                                        "ct": "Break in Constant & Trend"
                                    }.get(x),
                                    index=2)
        
        # Process Button
        process_button = st.sidebar.button("▶️ Run Analysis", use_container_width=True)
        
        if process_button:
            try:
                # Data processing with robust date parsing
                # Try to detect and handle various date formats including "2013M01" format
                try:
                    # First check if the data contains a format like "2013M01"
                    sample_date = df[date_col].iloc[0]
                    if isinstance(sample_date, str) and 'M' in sample_date and len(sample_date) == 7:
                        # This is likely a year-month format like "2013M01"
                        st.info(f"Detected date format like '{sample_date}' (YearMonth). Parsing accordingly.")
                        
                        # Custom parsing for "2013M01" format - extract year and month
                        def parse_year_month(date_str):
                            if isinstance(date_str, str) and 'M' in date_str:
                                parts = date_str.split('M')
                                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                                    year = int(parts[0])
                                    month = int(parts[1])
                                    return pd.Timestamp(year=year, month=month, day=1)
                            # If we can't parse it with our custom parser, try pandas
                            return pd.to_datetime(date_str, errors='coerce')
                        
                        df[date_col] = df[date_col].apply(parse_year_month)
                    elif isinstance(sample_date, str) and 'Q' in sample_date:
                        # Handle quarterly format like "2013Q1"
                        st.info(f"Detected quarterly date format like '{sample_date}'. Parsing accordingly.")
                        df[date_col] = pd.PeriodIndex(df[date_col], freq='Q').to_timestamp()
                    else:
                        # Standard datetime parsing for other formats
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        
                        # Check if we have NaT values after parsing and inform the user
                        nat_count = df[date_col].isna().sum()
                        if nat_count > 0:
                            st.warning(f"Found {nat_count} unparseable date values. These rows will be excluded from analysis.")
                except Exception as e:
                    st.error(f"Error parsing dates: {str(e)}")
                    st.info("Trying alternative date parsing methods...")
                    
                    # Try with format inference and various common formats
                    try:
                        # Try with some common time series formats
                        formats = ['%YM%m', '%Y-%m', '%Y/%m', '%Y%m', '%YQ%q', '%Y-Q%q']
                        
                        for fmt in formats:
                            try:
                                df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors='coerce')
                                if not df[date_col].isna().all():
                                    st.success(f"Successfully parsed dates with format: {fmt}")
                                    break
                            except:
                                continue
                    except:
                        pass
                
                # Drop rows with NaT dates and continue with processing
                df = df[[date_col, value_col]].dropna()
                df.set_index(date_col, inplace=True)
                ts = df[value_col]
                
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
                
                with st.spinner("Running unit root tests..."):
                    # ADF Test
                    if run_adf:
                        adf_result = adfuller(ts, regression=adf_regression, autolag='AIC')
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
                    
                    # PP Test - FIXED: Initialize with the correct trend parameter
                    if run_pp:
                        pp = PhillipsPerron(ts, trend=adf_regression)
                        # Remove the redundant reassignment that was causing issues
                        results['Phillips-Perron'] = {
                            'Test Statistic': pp.stat,
                            'p-value': pp.pvalue,
                            'Critical Values (1%)': pp.critical_values['1%'],
                            'Critical Values (5%)': pp.critical_values['5%'],
                            'Critical Values (10%)': pp.critical_values['10%'],
                            'Lags': pp.lags,
                            'Regression Type': adf_regression,
                            'Breakpoint': 'N/A'
                        }
                    
                    # KPSS Test
                    if run_kpss:
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
                    
                    # Zivot-Andrews Test
                    breakpoint_date = None
                    if run_za:
                        za = ZivotAndrews(ts, model=za_regression)
                        breakpoint_date = ts.index[za.break_idx] 
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
                    
                    # DFGLS Test
                    if run_dfgls:
                        dfgls = DFGLS(ts, trend=adf_regression in ['ct', 'ctt'])
                        results['DF-GLS'] = {
                            'Test Statistic': dfgls.stat,
                            'p-value': dfgls.pvalue,
                            'Critical Values (1%)': dfgls.critical_values['1%'],
                            'Critical Values (5%)': dfgls.critical_values['5%'],
                            'Critical Values (10%)': dfgls.critical_values['10%'],
                            'Lags': dfgls.lags,
                            'Regression Type': 'trend' if adf_regression in ['ct', 'ctt'] else 'constant',
                            'Breakpoint': 'N/A'
                        }
                    
                    # Variance Ratio Test
                    if run_vr:
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
                
                st.success("✅ Unit root tests completed!")
                
                # Check if there is valid data to proceed with
                if len(ts) == 0:
                    st.error("No valid data points after date parsing and cleaning. Please check your input data and column selections.")
                    return
                
                # Check if there are any results before proceeding
                if not results:
                    st.warning("No tests were selected. Please select at least one test to run.")
                else:
                    # Extract key results for summary table
                    summary_cols = ['Test Statistic', 'p-value', 'Critical Values (5%)', 'Breakpoint']
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
                        st.subheader("🔍 Quick Interpretation")
                        
                        # Create interpretation
                        interpretations = []
                        
                        for test, values in results.items():
                            p_value = values['p-value']
                            stat = values['Test Statistic']
                            crit_val = values.get('Critical Values (5%)', None)
                            
                            if test == 'KPSS':
                                # For KPSS, null hypothesis is stationarity
                                if p_value < 0.05:
                                    interpretations.append(f"• {test}: Non-stationary (p-value = {p_value:.4f} < 0.05)")
                                else:
                                    interpretations.append(f"• {test}: Stationary (p-value = {p_value:.4f} ≥ 0.05)")
                            else:
                                # For other tests, null hypothesis is non-stationarity
                                if p_value < 0.05:
                                    interpretations.append(f"• {test}: Stationary (p-value = {p_value:.4f} < 0.05)")
                                else:
                                    interpretations.append(f"• {test}: Non-stationary (p-value = {p_value:.4f} ≥ 0.05)")
                        
                        # Display interpretations
                        for interp in interpretations:
                            st.markdown(interp)
                    
                    with tab2:
                        st.subheader("📊 Detailed Test Results")
                        st.dataframe(detailed_results.style.format({
                            'Test Statistic': '{:.4f}',
                            'p-value': '{:.4f}',
                            'Critical Values (1%)': '{:.4f}',
                            'Critical Values (5%)': '{:.4f}',
                            'Critical Values (10%)': '{:.4f}',
                            'Lags': '{:.0f}'
                        }), use_container_width=True)
                    
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
                        if run_za and breakpoint_date is not None:
                            # Time Series Plot with Breaks
                            fig2, ax2 = plt.subplots(figsize=(10, 5))
                            ax2.plot(ts.index, ts.values, label='Time Series', linewidth=2)
                            
                            # Add breakpoint line
                            ax2.axvline(breakpoint_date, color='red', linestyle='--', linewidth=2, 
                                        label=f'Zivot-Andrews Break: {breakpoint_date.strftime("%Y-%m-%d")}')
                            
                            # Add shaded regions before and after break
                            ax2.axvspan(ts.index[0], breakpoint_date, alpha=0.1, color='blue', label='Pre-Break Period')
                            ax2.axvspan(breakpoint_date, ts.index[-1], alpha=0.1, color='red', label='Post-Break Period')
                            
                            ax2.set_title("Time Series with Structural Break Detection")
                            ax2.set_xlabel("Date")
                            ax2.set_ylabel(value_col)
                            ax2.legend(loc='best')
                            ax2.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig2)
                        else:
                            st.warning("Zivot-Andrews test must be selected to visualize structural breaks.")
                    
                    with viz_tab3:
                        # Create differentiated series
                        ts_diff = ts.diff().dropna()
                        
                        # Create subplot
                        fig3, axs = plt.subplots(2, 1, figsize=(10, 8))
                        
                        # Original series
                        axs[0].plot(ts.index, ts.values, label='Original Series', color='blue')
                        axs[0].set_title("Original Time Series")
                        axs[0].grid(True, alpha=0.3)
                        
                        # Differenced series
                        axs[1].plot(ts_diff.index, ts_diff.values, label='Differenced Series', color='green')
                        axs[1].set_title("First Differenced Series")
                        axs[1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig3)
                        
                        # Add explanatory text
                        st.markdown("""
                        **Stationarity Analysis:**
                        
                        The charts above show the original time series and its first difference. 
                        A stationary series should have:
                        
                        1. Constant mean over time
                        2. Constant variance over time
                        3. No systematic pattern in the autocovariance
                        
                        Often, differencing a non-stationary series can make it stationary,
                        which is visible as a more stable pattern in the differenced series.
                        """)
                    
                    # Download options
                    st.subheader("📥 Download Results")
                    
                    # Create Excel file with multiple sheets
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        # Summary sheet
                        results_df.to_excel(writer, sheet_name='Summary Results')
                        
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
                            'Tests Run': ', '.join(results.keys())
                        }
                        pd.DataFrame(list(meta_data.items()), columns=['Metadata', 'Value']).to_excel(writer, sheet_name='Metadata')
                    
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
                        with PdfPages(pdf_buffer) as pdf:
                            # Add all figures
                            if 'fig' in locals(): pdf.savefig(fig)
                            if 'fig1' in locals(): pdf.savefig(fig1)
                            if run_za and 'fig2' in locals() and breakpoint_date is not None: 
                                pdf.savefig(fig2)
                            if 'fig3' in locals(): pdf.savefig(fig3)
                        
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
        - **DF-GLS**: A more powerful variant of the ADF test
        - **Variance Ratio**: Tests the random walk hypothesis
        
        For ADF, PP, DF-GLS, and Zivot-Andrews tests, a p-value < 0.05 suggests stationarity.  
        For KPSS test, a p-value < 0.05 suggests non-stationarity.
        """)
        
    # Footer
    st.markdown("---")
    st.markdown(
        "Created with ❤️ | "
        "© 2025 Advanced Unit Root Testing Application | "
        "Version 2.0.0"
    )

