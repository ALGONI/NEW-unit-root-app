import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, ZivotAndrews
# from linearmodels.unitroot import ZivotAndrews as LMZivotAndrews

# --- Page Settings ---
st.set_page_config(page_title="Unit Root Test App", layout="wide")
st.title("📊 Unit Root Test Application with Structural Breaks")

uploaded_file = st.file_uploader("Upload a Time Series File (CSV or Excel, columns: date, value)", type=["csv", "xlsx"])

if uploaded_file:
    # STEP 1: Load CSV or Excel
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # STEP 2: Clean column names (lowercase, strip spaces)
    df.columns = [col.strip().lower() for col in df.columns]

    # STEP 3: Let user select which columns to use
    st.subheader("📌 Select Columns")
    date_col = st.selectbox("Select Date Column", options=df.columns)
    value_col = st.selectbox("Select Value Column", options=df.columns)

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df[[date_col, value_col]].dropna()
        df.set_index(date_col, inplace=True)
        ts = df[value_col]

        st.subheader("📉 Time Series Preview")
        st.line_chart(ts, use_container_width=True)

        results = {}

        # ✅ your test logic and plots can continue from here...

    except Exception as e:
        st.error(f"Error processing selected columns: {e}")
        # ADF Test
        adf_result = adfuller(ts, autolag='AIC')
        results['ADF'] = {'Test Statistic': adf_result[0], 'p-value': adf_result[1], 'Breakpoint': 'N/A'}

        # PP Test
        pp = PhillipsPerron(ts)
        results['PP'] = {'Test Statistic': pp.stat, 'p-value': pp.pvalue, 'Breakpoint': 'N/A'}

        # KPSS Test
        kpss_stat, kpss_pval, _, _ = kpss(ts, regression='c', nlags="auto")
        results['KPSS'] = {'Test Statistic': kpss_stat, 'p-value': kpss_pval, 'Breakpoint': 'N/A'}

        # Zivot-Andrews Test
        za = ZivotAndrews(ts)
        results['Zivot-Andrews'] = {
            'Test Statistic': za.stat,
            'p-value': za.pvalue,
            'Breakpoint': ts.index[za.break_point].strftime('%Y-%m')
        }

        # Results Summary Table
        results_df = pd.DataFrame(results).T.round(4)
        st.subheader("📋 Test Results Summary")
        st.dataframe(results_df)

        # P-value Heatmap
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        sns.heatmap(results_df[["p-value"]].astype(float), annot=True, cmap='coolwarm', fmt=".4f", ax=ax1)
        ax1.set_title("P-values from Unit Root Tests")
        st.pyplot(fig1)

        # Time Series Plot with Breaks
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(ts, label='Time Series')
        ax2.axvline(ts.index[za.break_point], color='red', linestyle='--', label='Zivot-Andrews Break')
        ax2.set_title("Time Series with Structural Breaks")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # Download Excel Results
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, sheet_name='Unit Root Tests')
        st.download_button("📥 Download Excel Results", data=excel_buffer.getvalue(), file_name="unit_root_results.xlsx")

        # Download Time Series Plot
        pdf_buffer = io.BytesIO()
        fig2.savefig(pdf_buffer, format='pdf')
        st.download_button("📥 Download Time Series Plot (PDF)", data=pdf_buffer.getvalue(), file_name="time_series_plot.pdf")

        # Download Heatmap
        heatmap_buffer = io.BytesIO()
        fig1.savefig(heatmap_buffer, format='pdf')
        st.download_button("📥 Download P-Value Heatmap (PDF)", data=heatmap_buffer.getvalue(), file_name="p_value_heatmap.pdf")

    else:
        st.warning("CSV must contain columns: 'date' and 'value'")
else:
    st.info("Please upload a CSV file with 'date' and 'value' columns.")
