# Updated test execution code with standardized options
# This section goes inside your "if st.sidebar.button("▶️ Run Analysis", use_container_width=True):" block

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
