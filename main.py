import streamlit as st
import pandas as pd
import os
import tempfile
import importlib.util
import runpy
import sys
import traceback
from pathlib import Path
from dotenv import load_dotenv
from src.utils import PDFtoCSV, generate_weekly_summary, generate_monthly_summary
load_dotenv()

def ensure_amount_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a numeric 'Amount' column.

    Handles common patterns: existing 'Amount', separate 'Debit'/'Credit',
    single 'Debit' or 'Credit', or alt names like 'Value'/'Amt'. Falls back to 0.0.
    Returns a copy of the DataFrame with 'Amount' present.
    """
    df = df.copy()
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        return df

    # Debit/Credit pair
    if 'Debit' in df.columns and 'Credit' in df.columns:
        df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
        df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
        df['Amount'] = df['Credit'] - df['Debit']
        return df

    # Single-column alternatives
    alt = next((c for c in df.columns if c.lower() in ('value', 'amt', 'transaction_amount', 'amount_usd', 'amount(usd)')), None)
    if alt:
        # remove common currency formatting
        try:
            df[alt] = df[alt].astype(str).str.replace(r'[\$,()]', '', regex=True)
        except Exception:
            pass
        df['Amount'] = pd.to_numeric(df[alt], errors='coerce').fillna(0)
        return df

    # If only Debit present, treat as negative outflow
    if 'Debit' in df.columns:
        df['Amount'] = -pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
        return df

    # If only Credit present
    if 'Credit' in df.columns:
        df['Amount'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
        return df

    # Last resort: add zero Amount column
    df['Amount'] = 0.0
    return df


def load_script_function(script_path: Path, func_name: str = "main"):
    """Dynamically load a function from a script file.

    Returns the callable or raises ImportError.
    """
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    spec_name = f"script_{script_path.stem}_{abs(hash(str(script_path))) % (10**8)}"
    try:
        spec = importlib.util.spec_from_file_location(spec_name, str(script_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to load script {script_path}: {e}")

    if not hasattr(module, func_name):
        # if main not present, try to return the module to run via runpy
        return None, module

    return getattr(module, func_name), module


def run_script(script_rel: str, func_name: str = "main", *args, **kwargs):
    """Helper to run a script function and return result or raise.
    script_rel is a path relative to repo root.
    """
    script_path = Path.cwd() / script_rel
    try:
        func, module = load_script_function(script_path, func_name)
        if func:
            return func(*args, **kwargs)
        else:
            # fallback: execute script file in a new namespace
            ns = runpy.run_path(str(script_path), run_name="__main__")
            return ns
    except Exception:
        tb = traceback.format_exc()
        raise RuntimeError(f"Error running script {script_rel}:\n{tb}")


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Bank Statement Analyzer", layout="wide")
st.title("📄 Finance Advisor — Bank Statement Toolkit")

st.sidebar.title("Tools")
tool = st.sidebar.radio("Select tool", [
    "PDF Extractor",
    "Analytics Engine",
    "Anomaly Detector",
    "LLM Advisor",
    "Training"
])

st.cache_data.clear()
st.cache_resource.clear()


if tool == "PDF Extractor":
    st.header("🔓 PDF -> CSV Extractor")

    uploaded_pdf = st.file_uploader("Upload Bank Statement PDF", type=["pdf"])
    pdf_password = st.text_input("PDF Password (if any)", type="password")

    if uploaded_pdf:
        st.success("PDF uploaded")
        temp_dir = Path(tempfile.mkdtemp())
        pdf_path = temp_dir / "uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

        # Try to run user's 01_pdf_extractor script if available
        extractor_script = Path("scripts") / "01_pdf_extractor.py"
        if st.button("Convert using script (if available) / fallback"):
            with st.spinner("Converting PDF..."):
                try:
                    # Prefer script if exists
                    if extractor_script.exists():
                        # many extractor scripts expose a function like `pdf_to_csv` or `main`
                        try:
                            res = run_script(str(extractor_script), "pdf_to_csv", str(pdf_path), pdf_password)
                            # If script returns path or DataFrame, handle
                            if isinstance(res, pd.DataFrame):
                                df = res
                            elif isinstance(res, dict) and "df" in res:
                                df = res["df"]
                            else:
                                # fallback to running script which may have written files
                                st.info("Extractor script executed; scanning output folder for CSVs.")
                                # try fallback parser
                                converter = PDFtoCSV(output_folder=str(temp_dir / "tables"))
                                df = converter.convert(str(pdf_path), password=pdf_password if pdf_password else None)
                        except Exception:
                            # if script fails, fallback
                            converter = PDFtoCSV(output_folder=str(temp_dir / "tables"))
                            df = converter.convert(str(pdf_path), password=pdf_password if pdf_password else None)
                    else:
                        converter = PDFtoCSV(output_folder=str(temp_dir / "tables"))
                        df = converter.convert(str(pdf_path), password=pdf_password if pdf_password else None)

                    st.session_state['df'] = df
                    st.success("Conversion complete")
                    st.dataframe(df)
                    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="statement.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Conversion failed: {e}")


elif tool == "Analytics Engine":
    st.header("📊 Advanced Analytics & Insights")
    
    if "df" not in st.session_state:
        st.info("No DataFrame found in session. Please use the PDF Extractor to upload/convert a PDF first.")
    else:
        df = st.session_state['df']
        
        # Initialize Analytics Engine
        try:
            from scripts.analytics_engine import AnalyticsEngine
            ae = AnalyticsEngine(df)
            
            # Display Summary Metrics Dashboard
            st.subheader("📈 Financial Overview")
            ae.display_summary_metrics()
            
            st.divider()
            
            # Tabs for organized analytics
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Visualizations", 
                "📋 Summaries", 
                "🔍 Deep Dive", 
                "🚨 Insights", 
                "📁 Data Export"
            ])
            
            # ==================== TAB 1: VISUALIZATIONS ====================
            with tab1:
                st.subheader("Interactive Charts")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    if st.checkbox("Daily Spending Trend", value=True):
                        ae.plot_monthly_spending_line()
                    
                    if st.checkbox("Category Distribution"):
                        ae.plot_category_pie()
                    
                    if st.checkbox("Spending Heatmap"):
                        ae.plot_heatmap()
                
                with viz_col2:
                    if st.checkbox("Inflow vs Outflow", value=True):
                        ae.plot_inflow_outflow_bar()
                    
                    if st.checkbox("Savings Trend"):
                        ae.plot_savings_trend()
                    
                    if st.checkbox("Anomaly Detection"):
                        ae.plot_anomalies()
            
            # ==================== TAB 2: SUMMARIES ====================
            with tab2:
                st.subheader("Quick Summaries")
                
                sum_col1, sum_col2 = st.columns(2)
                
                with sum_col1:
                    if st.button("📅 Weekly Summary", use_container_width=True):
                        with st.spinner("Generating weekly summary..."):
                            try:
                                from src.features.weekly import generate_weekly_summary
                                summary = generate_weekly_summary(df)
                                st.dataframe(summary, use_container_width=True)
                                st.download_button(
                                    "💾 Download Weekly CSV",
                                    data=summary.to_csv(index=False),
                                    file_name="weekly_summary.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Failed to compute weekly summary: {e}")
                    
                    if st.button("📆 Monthly Summary", use_container_width=True):
                        with st.spinner("Generating monthly summary..."):
                            try:
                                from src.features.monthly import generate_monthly_summary
                                summary = generate_monthly_summary(df)
                                st.dataframe(summary, use_container_width=True)
                                st.download_button(
                                    "💾 Download Monthly CSV",
                                    data=summary.to_csv(index=False),
                                    file_name="monthly_summary.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Failed to compute monthly summary: {e}")
                    
                    if st.button("📊 Quarterly Summary", use_container_width=True):
                        with st.spinner("Generating quarterly summary..."):
                            try:
                                quarterly = ae.get_quarterly_summary()
                                st.dataframe(quarterly, use_container_width=True)
                                st.download_button(
                                    "💾 Download Quarterly CSV",
                                    data=quarterly.to_csv(index=False),
                                    file_name="quarterly_summary.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Failed to compute quarterly summary: {e}")
                
                with sum_col2:
                    if st.button("💰 Savings Analysis", use_container_width=True):
                        with st.spinner("Calculating savings..."):
                            try:
                                savings = ae.get_monthly_savings()
                                st.dataframe(savings, use_container_width=True)
                                
                                # Display key metrics
                                avg_savings = savings["Savings"].mean()
                                total_savings = savings["Savings"].sum()
                                avg_rate = savings["SavingsRate"].mean()
                                
                                metric_col1, metric_col2, metric_col3 = st.columns(3)
                                metric_col1.metric("Avg Monthly Savings", f"₹{avg_savings:,.2f}")
                                metric_col2.metric("Total Savings", f"₹{total_savings:,.2f}")
                                metric_col3.metric("Avg Savings Rate", f"{avg_rate:.1f}%")
                                
                                st.download_button(
                                    "💾 Download Savings CSV",
                                    data=savings.to_csv(index=False),
                                    file_name="savings_analysis.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Failed to compute savings: {e}")
                    
                    if st.button("📈 Month-over-Month Changes", use_container_width=True):
                        with st.spinner("Calculating changes..."):
                            try:
                                mom = ae.get_month_over_month_change()
                                st.dataframe(mom, use_container_width=True)
                                st.download_button(
                                    "💾 Download MoM CSV",
                                    data=mom.to_csv(index=False),
                                    file_name="month_over_month.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Failed to compute MoM changes: {e}")
                    
                    if st.button("🏪 Top Merchants", use_container_width=True):
                        with st.spinner("Finding top merchants..."):
                            try:
                                n_merchants = st.slider("Number of merchants", 5, 50, 10)
                                merchants = ae.get_top_merchants(n=n_merchants)
                                if merchants is not None:
                                    st.dataframe(merchants, use_container_width=True)
                                    st.download_button(
                                        "💾 Download Merchants CSV",
                                        data=merchants.to_csv(index=False),
                                        file_name="top_merchants.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                else:
                                    st.info("No 'Description' column found in data.")
                            except Exception as e:
                                st.error(f"Failed to get top merchants: {e}")
            
            # ==================== TAB 3: DEEP DIVE ====================
            with tab3:
                st.subheader("Detailed Analysis")
                
                # Spending Patterns
                with st.expander("🔍 Spending Patterns", expanded=True):
                    try:
                        patterns = ae.get_spending_patterns()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Weekend vs Weekday Spending**")
                            weekend_total = patterns["weekend_vs_weekday"]["Weekend"]
                            weekday_total = patterns["weekend_vs_weekday"]["Weekday"]
                            st.metric("Weekend Total", f"₹{weekend_total:,.2f}")
                            st.metric("Weekday Total", f"₹{weekday_total:,.2f}")
                            
                            st.write("**Peak Spending Day**")
                            st.info(f"📅 {patterns['peak_spending_day']}")
                        
                        with col2:
                            st.write("**Average Spending by Day**")
                            weekday_df = pd.DataFrame.from_dict(
                                patterns["weekday_avg"], 
                                orient="index", 
                                columns=["Average"]
                            )
                            st.dataframe(weekday_df)
                            
                            st.write("**Transaction Size Distribution**")
                            size_df = pd.DataFrame.from_dict(
                                patterns["by_size"], 
                                orient="index", 
                                columns=["Count"]
                            )
                            st.dataframe(size_df)
                    except Exception as e:
                        st.error(f"Failed to analyze patterns: {e}")
                
                # Category Analysis
                with st.expander("🧩 Category Deep Dive"):
                    try:
                        categories = ae.get_category_totals()
                        if categories is not None:
                            st.dataframe(categories, use_container_width=True)
                            
                            # Show category metrics
                            top_category = categories.iloc[0]
                            st.success(f"🏆 Top Category: **{top_category['Category']}** - ₹{top_category['Total']:,.2f} ({top_category['Percentage']:.1f}%)")
                        else:
                            st.info("Category information not available. Run ML categorization first.")
                    except Exception as e:
                        st.error(f"Failed to analyze categories: {e}")
                
                # Budget Analysis
                with st.expander("💵 Budget Analysis"):
                    budget_amount = st.number_input(
                        "Enter Monthly Budget (₹)",
                        min_value=0.0,
                        value=50000.0,
                        step=1000.0
                    )
                    
                    if st.button("Analyze Budget Performance"):
                        try:
                            budget_analysis = ae.get_budget_analysis(budget_amount)
                            st.dataframe(budget_analysis, use_container_width=True)
                            
                            # Summary metrics
                            over_budget_months = len(budget_analysis[budget_analysis["Status"] == "Over Budget"])
                            under_budget_months = len(budget_analysis[budget_analysis["Status"] == "Under Budget"])
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Months Under Budget", under_budget_months, delta="Good", delta_color="normal")
                            col2.metric("Months Over Budget", over_budget_months, delta="Warning", delta_color="inverse")
                        except Exception as e:
                            st.error(f"Failed to analyze budget: {e}")
                
                # Recurring Transactions
                with st.expander("🔄 Recurring Transactions"):
                    min_occurrences = st.slider("Minimum occurrences", 2, 10, 3)
                    
                    if st.button("Find Recurring Transactions"):
                        try:
                            recurring = ae.get_recurring_transactions(similarity_threshold=min_occurrences)
                            if len(recurring) > 0:
                                st.dataframe(recurring, use_container_width=True)
                                st.info(f"Found {len(recurring)} potentially recurring transactions")
                            else:
                                st.warning("No recurring transactions found with current threshold")
                        except Exception as e:
                            st.error(f"Failed to find recurring transactions: {e}")
            
            # ==================== TAB 4: INSIGHTS ====================
            with tab4:
                st.subheader("🚨 Automated Insights")
                
                # Anomalies
                with st.expander("⚠️ Unusual Transactions", expanded=True):
                    sensitivity = st.slider(
                        "Sensitivity (higher = fewer anomalies)",
                        1.5, 4.0, 2.5, 0.5
                    )
                    
                    try:
                        anomalies = ae.get_anomalies(threshold=sensitivity)
                        if len(anomalies) > 0:
                            st.warning(f"🚨 Found {len(anomalies)} unusual transactions")
                            st.dataframe(anomalies, use_container_width=True)
                            
                            st.download_button(
                                "💾 Download Anomalies",
                                data=anomalies.to_csv(index=False),
                                file_name="anomalies.csv",
                                mime="text/csv"
                            )
                        else:
                            st.success("✅ No significant anomalies detected")
                    except Exception as e:
                        st.error(f"Failed to detect anomalies: {e}")
                
                # Cash Flow Forecast
                with st.expander("🔮 Cash Flow Forecast"):
                    forecast_months = st.slider("Forecast periods (months)", 1, 12, 3)
                    
                    if st.button("Generate Forecast"):
                        try:
                            forecast = ae.get_cash_flow_forecast(periods=forecast_months)
                            st.dataframe(forecast, use_container_width=True)
                            
                            st.info("📊 Forecast is based on the average of the last 6 months")
                            
                            # Visualize forecast
                            import plotly.graph_objects as go
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=forecast["Month"],
                                y=forecast["ForecastInflow"],
                                name="Expected Inflow",
                                marker_color="green"
                            ))
                            
                            fig.add_trace(go.Bar(
                                x=forecast["Month"],
                                y=forecast["ForecastOutflow"],
                                name="Expected Outflow",
                                marker_color="red"
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast["Month"],
                                y=forecast["ForecastSavings"],
                                mode="lines+markers",
                                name="Expected Savings",
                                line=dict(color="blue", width=3)
                            ))
                            
                            fig.update_layout(
                                title="Cash Flow Forecast",
                                barmode="group",
                                hovermode="x unified"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to generate forecast: {e}")
                
                # Smart Recommendations
                with st.expander("💡 Smart Recommendations"):
                    try:
                        savings = ae.get_monthly_savings()
                        avg_savings_rate = savings["SavingsRate"].mean()
                        
                        st.write("**Based on your spending patterns:**")
                        
                        if avg_savings_rate < 10:
                            st.warning("🔴 Your savings rate is below 10%. Consider reviewing your expenses.")
                        elif avg_savings_rate < 20:
                            st.info("🟡 Your savings rate is moderate. There's room for improvement.")
                        else:
                            st.success("🟢 Great job! You're maintaining a healthy savings rate.")
                        
                        # Additional recommendations
                        patterns = ae.get_spending_patterns()
                        weekend_pct = patterns["weekend_vs_weekday"]["Weekend"] / (patterns["weekend_vs_weekday"]["Weekend"] + patterns["weekend_vs_weekday"]["Weekday"]) * 100
                        
                        if weekend_pct > 40:
                            st.info(f"💡 {weekend_pct:.1f}% of your spending happens on weekends. Consider planning weekend activities on a budget.")
                        
                        # Check for large transactions
                        large_transactions = len(ae.df[ae.df["TransactionSize"] == "Very Large"])
                        if large_transactions > 5:
                            st.info(f"💡 You have {large_transactions} very large transactions. Review these for potential savings.")
                    except Exception as e:
                        st.error(f"Failed to generate recommendations: {e}")
            
            # ==================== TAB 5: DATA EXPORT ====================
            with tab5:
                st.subheader("📁 Export Analytics Data")
                
                st.write("Export your complete analytics for external use:")
                
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    if st.button("📊 Export All Summaries", use_container_width=True):
                        try:
                            all_summaries = {
                                'monthly_inflow_outflow': ae.get_monthly_inflow_outflow(),
                                'category_totals': ae.get_category_totals(),
                                'daily_trends': ae.get_daily_trends(),
                                'monthly_savings': ae.get_monthly_savings(),
                                'top_merchants': ae.get_top_merchants(10),
                                'spending_patterns': str(ae.get_spending_patterns()),
                                'quarterly_summary': ae.get_quarterly_summary(),
                                'anomalies': ae.get_anomalies(),
                                'recurring_transactions': ae.get_recurring_transactions()
                            }
                            
                            # Create a downloadable package
                            st.success("✅ Summaries generated successfully!")
                            
                            for key, data in all_summaries.items():
                                if data is not None and isinstance(data, pd.DataFrame):
                                    st.download_button(
                                        f"Download {key.replace('_', ' ').title()}",
                                        data=data.to_csv(index=False),
                                        file_name=f"{key}.csv",
                                        mime="text/csv",
                                        key=f"export_{key}"
                                    )
                        except Exception as e:
                            st.error(f"Export failed: {e}")
                
                with export_col2:
                    if st.button("📈 Export Processed DataFrame", use_container_width=True):
                        try:
                            processed_df = ae.df
                            csv = processed_df.to_csv(index=False)
                            st.download_button(
                                "💾 Download Processed Data",
                                data=csv,
                                file_name="processed_transactions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            st.success("✅ Processed data ready for download!")
                        except Exception as e:
                            st.error(f"Export failed: {e}")
                
                st.divider()
                
                # Data Preview
                st.subheader("📋 Data Preview")
                preview_rows = st.slider("Rows to preview", 10, 500, 100)
                st.dataframe(ae.df.head(preview_rows), use_container_width=True)
                
                # Data Info
                with st.expander("ℹ️ Dataset Information"):
                    info_col1, info_col2, info_col3 = st.columns(3)
                    
                    info_col1.metric("Total Transactions", len(ae.df))
                    info_col2.metric("Date Range", f"{ae.df['Date'].min().date()} to {ae.df['Date'].max().date()}")
                    info_col3.metric("Columns", len(ae.df.columns))
                    
                    st.write("**Column Names:**")
                    st.write(", ".join(ae.df.columns.tolist()))
        
        except ImportError as e:
            st.error(f"Failed to import AnalyticsEngine: {e}")
            st.info("Make sure the enhanced analytics engine is saved in src/analytics/engine.py")
        except Exception as e:
            st.error(f"Analytics initialization failed: {e}")
            import traceback
            st.code(traceback.format_exc())


elif tool == "Anomaly Detector":
    st.header("🚨 Advanced Anomaly Detection")
    
    if "df" not in st.session_state:
        st.info("No data available — convert a PDF first.")
    else:
        df = ensure_amount_column(st.session_state['df'])
        
        # Sidebar configuration
        with st.sidebar:
            st.subheader("Detection Settings")
            method = st.selectbox(
                "Detection Method", 
                ["model", "z-score", "absolute_threshold"],
                help="Choose the anomaly detection algorithm"
            )
            
            if method == 'model':
                st.markdown("**Model Parameters**")
                model_path = st.text_input(
                    "Model path", 
                    value="models/anomaly_model.pkl",
                    help="Path to save/load the trained model"
                )
                contamination = st.slider(
                    "Expected anomaly rate (%)", 
                    min_value=1.0, 
                    max_value=10.0, 
                    value=3.0,
                    step=0.5,
                    help="Expected percentage of anomalies in data"
                ) / 100
                
                use_custom_threshold = st.checkbox(
                    "Use custom threshold", 
                    value=False,
                    help="Override automatic threshold with percentile-based cutoff"
                )
                
                threshold_percentile = None
                if use_custom_threshold:
                    threshold_percentile = st.slider(
                        "Threshold percentile",
                        min_value=1,
                        max_value=50,
                        value=5,
                        help="Lower values = more sensitive detection"
                    )

        anomalies = None
        summary = None
        
        # Method: Model-based detection
        if method == 'model':
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                run_detection = st.button("🔍 Run Detection", type="primary", use_container_width=True)
            with col2:
                retrain_model = st.button("🔄 Retrain Model", use_container_width=True)
            with col3:
                show_info = st.button("ℹ️ Info", use_container_width=True)
            
            if show_info:
                st.info("""
                **Model-based Detection** uses Isolation Forest algorithm:
                - Automatically learns normal transaction patterns
                - Identifies outliers based on feature isolation
                - Features: amount, day/time, category, rolling statistics
                - Model is saved and reused for consistency
                """)
            
            # Retrain model
            if retrain_model:
                with st.spinner("Training new model on current data..."):
                    try:
                        import sys
                        from pathlib import Path
                        
                        # Import the enhanced detector
                        sys.path.insert(0, str(Path.cwd() / "scripts"))
                        from scripts.anomaly_detector import retrain_model as retrain_fn
                        
                        metrics = retrain_fn(
                            df, 
                            model_path=str(model_path),
                            contamination=contamination
                        )
                        
                        st.success("✅ Model retrained successfully!")
                        
                        # Display training metrics
                        with st.expander("📊 Training Metrics", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Anomalies Found", metrics['anomaly_count'])
                            with col2:
                                st.metric("Anomaly Rate", f"{metrics['anomaly_percentage']}%")
                            with col3:
                                st.metric("Training Samples", metrics.get('training_samples', 'N/A'))
                            
                            st.json(metrics)
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {e}")
                        st.exception(e)
            
            # Run detection
            if run_detection:
                with st.spinner("Running model-based anomaly detection..."):
                    try:
                        import sys
                        from pathlib import Path
                        
                        # Import the enhanced detector
                        sys.path.insert(0, str(Path.cwd() / "scripts"))
                        from scripts.anomaly_detector import detect_anomalies
                        
                        # Run detection with summary
                        result, summary = detect_anomalies(
                            df, 
                            model_path=str(model_path),
                            train_if_missing=True,
                            contamination=contamination,
                            return_summary=True
                        )
                        
                        # Apply custom threshold if specified
                        if use_custom_threshold and threshold_percentile is not None:
                            from scripts.anomaly_detector import AnomalyDetector
                            detector = AnomalyDetector(model_path=str(model_path))
                            detector.load()
                            result = detector.detect(df, threshold_percentile=threshold_percentile)
                            
                            # Recalculate summary
                            summary = detector.get_anomaly_summary(result)
                        
                        anomalies = result[result['Anomaly'] == 'Suspicious']
                        st.session_state['anomaly_results'] = result
                        st.session_state['anomaly_summary'] = summary
                        
                        st.success(f"✅ Detection complete! Found {len(anomalies)} suspicious transactions.")
                        
                    except Exception as e:
                        st.error(f"❌ Model detection failed: {e}")
                        st.exception(e)
            
            # Display cached results if available
            if 'anomaly_results' in st.session_state:
                anomalies = st.session_state['anomaly_results'][
                    st.session_state['anomaly_results']['Anomaly'] == 'Suspicious'
                ]
                summary = st.session_state.get('anomaly_summary')

        # Method: Z-score
        elif method == 'z-score':
            thresh = st.slider(
                'Z-score threshold', 
                min_value=2.0, 
                max_value=6.0, 
                value=3.0,
                step=0.1,
                help="Number of standard deviations from mean"
            )
            
            if st.button("🔍 Run Z-score Detection", type="primary"):
                if 'Amount' in df.columns:
                    with st.spinner("Computing z-scores..."):
                        mean_amt = df['Amount'].mean()
                        std_amt = df['Amount'].std()
                        
                        if std_amt == 0:
                            st.warning("⚠️ Standard deviation is zero. All amounts are identical.")
                            anomalies = pd.DataFrame()
                        else:
                            df_temp = df.copy()
                            df_temp['z_score'] = (df_temp['Amount'] - mean_amt) / std_amt
                            df_temp['Anomaly'] = df_temp['z_score'].abs().apply(
                                lambda x: 'Suspicious' if x > thresh else 'Normal'
                            )
                            df_temp['Anomaly_Score'] = -df_temp['z_score'].abs()  # Negative for consistency
                            
                            anomalies = df_temp[df_temp['Anomaly'] == 'Suspicious']
                            
                            st.session_state['anomaly_results'] = df_temp
                            st.success(f"✅ Found {len(anomalies)} anomalies using z-score method.")
                else:
                    st.error("❌ No 'Amount' column found for z-score method.")
            
            # Display cached results
            if 'anomaly_results' in st.session_state:
                anomalies = st.session_state['anomaly_results'][
                    st.session_state['anomaly_results']['Anomaly'] == 'Suspicious'
                ]

        # Method: Absolute threshold
        else:  # absolute_threshold
            thresh = st.number_input(
                'Absolute amount threshold', 
                min_value=0.0,
                value=10000.0,
                step=1000.0,
                help="Flag transactions above this amount"
            )
            
            if st.button("🔍 Run Threshold Detection", type="primary"):
                if 'Amount' in df.columns:
                    with st.spinner("Applying threshold..."):
                        df_temp = df.copy()
                        df_temp['Anomaly'] = df_temp['Amount'].abs().apply(
                            lambda x: 'Suspicious' if x > thresh else 'Normal'
                        )
                        df_temp['Anomaly_Score'] = -df_temp['Amount'].abs()  # Negative for consistency
                        
                        anomalies = df_temp[df_temp['Anomaly'] == 'Suspicious']
                        
                        st.session_state['anomaly_results'] = df_temp
                        st.success(f"✅ Found {len(anomalies)} anomalies above threshold.")
                else:
                    st.error("❌ No 'Amount' column found for absolute threshold method.")
            
            # Display cached results
            if 'anomaly_results' in st.session_state:
                anomalies = st.session_state['anomaly_results'][
                    st.session_state['anomaly_results']['Anomaly'] == 'Suspicious'
                ]

        # ===================================================================
        # DISPLAY RESULTS
        # ===================================================================
        
        st.markdown("---")
        
        # Summary metrics (for model-based detection)
        if summary is not None:
            st.subheader("📊 Detection Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Transactions", 
                    f"{summary['total_transactions']:,}"
                )
            with col2:
                st.metric(
                    "Suspicious", 
                    f"{summary['suspicious_count']:,}",
                    delta=f"{summary['suspicious_percentage']}%"
                )
            with col3:
                st.metric(
                    "Avg Suspicious Amount",
                    f"₹{summary['avg_suspicious_amount']:,.2f}"
                )
            with col4:
                st.metric(
                    "Total Suspicious Amount",
                    f"₹{summary['total_suspicious_amount']:,.2f}"
                )
            
            # Risk breakdown
            if summary.get('risk_breakdown'):
                st.markdown("**Risk Level Breakdown:**")
                risk_cols = st.columns(len(summary['risk_breakdown']))
                for idx, (risk_level, count) in enumerate(summary['risk_breakdown'].items()):
                    with risk_cols[idx]:
                        color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk_level, "⚪")
                        st.metric(f"{color} {risk_level}", count)

        # Display anomalies table
        st.subheader('🔍 Suspicious Transactions')
        
        if anomalies is None:
            st.info('👆 Choose a detection method and click the button to find anomalies.')
        elif len(anomalies) == 0:
            st.success('✅ No anomalies detected! All transactions appear normal.')
        else:
            # Filter controls
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f'Found **{len(anomalies)}** suspicious transactions')
            with col2:
                sort_by = st.selectbox(
                    "Sort by",
                    ['Anomaly_Score', 'Amount', 'Date'] if 'Date' in anomalies.columns else ['Anomaly_Score', 'Amount'],
                    help="Sort suspicious transactions"
                )
            
            # Sort and display
            display_anomalies = anomalies.sort_values(sort_by, ascending=False)
            
            # Add color coding for risk levels if available
            if 'Risk_Level' in display_anomalies.columns:
                def highlight_risk(row):
                    if row['Risk_Level'] == 'High':
                        return ['background-color: #ffcccc'] * len(row)
                    elif row['Risk_Level'] == 'Medium':
                        return ['background-color: #fff4cc'] * len(row)
                    else:
                        return [''] * len(row)
                
                st.dataframe(
                    display_anomalies.style.apply(highlight_risk, axis=1),
                    use_container_width=True,
                    height=400
                )
            else:
                st.dataframe(display_anomalies, use_container_width=True, height=400)
            
            # Download options
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = display_anomalies.to_csv(index=False)
                st.download_button(
                    label="📥 Download Suspicious Transactions (CSV)",
                    data=csv,
                    file_name=f"anomalies_{method}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                if 'anomaly_results' in st.session_state:
                    full_csv = st.session_state['anomaly_results'].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results (CSV)",
                        data=full_csv,
                        file_name=f"full_results_{method}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Visualization
            if len(anomalies) > 0 and 'Amount' in anomalies.columns:
                st.subheader("📈 Anomaly Distribution")
                
                tab1, tab2 = st.tabs(["Amount Distribution", "Time Series"])
                
                with tab1:
                    import plotly.express as px
                    
                    fig = px.histogram(
                        anomalies,
                        x='Amount',
                        nbins=30,
                        title="Distribution of Suspicious Transaction Amounts",
                        labels={'Amount': 'Transaction Amount', 'count': 'Frequency'},
                        color_discrete_sequence=['#ff6b6b']
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    if 'Date' in anomalies.columns:
                        anomalies_sorted = anomalies.sort_values('Date')
                        fig = px.scatter(
                            anomalies_sorted,
                            x='Date',
                            y='Amount',
                            title="Suspicious Transactions Over Time",
                            labels={'Date': 'Date', 'Amount': 'Transaction Amount'},
                            color='Risk_Level' if 'Risk_Level' in anomalies.columns else None,
                            color_discrete_map={'High': '#ff0000', 'Medium': '#ffa500', 'Low': '#ffff00'} if 'Risk_Level' in anomalies.columns else None,
                            hover_data=['Anomaly_Score'] if 'Anomaly_Score' in anomalies.columns else None
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Date column not available for time series visualization.")


elif tool == "LLM Advisor":
    st.header("🤖 LLM Advisor (AI-powered finance insights)")
    prompt = st.text_area("Ask a finance question about the statement (e.g., 'Where did I spend most?')", value="Analyze my spending and provide recommendations.")
    llm_script = Path('scripts') / '04_llm_advisor.py'
    
    if "df" not in st.session_state:
        st.info("No data available — convert a PDF first.")
    else:
        if st.button("Get advice"):
            with st.spinner('Analyzing statement and generating advice...'):
                try:
                    # Pass the actual DataFrame from session state to the answer function
                    df_for_advisor = ensure_amount_column(st.session_state['df'])
                    
                    if llm_script.exists():
                        try:
                            # Call answer with prompt and dataframe
                            res = run_script(str(llm_script), 'answer', prompt, df_for_advisor)
                            
                            # Display result
                            if isinstance(res, dict):
                                st.subheader("💡 Advice")
                                
                                # Show raw response
                                if res.get('raw'):
                                    st.markdown(res['raw'])
                                
                                # If parsed JSON available, show structured view
                                if res.get('parsed'):
                                    parsed = res['parsed']
                                    if 'summary' in parsed:
                                        st.subheader("Summary")
                                        st.write(parsed['summary'])
                                    if 'recommendations' in parsed:
                                        st.subheader("Recommendations")
                                        for i, rec in enumerate(parsed['recommendations'], 1):
                                            st.write(f"{i}. {rec}")
                                    if 'warnings' in parsed:
                                        st.subheader("⚠️ Warnings")
                                        for w in parsed['warnings']:
                                            st.warning(w)
                                    if 'habit_tip' in parsed:
                                        st.subheader("💪 Habit Tip")
                                        st.write(parsed['habit_tip'])
                            else:
                                st.write(res)
                        except Exception as e:
                            st.error(f"Script execution error: {e}")
                    else:
                        # Fallback: direct import and call
                        try:
                            import importlib.util
                            spec = importlib.util.spec_from_file_location("llm_advisor_module", str(llm_script))
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            res = mod.answer(prompt, df_for_advisor)
                            st.subheader("💡 Advice")
                            if isinstance(res, dict) and res.get('raw'):
                                st.markdown(res['raw'])
                            else:
                                st.write(res)
                        except Exception as e:
                            st.error(f"Advisor error: {e}")
                            
                except Exception as e:
                    st.error(f'Advisor failed: {e}')


elif tool == "Training":
    st.header("🏋️ Training Utilities")
    train_folder = Path('scripts') / 'training'
    st.write('Training scripts folder:', train_folder)
    if train_folder.exists():
        scripts = sorted([p.name for p in train_folder.glob('*.py')])
        choice = st.selectbox('Choose training script', scripts)
        if st.button('Run selected training script'):
            script_path = train_folder / choice
            with st.spinner(f'Running {choice}...'):
                try:
                    run_script(str(script_path))
                    st.success(f'{choice} executed (check console/logs).')
                except Exception as e:
                    st.error(f'Training script failed: {e}')
    else:
        st.info('No training scripts folder found at scripts/training')

st.sidebar.markdown('---')
st.sidebar.write('Project: Finance-Advisor')

