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
    st.header("📊 Analytics & Summaries")
    if "df" not in st.session_state:
        st.info("No DataFrame found in session. Please use the PDF Extractor to upload/convert a PDF first.")
    else:
        df = st.session_state['df']
        st.subheader("Preview")
        st.dataframe(df.head(200))

        st.subheader("Quick Summaries")
        if st.button("Show weekly summary"):
            try:
                summary = generate_weekly_summary(df)
                st.dataframe(summary)
                st.download_button("Download weekly CSV", data=summary.to_csv(index=False), file_name="weekly_summary.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Failed to compute weekly summary: {e}")

        if st.button("Show monthly summary"):
            try:
                summary = generate_monthly_summary(df)
                st.dataframe(summary)
                st.download_button("Download monthly CSV", data=summary.to_csv(index=False), file_name="monthly_summary.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Failed to compute monthly summary: {e}")

        st.subheader("Basic Analytics")
        if st.button("Show totals and top merchants"):
            try:
                totals = {
                    'total_debit': df['Debit'].sum(),
                    'total_credit': df['Credit'].sum(),
                    'net': df['Credit'].sum() - df['Debit'].sum()
                }
                st.write(totals)
                if 'Details' in df.columns:
                    top = df.groupby('Details')['Amount'].sum().abs().sort_values(ascending=False).head(20)
                    st.subheader('Top merchants / descriptions by absolute amount')
                    st.dataframe(top)
            except Exception as e:
                st.error(f"Analytics failed: {e}")


elif tool == "Anomaly Detector":
    st.header("🚨 Simple Anomaly Detection")
    
    if "df" not in st.session_state:
        st.info("No data available — convert a PDF first.")
    else:
        
        df = ensure_amount_column(st.session_state['df'])
        method = st.selectbox("Method", ["model", "z-score", "absolute_threshold"])

        anomalies = None
        if method == 'model':
            st.info("Using saved IsolationForest model (or will train if missing).")
            model_path = st.text_input("Model path", value="models/anomaly_model.pkl")
            if st.button("Run model-based detection"):
                with st.spinner("Running model-based anomaly detection..."):
                    try:
                        # Use the script wrapper function if available
                        try:
                            res = run_script("scripts/03_anomaly_detector.py", "detect_anomalies", df, str(model_path), True)
                            # script wrapper returns a DataFrame
                            if isinstance(res, pd.DataFrame):
                                anomalies = res
                            elif isinstance(res, dict) and "df" in res:
                                anomalies = res["df"]
                            else:
                                # If run_script executed module without returning, try importing directly
                                import importlib.util
                                spec = importlib.util.spec_from_file_location("anomaly_module", str(Path.cwd() / "scripts" / "03_anomaly_detector.py"))
                                mod = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(mod)
                                anomalies = mod.detect_anomalies(df, model_path=str(model_path), train_if_missing=True)
                        except Exception:
                            # fallback: attempt to run module in new namespace (train & detect)
                            ns = run_script("scripts/03_anomaly_detector.py")
                            # If module registered a wrapper in globals, try to call it
                            try:
                                anomalies = ns.get('detect_anomalies', lambda *a, **k: None)(df, model_path=str(model_path), train_if_missing=True)
                            except Exception:
                                anomalies = None

                    except Exception as e:
                        st.error(f"Model detection failed: {e}")

        elif method == 'z-score':
            thresh = st.slider('Z-score threshold', 2.0, 6.0, 3.0)
            if 'Amount' in df.columns:
                df['z'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
                anomalies = df[df['z'].abs() > thresh]
            else:
                st.error("No 'Amount' column found for z-score method.")

        else:  # absolute_threshold
            thresh = st.number_input('Absolute amount threshold', value=10000.0)
            if 'Amount' in df.columns:
                anomalies = df[df['Amount'].abs() > thresh]
            else:
                st.error("No 'Amount' column found for absolute threshold method.")

        st.subheader('Anomalies')
        if anomalies is None:
            st.write('No anomalies computed yet — choose a method and run detection.')
        else:
            st.write(f'Found {len(anomalies)} anomalous transactions')
            st.dataframe(anomalies)


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

