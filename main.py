import streamlit as st
import pandas as pd
import os
import tempfile
import importlib.util
import runpy
import sys
import traceback
from pathlib import Path

from src.utils import PDFtoCSV, generate_weekly_summary, generate_monthly_summary


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
        df = st.session_state['df'].copy()
        method = st.selectbox("Method", ["z-score", "absolute_threshold"])
        if method == 'z-score':
            thresh = st.slider('Z-score threshold', 2.0, 6.0, 3.0)
            df['z'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
            anomalies = df[df['z'].abs() > thresh]
        else:
            thresh = st.number_input('Absolute amount threshold', value=10000.0)
            anomalies = df[df['Amount'].abs() > thresh]

        st.subheader('Anomalies')
        st.write(f'Found {len(anomalies)} anomalous transactions')
        st.dataframe(anomalies)


elif tool == "LLM Advisor":
    st.header("🤖 LLM Advisor (local script or placeholder)")
    prompt = st.text_area("Ask a finance question about the statement (e.g., 'Where did I spend most?')")
    llm_script = Path('scripts') / '04_llm_advisor.py'
    if st.button("Get advice"):
        with st.spinner('Running advisor...'):
            try:
                if llm_script.exists():
                    # try to call a function `answer` or `main`
                    try:
                        res = run_script(str(llm_script), 'answer', prompt, st.session_state.get('df', None))
                        st.write(res)
                    except Exception:
                        run_script(str(llm_script))
                        st.success('Advisor script executed')
                else:
                    # Simple heuristic advisor fallback
                    df = st.session_state.get('df')
                    if df is None:
                        st.info('Upload a statement first or provide more context in the prompt.')
                    else:
                        # simple answer examples
                        top = df.groupby('Details')['Amount'].sum().sort_values().head(5)
                        st.write('Top 5 spending descriptions (by negative net amount):')
                        st.dataframe(top)
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

