import streamlit as st
import pandas as pd
import os
import tempfile
from src.utils import PDFtoCSV, generate_weekly_summary, generate_monthly_summary

# -----------------------------
# AUTO CLEAR CACHE ON REFRESH
# -----------------------------
st.cache_data.clear()
st.cache_resource.clear()

st.set_page_config(page_title="Bank Statement Analyzer", layout="wide")

st.title("📄 Bank Statement Analyzer")

st.write("Upload your **password-protected bank statement PDF**, convert it to CSV, and generate weekly/monthly summaries.")

uploaded_pdf = st.file_uploader("Upload Bank Statement PDF", type=["pdf"])
pdf_password = st.text_input("Enter PDF Password (if locked)", type="password")

if uploaded_pdf:
    st.success("PDF uploaded successfully!")

    # Temporary directory for cache files
    temp_dir = tempfile.mkdtemp()

    pdf_path = os.path.join(temp_dir, "uploaded.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    if st.button("🔓 Convert PDF to CSV"):
        with st.spinner("Converting PDF to CSV…"):
            try:
                # Initialize PDFtoCSV converter
                converter = PDFtoCSV(output_folder=os.path.join(temp_dir, "tables"))
                
                # Convert PDF to DataFrame
                df = converter.convert(pdf_path, password=pdf_password if pdf_password else None)

                st.success("PDF successfully converted to CSV!")
                st.dataframe(df)

                # Download button
                st.download_button(
                    "⬇ Download CSV",
                    data=df.to_csv(index=False),
                    file_name="statement.csv",
                    mime="text/csv"
                )

                # Store df in session
                st.session_state["df"] = df

            except Exception as e:
                st.error(f"Failed to convert: {e}")

# -----------------------------
# WEEKLY & MONTHLY PROCESSING
# -----------------------------
if "df" in st.session_state:

    st.subheader("📆 Generate Weekly / Monthly CSV")

    option = st.selectbox(
        "Choose Summary Type",
        ["Weekly Summary", "Monthly Summary"]
    )

    if st.button("Generate Summary"):
        df = st.session_state["df"]

        try:
            if option == "Weekly Summary":
                summary = generate_weekly_summary(df)
                file_name = "weekly_summary.csv"
            else:
                summary = generate_monthly_summary(df)
                file_name = "monthly_summary.csv"

            st.success(f"{option} generated successfully!")
            st.dataframe(summary)

            st.download_button(
                "⬇ Download Summary CSV",
                data=summary.to_csv(index=False),
                file_name=file_name,
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error generating summary: {e}")

# -----------------------------
# AUTO CLEANUP ON REFRESH
# -----------------------------
st.warning("Refreshing the page will delete cached/generated files automatically.")
