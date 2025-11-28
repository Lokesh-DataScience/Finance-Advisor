import os
import tempfile
import pandas as pd
import streamlit as st
from src.utils import PDFtoCSV


class PDFExtractor:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pdf_path = os.path.join(self.temp_dir, "uploaded.pdf")
        self.tables_dir = os.path.join(self.temp_dir, "tables")

        # Clear cache on each reload
        st.cache_data.clear()
        st.cache_resource.clear()

    def upload_pdf(self):
        """Handles Streamlit PDF upload UI."""
        st.title("📄 Bank Statement Analyzer")
        st.write("Upload your **password-protected bank statement PDF** to convert it into CSV.")

        uploaded_pdf = st.file_uploader("Upload Bank Statement PDF", type=["pdf"])
        pdf_password = st.text_input("Enter PDF Password (if locked)", type="password")

        if uploaded_pdf:
            st.success("PDF uploaded successfully!")
            with open(self.pdf_path, "wb") as f:
                f.write(uploaded_pdf.read())

        return uploaded_pdf, pdf_password

    def convert_pdf_to_csv(self, uploaded_pdf, pdf_password):
        """Converts uploaded PDF to CSV using PDFtoCSV."""
        if uploaded_pdf is None:
            return None

        if st.button("🔓 Convert PDF to CSV"):
            with st.spinner("Converting PDF to CSV…"):
                try:
                    converter = PDFtoCSV(output_folder=self.tables_dir)
                    df = converter.convert(self.pdf_path, password=pdf_password or None)

                    st.success("PDF successfully converted to CSV!")
                    st.dataframe(df)

                    # CSV download
                    st.download_button(
                        "⬇ Download CSV",
                        data=df.to_csv(index=False),
                        file_name="statement.csv",
                        mime="text/csv"
                    )

                    # Save DataFrame to session
                    st.session_state["df"] = df

                    return df

                except Exception as e:
                    st.error(f"Failed to convert: {e}")
                    return None
