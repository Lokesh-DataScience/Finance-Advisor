import pdfplumber
import pandas as pd
import os


class PDFtoCSV:
    """
    Convert password-protected bank statement PDFs to clean CSV format.
    """

    def __init__(self, output_folder: str = "./data/output_tables"):
        """
        Initialize the PDF converter.
        
        Args:
            output_folder: Path to save intermediate CSV files
        """
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def extract_tables_from_pdf(self, pdf_path: str, password: str = None) -> list:
        """
        Extract all tables from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            password: Password for encrypted PDF (optional)
            
        Returns:
            List of DataFrames containing extracted tables
        """
        all_tables = []

        try:
            with pdfplumber.open(pdf_path, password=password) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()

                    if tables:
                        for i, table in enumerate(tables):
                            df = pd.DataFrame(table)
                            csv_name = os.path.join(
                                self.output_folder,
                                f"table_page{page_number}_{i+1}.csv"
                            )
                            df.to_csv(csv_name, index=False)
                            all_tables.append(df)

            return all_tables

        except Exception as e:
            raise ValueError(f"Error extracting PDF tables: {e}")

    def clean_tables(self, tables: list) -> pd.DataFrame:
        """
        Clean and standardize extracted tables.
        
        Args:
            tables: List of DataFrames to clean
            
        Returns:
            Cleaned and combined DataFrame
        """
        expected_columns = ["Date", "Details", "Ref", "Debit", "Credit", "Balance"]
        cleaned_tables = []

        for df in tables:
            # Rename first row as header if needed
            if list(df.columns) != expected_columns:
                df.columns = df.iloc[0]
                df = df[1:]

            # Drop any rows that exactly match header names
            df = df[df.iloc[:, 0] != "Date"]

            cleaned_tables.append(df)

        # Combine all cleaned tables
        final_df = pd.concat(cleaned_tables, ignore_index=True)
        return final_df

    def convert(self, pdf_path: str, password: str = None) -> pd.DataFrame:
        """
        Main conversion method: extract tables from PDF and return cleaned DataFrame.
        
        Args:
            pdf_path: Path to the PDF file
            password: Password for encrypted PDF (optional)
            
        Returns:
            Cleaned DataFrame with bank statement data
        """
        tables = self.extract_tables_from_pdf(pdf_path, password)
        clean_df = self.clean_tables(tables)
        return clean_df
    