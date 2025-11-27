import pdfplumber
import pandas as pd
import glob

pdf_path = "..//data//unlocked.pdf"
output_folder = "..//data//output_tables"

import os
os.makedirs(output_folder, exist_ok=True)

all_tables = []

with pdfplumber.open(pdf_path) as pdf:
    for page_number, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables()

        # Some pages may contain multiple tables
        for i, table in enumerate(tables):
            df = pd.DataFrame(table)
            csv_name = f"{output_folder}/table_page{page_number}_{i+1}.csv"
            df.to_csv(csv_name, index=False)
            print(f"Saved: {csv_name}")
            all_tables.append(df)

print("\nExtraction complete!")

files = glob.glob("..//data//output_tables/*.csv")

cleaned_tables = []

for f in files:
    df = pd.read_csv(f)

    # Standard header format
    expected_columns = ["Date", "Details", "Ref", "Debit", "Credit", "Balance"]

    # Rename first row as header if needed
    if list(df.columns) != expected_columns:
        df.columns = df.iloc[0]   # first row becomes header
        df = df[1:]               # drop first row

    # Drop any rows that exactly match header names
    df = df[df.iloc[:, 0] != "Date"]  

    cleaned_tables.append(df)

# Combine clean tables
final_df = pd.concat(cleaned_tables, ignore_index=True)

# Save
final_df.to_csv("..//data//bank_statement_clean.csv", index=False)

print("✔ Cleaned CSV saved!")