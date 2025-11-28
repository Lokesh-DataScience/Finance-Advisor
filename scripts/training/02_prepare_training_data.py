import pandas as pd

def build_training_dataset():
    """
    Helps user create labeled dataset from raw CSV.
    """
    df = pd.read_csv("statement.csv")

    # Keep only necessary columns
    df = df[["Description", "Amount"]]

    # Add empty label column for manual annotation
    df["Category"] = ""

    df.to_csv("training_template.csv", index=False)
    print("Template created: training_template.csv")
