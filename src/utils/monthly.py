import pandas as pd


def generate_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a monthly summary of transactions.
    
    Args:
        df: DataFrame with columns ['Date', 'Debit', 'Credit', 'Details', ...]
        
    Returns:
        DataFrame with monthly aggregated summary
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
    df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
    df['Amount'] = df['Credit'] - df['Debit']

    df['Type'] = df.apply(
        lambda x: "Income" if x['Credit'] > 0 else ("Expense" if x['Debit'] > 0 else "Neutral"),
        axis=1
    )

    df['Month'] = df['Date'].dt.to_period('M')

    monthly_summary = df.groupby('Month').agg({
        'Amount': ['sum', 'count'],
        'Type': lambda x: x.value_counts().to_dict()
    }).reset_index()

    monthly_summary.columns = ['Month', 'Total_Amount', 'Transaction_Count', 'Type_Distribution']
    
    # Convert Period to string for better CSV export
    monthly_summary['Month'] = monthly_summary['Month'].astype(str)

    return monthly_summary
