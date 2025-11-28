import pandas as pd


def generate_weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a weekly summary of transactions.
    
    Args:
        df: DataFrame with columns ['Date', 'Debit', 'Credit', 'Details', ...]
        
    Returns:
        DataFrame with weekly aggregated summary
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Parse date
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Create Amount column (Credit - Debit)
    df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
    df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
    df['Amount'] = df['Credit'] - df['Debit']

    # Create Type column
    df['Type'] = df.apply(
        lambda x: "Income" if x['Credit'] > 0 else ("Expense" if x['Debit'] > 0 else "Neutral"),
        axis=1
    )

    # Add week number
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.isocalendar().year

    # Group by Year and Week
    weekly_summary = df.groupby(['Year', 'Week']).agg({
        'Amount': ['sum', 'count'],
        'Type': lambda x: x.value_counts().to_dict()
    }).reset_index()

    weekly_summary.columns = ['Year', 'Week', 'Total_Amount', 'Transaction_Count', 'Type_Distribution']

    return weekly_summary
