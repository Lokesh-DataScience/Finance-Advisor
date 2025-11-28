import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.features.weekly import generate_weekly_summary
from src.features.monthly import generate_monthly_summary


class AnalyticsEngine:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.preprocess()

    # -----------------------------------------------------
    # BASIC PREPROCESSING
    # -----------------------------------------------------
    def preprocess(self):
        """Ensure date parsing and add useful features."""
        # Standardize date column
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")

        # Create month, day, weekday features
        self.df["Year"] = self.df["Date"].dt.year
        self.df["Month"] = self.df["Date"].dt.to_period("M").astype(str)
        self.df["Day"] = self.df["Date"].dt.day
        self.df["Weekday"] = self.df["Date"].dt.day_name()

        # Ensure Amount is numeric
        self.df["Amount"] = pd.to_numeric(self.df["Amount"], errors="coerce")

        # Auto-detect inflow/outflow
        self.df["Transaction_Type"] = np.where(self.df["Amount"] < 0, "Outflow", "Inflow")

    # -----------------------------------------------------
    # COMPUTATIONS
    # -----------------------------------------------------
    def get_monthly_inflow_outflow(self):
        return (
            self.df.groupby(["Month", "Transaction_Type"])["Amount"]
            .sum()
            .reset_index()
        )

    def get_category_totals(self):
        if "Category" not in self.df.columns:
            return None
        
        return (
            self.df.groupby("Category")["Amount"]
            .sum()
            .reset_index()
        )

    def get_daily_trends(self):
        return (
            self.df.groupby("Date")["Amount"]
            .sum()
            .reset_index()
        )

    def get_monthly_savings(self):
        """Savings = inflow - outflow per month."""
        grouped = (
            self.df.groupby(["Month", "Transaction_Type"])["Amount"]
            .sum()
            .reset_index()
        )

        inflow = grouped[grouped["Transaction_Type"] == "Inflow"].set_index("Month")["Amount"]
        outflow = grouped[grouped["Transaction_Type"] == "Outflow"].set_index("Month")["Amount"]

        savings = (inflow - outflow).fillna(0)
        return savings.reset_index(name="Savings")

    def get_month_over_month_change(self):
        """% change in total spending."""
        monthly_spend = (
            self.df[self.df["Transaction_Type"] == "Outflow"]
            .groupby("Month")["Amount"]
            .sum()
        )

        change = monthly_spend.pct_change().fillna(0) * 100
        return change.reset_index(name="MoM Change %")

    def get_top_merchants(self, n=5):
        if "Description" not in self.df.columns:
            return None

        return (
            self.df.groupby("Description")["Amount"]
            .sum()
            .reset_index()
            .sort_values(by="Amount", ascending=False)
            .head(n)
        )

    def get_weekday_heatmap_data(self):
        """Pivot table for heatmap visualization."""
        heatmap_data = (
            self.df.pivot_table(
                index=self.df["Date"].dt.day_name(),
                columns=self.df["Date"].dt.week,
                values="Amount",
                aggfunc="sum"
            )
            .fillna(0)
        )
        return heatmap_data

    # -----------------------------------------------------
    # VISUALIZATIONS
    # -----------------------------------------------------
    def plot_monthly_spending_line(self):
        daily = self.get_daily_trends()
        fig = px.line(daily, x="Date", y="Amount", title="📈 Daily Spending Trend")
        st.plotly_chart(fig, use_container_width=True)

    def plot_category_pie(self):
        cat = self.get_category_totals()
        if cat is not None:
            fig = px.pie(cat, names="Category", values="Amount", title="🧩 Category-wise Spending")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'Category' column found. Please run Phase 3 ML first.")

    def plot_inflow_outflow_bar(self):
        data = self.get_monthly_inflow_outflow()
        fig = px.bar(
            data,
            x="Month",
            y="Amount",
            color="Transaction_Type",
            barmode="group",
            title="💸 Monthly Inflow vs Outflow"
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_heatmap(self):
        heatmap_data = self.get_weekday_heatmap_data()
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index
            )
        )
        fig.update_layout(title="🔥 Weekday Spending Heatmap")
        st.plotly_chart(fig, use_container_width=True)


# Module-level helper wrappers so the script can be used programmatically
def compute_summaries(df: pd.DataFrame) -> dict:
    ae = AnalyticsEngine(df)
    return {
        'monthly_inflow_outflow': ae.get_monthly_inflow_outflow(),
        'category_totals': ae.get_category_totals(),
        'daily_trends': ae.get_daily_trends(),
        'monthly_savings': ae.get_monthly_savings(),
        'top_merchants': ae.get_top_merchants(10)
    }


def compute_weekly(df: pd.DataFrame):
    return generate_weekly_summary(df)


def compute_monthly(df: pd.DataFrame):
    return generate_monthly_summary(df)
