import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from src.features.weekly import generate_weekly_summary
from src.features.monthly import generate_monthly_summary


class AnalyticsEngine:
    """Enhanced Analytics Engine for comprehensive financial analysis."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.preprocess()
        self._cache = {}  # Cache for expensive computations

    # -----------------------------------------------------
    # ENHANCED PREPROCESSING
    # -----------------------------------------------------
    def preprocess(self):
        """Ensure date parsing and add comprehensive features."""
        # Standardize date column
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        
        # Remove rows with invalid dates
        self.df = self.df.dropna(subset=["Date"])
        
        # Sort by date for time-series analysis
        self.df = self.df.sort_values("Date").reset_index(drop=True)

        # Time-based features
        self.df["Year"] = self.df["Date"].dt.year
        self.df["Month"] = self.df["Date"].dt.to_period("M").astype(str)
        self.df["Quarter"] = self.df["Date"].dt.to_period("Q").astype(str)
        self.df["Day"] = self.df["Date"].dt.day
        self.df["Weekday"] = self.df["Date"].dt.day_name()
        self.df["Week"] = self.df["Date"].dt.isocalendar().week
        self.df["DayOfYear"] = self.df["Date"].dt.dayofyear
        self.df["IsWeekend"] = self.df["Date"].dt.dayofweek.isin([5, 6])
        self.df["MonthName"] = self.df["Date"].dt.month_name()

        # Ensure Amount column exists (handle various formats)
        if "Amount" not in self.df.columns:
            if "Debit" in self.df.columns and "Credit" in self.df.columns:
                self.df["Debit"] = pd.to_numeric(self.df["Debit"], errors="coerce").fillna(0)
                self.df["Credit"] = pd.to_numeric(self.df["Credit"], errors="coerce").fillna(0)
                self.df["Amount"] = self.df["Credit"] - self.df["Debit"]
            elif "Credit" in self.df.columns:
                self.df["Amount"] = pd.to_numeric(self.df["Credit"], errors="coerce").fillna(0)
            elif "Debit" in self.df.columns:
                self.df["Amount"] = -pd.to_numeric(self.df["Debit"], errors="coerce").fillna(0)
            else:
                # Fallback: create zero Amount column
                self.df["Amount"] = 0.0
        
        # Ensure Amount is numeric
        self.df["Amount"] = pd.to_numeric(self.df["Amount"], errors="coerce")
        self.df = self.df.dropna(subset=["Amount"])

        # Transaction classification
        self.df["Transaction_Type"] = np.where(
            self.df["Amount"] < 0, "Outflow", "Inflow"
        )
        self.df["AbsAmount"] = self.df["Amount"].abs()
        
        # Transaction size classification
        self.df["TransactionSize"] = pd.cut(
            self.df["AbsAmount"],
            bins=[0, 50, 200, 1000, np.inf],
            labels=["Small", "Medium", "Large", "Very Large"]
        )

    # -----------------------------------------------------
    # CORE COMPUTATIONS
    # -----------------------------------------------------
    def get_monthly_inflow_outflow(self) -> pd.DataFrame:
        """Get monthly inflow and outflow with net calculations."""
        grouped = (
            self.df.groupby(["Month", "Transaction_Type"])["Amount"]
            .sum()
            .reset_index()
        )
        
        # Add net flow per month
        net_flow = (
            self.df.groupby("Month")["Amount"]
            .sum()
            .reset_index()
            .rename(columns={"Amount": "NetFlow"})
        )
        
        return grouped.merge(net_flow, on="Month", how="left")

    def get_category_totals(self) -> Optional[pd.DataFrame]:
        """Get spending by category with additional metrics."""
        if "Category" not in self.df.columns:
            return None
        
        cat_stats = self.df.groupby("Category").agg({
            "Amount": ["sum", "mean", "count", "std"]
        }).reset_index()
        
        cat_stats.columns = ["Category", "Total", "Average", "Count", "StdDev"]
        cat_stats["Percentage"] = (
            cat_stats["Total"] / cat_stats["Total"].sum() * 100
        )
        
        return cat_stats.sort_values("Total", ascending=False)

    def get_daily_trends(self) -> pd.DataFrame:
        """Get daily spending trends with rolling averages."""
        daily = (
            self.df.groupby("Date")["Amount"]
            .sum()
            .reset_index()
        )
        
        # Add rolling averages
        daily["MA7"] = daily["Amount"].rolling(window=7, min_periods=1).mean()
        daily["MA30"] = daily["Amount"].rolling(window=30, min_periods=1).mean()
        
        return daily

    def get_monthly_savings(self) -> pd.DataFrame:
        """Calculate monthly savings with cumulative tracking."""
        grouped = (
            self.df.groupby(["Month", "Transaction_Type"])["Amount"]
            .sum()
            .reset_index()
        )

        inflow = grouped[grouped["Transaction_Type"] == "Inflow"].set_index("Month")["Amount"]
        outflow = grouped[grouped["Transaction_Type"] == "Outflow"].set_index("Month")["Amount"]

        savings_df = pd.DataFrame({
            "Inflow": inflow,
            "Outflow": outflow.abs(),
            "Savings": inflow - outflow
        }).fillna(0).reset_index()
        
        # Add cumulative savings
        savings_df["CumulativeSavings"] = savings_df["Savings"].cumsum()
        savings_df["SavingsRate"] = (
            savings_df["Savings"] / savings_df["Inflow"] * 100
        ).fillna(0)
        
        return savings_df

    def get_month_over_month_change(self) -> pd.DataFrame:
        """Calculate month-over-month changes for spending and income."""
        spending = (
            self.df[self.df["Transaction_Type"] == "Outflow"]
            .groupby("Month")["Amount"]
            .sum()
            .abs()
        )
        
        income = (
            self.df[self.df["Transaction_Type"] == "Inflow"]
            .groupby("Month")["Amount"]
            .sum()
        )
        
        result = pd.DataFrame({
            "Spending": spending,
            "Income": income,
            "SpendingMoM": spending.pct_change() * 100,
            "IncomeMoM": income.pct_change() * 100
        }).fillna(0).reset_index()
        
        return result

    def get_top_merchants(self, n: int = 5, transaction_type: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get top merchants with frequency and average transaction."""
        if "Description" not in self.df.columns:
            return None

        df_filtered = self.df.copy()
        if transaction_type:
            df_filtered = df_filtered[df_filtered["Transaction_Type"] == transaction_type]

        merchants = df_filtered.groupby("Description").agg({
            "Amount": ["sum", "mean", "count"]
        }).reset_index()
        
        merchants.columns = ["Merchant", "Total", "Average", "Frequency"]
        merchants = merchants.sort_values("Total", ascending=False).head(n)
        
        return merchants

    def get_weekday_heatmap_data(self) -> pd.DataFrame:
        """Enhanced heatmap with proper weekday ordering."""
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        heatmap_data = (
            self.df.pivot_table(
                index="Weekday",
                columns="Week",
                values="AbsAmount",
                aggfunc="sum"
            )
            .fillna(0)
            .reindex(weekday_order)
        )
        
        return heatmap_data

    # -----------------------------------------------------
    # ADVANCED ANALYTICS
    # -----------------------------------------------------
    def get_spending_patterns(self) -> Dict:
        """Analyze spending patterns across different dimensions."""
        patterns = {
            "weekday_avg": self.df.groupby("Weekday")["AbsAmount"].mean().to_dict(),
            "weekend_vs_weekday": {
                "Weekend": self.df[self.df["IsWeekend"]]["AbsAmount"].sum(),
                "Weekday": self.df[~self.df["IsWeekend"]]["AbsAmount"].sum()
            },
            "by_size": self.df.groupby("TransactionSize")["Amount"].count().to_dict(),
            "peak_spending_day": self.df.groupby("Weekday")["AbsAmount"].sum().idxmax()
        }
        
        return patterns

    def get_quarterly_summary(self) -> pd.DataFrame:
        """Get quarterly financial summary."""
        quarterly = self.df.groupby(["Quarter", "Transaction_Type"]).agg({
            "Amount": "sum",
            "AbsAmount": ["mean", "count"]
        }).reset_index()
        
        quarterly.columns = ["Quarter", "Type", "Total", "Average", "Count"]
        
        return quarterly

    def get_anomalies(self, threshold: float = 2.5) -> pd.DataFrame:
        """Detect unusual transactions using statistical methods."""
        outflows = self.df[self.df["Transaction_Type"] == "Outflow"].copy()
        
        mean_spend = outflows["AbsAmount"].mean()
        std_spend = outflows["AbsAmount"].std()
        
        outflows["ZScore"] = (outflows["AbsAmount"] - mean_spend) / std_spend
        anomalies = outflows[outflows["ZScore"].abs() > threshold]
        
        return anomalies[["Date", "Description", "Amount", "ZScore"]].sort_values("ZScore", ascending=False)

    def get_budget_analysis(self, monthly_budget: float) -> pd.DataFrame:
        """Analyze spending against budget."""
        monthly_spend = (
            self.df[self.df["Transaction_Type"] == "Outflow"]
            .groupby("Month")["AbsAmount"]
            .sum()
            .reset_index()
        )
        
        monthly_spend["Budget"] = monthly_budget
        monthly_spend["Variance"] = monthly_budget - monthly_spend["AbsAmount"]
        monthly_spend["VariancePercent"] = (monthly_spend["Variance"] / monthly_budget) * 100
        monthly_spend["Status"] = monthly_spend["Variance"].apply(
            lambda x: "Under Budget" if x >= 0 else "Over Budget"
        )
        
        return monthly_spend

    def get_recurring_transactions(self, similarity_threshold: int = 3) -> pd.DataFrame:
        """Identify potentially recurring transactions."""
        if "Description" not in self.df.columns:
            return pd.DataFrame()
        
        recurring = (
            self.df.groupby("Description")
            .agg({
                "Amount": ["count", "mean", "std"],
                "Date": ["min", "max"]
            })
            .reset_index()
        )
        
        recurring.columns = ["Description", "Frequency", "AvgAmount", "StdAmount", "FirstDate", "LastDate"]
        recurring = recurring[recurring["Frequency"] >= similarity_threshold]
        recurring["DaysBetween"] = (recurring["LastDate"] - recurring["FirstDate"]).dt.days / recurring["Frequency"]
        
        return recurring.sort_values("Frequency", ascending=False)

    def get_cash_flow_forecast(self, periods: int = 3) -> pd.DataFrame:
        """Simple forecast based on historical averages."""
        recent_months = self.df["Month"].unique()[-6:]  # Last 6 months
        recent_df = self.df[self.df["Month"].isin(recent_months)]
        
        avg_inflow = recent_df[recent_df["Transaction_Type"] == "Inflow"].groupby("Month")["Amount"].sum().mean()
        avg_outflow = recent_df[recent_df["Transaction_Type"] == "Outflow"].groupby("Month")["Amount"].sum().mean()
        
        last_date = self.df["Date"].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='ME')
        
        forecast = pd.DataFrame({
            "Month": forecast_dates.to_period("M").astype(str),
            "ForecastInflow": avg_inflow,
            "ForecastOutflow": avg_outflow,
            "ForecastSavings": avg_inflow - avg_outflow
        })
        
        return forecast

    # -----------------------------------------------------
    # ENHANCED VISUALIZATIONS
    # -----------------------------------------------------
    def plot_monthly_spending_line(self):
        """Enhanced daily spending with moving averages."""
        daily = self.get_daily_trends()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily["Date"], y=daily["Amount"],
            mode="lines", name="Daily",
            line=dict(color="lightblue", width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=daily["Date"], y=daily["MA7"],
            mode="lines", name="7-Day MA",
            line=dict(color="blue", width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=daily["Date"], y=daily["MA30"],
            mode="lines", name="30-Day MA",
            line=dict(color="darkblue", width=2, dash="dash")
        ))
        
        fig.update_layout(
            title="📈 Daily Spending Trend with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Amount",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_category_pie(self):
        """Enhanced category visualization."""
        cat = self.get_category_totals()
        if cat is not None:
            fig = px.pie(
                cat,
                names="Category",
                values="Total",
                title="🧩 Category-wise Spending Distribution",
                hover_data=["Count", "Average"],
                labels={"Total": "Total Spending", "Count": "Transactions"}
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'Category' column found. Please run Phase 3 ML first.")

    def plot_inflow_outflow_bar(self):
        """Enhanced inflow/outflow with net visualization."""
        data = self.get_monthly_inflow_outflow()
        
        fig = px.bar(
            data,
            x="Month",
            y="Amount",
            color="Transaction_Type",
            barmode="group",
            title="💸 Monthly Inflow vs Outflow",
            color_discrete_map={"Inflow": "green", "Outflow": "red"}
        )
        
        # Add net flow line
        net_data = data.groupby("Month")["NetFlow"].first()
        fig.add_trace(go.Scatter(
            x=net_data.index,
            y=net_data.values,
            mode="lines+markers",
            name="Net Flow",
            line=dict(color="blue", width=3)
        ))
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_heatmap(self):
        """Enhanced heatmap visualization."""
        heatmap_data = self.get_weekday_heatmap_data()
        
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale="Reds",
                hoverongaps=False
            )
        )
        
        fig.update_layout(
            title="🔥 Weekday Spending Heatmap",
            xaxis_title="Week of Year",
            yaxis_title="Day of Week"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_savings_trend(self):
        """Visualize savings over time."""
        savings = self.get_monthly_savings()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=savings["Month"],
            y=savings["Savings"],
            name="Monthly Savings",
            marker_color=savings["Savings"].apply(lambda x: "green" if x >= 0 else "red")
        ))
        
        fig.add_trace(go.Scatter(
            x=savings["Month"],
            y=savings["CumulativeSavings"],
            mode="lines+markers",
            name="Cumulative Savings",
            yaxis="y2",
            line=dict(color="blue", width=3)
        ))
        
        fig.update_layout(
            title="💰 Savings Trend Analysis",
            yaxis=dict(title="Monthly Savings"),
            yaxis2=dict(title="Cumulative Savings", overlaying="y", side="right"),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_anomalies(self):
        """Visualize unusual transactions."""
        anomalies = self.get_anomalies()
        
        if len(anomalies) > 0:
            fig = px.scatter(
                anomalies,
                x="Date",
                y="Amount",
                size="ZScore",
                hover_data=["Description"],
                title="🚨 Unusual Transactions (Anomalies)",
                color="ZScore",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant anomalies detected.")

    # -----------------------------------------------------
    # SUMMARY DASHBOARD
    # -----------------------------------------------------
    def display_summary_metrics(self):
        """Display key financial metrics in columns."""
        col1, col2, col3, col4 = st.columns(4)
        
        total_inflow = self.df[self.df["Transaction_Type"] == "Inflow"]["Amount"].sum()
        total_outflow = self.df[self.df["Transaction_Type"] == "Outflow"]["Amount"].sum()
        net_savings = total_inflow + total_outflow
        avg_daily_spend = self.df[self.df["Transaction_Type"] == "Outflow"]["AbsAmount"].mean()
        
        col1.metric("Total Income", f"₹{total_inflow:,.2f}")
        col2.metric("Total Spending", f"₹{abs(total_outflow):,.2f}")
        col3.metric("Net Savings", f"₹{net_savings:,.2f}", 
                   delta=f"{(net_savings/total_inflow*100):.1f}%" if total_inflow > 0 else "0%")
        col4.metric("Avg Daily Spend", f"₹{avg_daily_spend:,.2f}")


# -----------------------------------------------------
# MODULE-LEVEL HELPER FUNCTIONS
# -----------------------------------------------------
def compute_summaries(df: pd.DataFrame) -> Dict:
    """Compute comprehensive analytics summaries."""
    ae = AnalyticsEngine(df)
    return {
        'monthly_inflow_outflow': ae.get_monthly_inflow_outflow(),
        'category_totals': ae.get_category_totals(),
        'daily_trends': ae.get_daily_trends(),
        'monthly_savings': ae.get_monthly_savings(),
        'top_merchants': ae.get_top_merchants(10),
        'spending_patterns': ae.get_spending_patterns(),
        'quarterly_summary': ae.get_quarterly_summary(),
        'anomalies': ae.get_anomalies(),
        'recurring_transactions': ae.get_recurring_transactions()
    }


def compute_weekly(df: pd.DataFrame):
    """Generate weekly summary."""
    return generate_weekly_summary(df)


def compute_monthly(df: pd.DataFrame):
    """Generate monthly summary."""
    return generate_monthly_summary(df)