import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path


class AnomalyDetector:

    def __init__(self, model_path: str = "models/anomaly_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None

    # ---------------------------------------------------------
    # PREPARE FEATURES (robust to different column names / missing data)
    # ---------------------------------------------------------
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Date ---
        # Accept common date column names and coerce to datetime
        date_col = next((c for c in df.columns if c.lower() in ("date", "transaction_date", "posted_date")), None)
        if date_col is not None:
            df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            # try existing 'Date' if present, else create NaT column
            df["Date"] = pd.to_datetime(df.get("Date", pd.NaT), errors="coerce")

        df["DayOfWeek"] = df["Date"].dt.dayofweek.fillna(0).astype(int)

        # --- Hour ---
        # Look for a time column (case-insensitive) or extract from datetime
        time_col = next((c for c in df.columns if c.lower() in ("time", "transaction_time")), None)
        if time_col:
            df["Hour"] = pd.to_datetime(df[time_col], errors="coerce").dt.hour.fillna(0).astype(int)
        else:
            df["Hour"] = df["Date"].dt.hour.fillna(0).astype(int)

        # --- Amount ---
        # Support common formats: single 'Amount' column or separate 'Debit'/'Credit'
        if "Amount" not in df.columns:
            if "Debit" in df.columns and "Credit" in df.columns:
                df["Debit"] = pd.to_numeric(df["Debit"], errors="coerce").fillna(0)
                df["Credit"] = pd.to_numeric(df["Credit"], errors="coerce").fillna(0)
                df["Amount"] = df["Credit"] - df["Debit"]
            else:
                alt_amt = next((c for c in df.columns if c.lower() in ("value", "amt", "transaction_amount")), None)
                if alt_amt:
                    df["Amount"] = pd.to_numeric(df[alt_amt], errors="coerce").fillna(0)
                else:
                    # fallback to zeros — better than raising KeyError
                    df["Amount"] = 0.0
        else:
            df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)

        # --- Category encoding ---
        cat_col = None
        for candidate in ("Predicted_Category", "Category", "Description"):
            if candidate in df.columns:
                cat_col = candidate
                break

        if cat_col:
            df["Category_Code"] = df[cat_col].astype("category").cat.codes.fillna(0).astype(int)
        else:
            df["Category_Code"] = 0

        features = df[["Amount", "DayOfWeek", "Hour", "Category_Code"]].fillna(0)
        return features

    # ---------------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------------
    def train(self, df: pd.DataFrame):
        features = self._prepare_features(df)
        if features.shape[0] < 3:
            raise ValueError("Not enough rows to train anomaly detector; need at least 3 rows.")

        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.03,  # default ~3% anomalies
            random_state=42,
        )

        # fit accepts DataFrame directly
        self.model.fit(features)
        print("Anomaly model trained successfully!")

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------
    def save(self):
        if self.model is None:
            raise RuntimeError("No model to save. Train or load a model first.")

        # Ensure parent directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(self.model_path))
        print(f"Anomaly model saved to {self.model_path}")

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------
    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(str(self.model_path))
        return self.model

    # ---------------------------------------------------------
    # PREDICT ANOMALIES
    # ---------------------------------------------------------
    def detect(self, df: pd.DataFrame):
        # If model missing, attempt to load — caller may handle training fallback
        if self.model is None:
            try:
                self.load()
            except FileNotFoundError:
                raise RuntimeError("No trained model available. Load or train a model first.")

        features = self._prepare_features(df)

        preds = self.model.predict(features)
        scores = self.model.decision_function(features)

        out = df.copy()
        out["Anomaly"] = preds
        out["Anomaly_Score"] = scores

        # In IsolationForest: -1 = anomaly, 1 = normal
        out["Anomaly"] = out["Anomaly"].map({-1: "Suspicious", 1: "Normal"})
        return out


# Module-level helper
def detect_anomalies(df: pd.DataFrame, model_path: str = "models/anomaly_model.pkl", train_if_missing: bool = True) -> pd.DataFrame:
    """Wrapper to detect anomalies using the AnomalyDetector.

    If a saved model exists it will be loaded. If not and `train_if_missing` is True,
    the function will train a fresh model on `df`, save it, and then run detection.

    Returns a DataFrame with added `Anomaly` and `Anomaly_Score` columns.
    """
    det = AnomalyDetector(model_path=model_path)
    try:
        det.load()
    except FileNotFoundError:
        if train_if_missing:
            det.train(df)
            # ensure model directory exists and save
            det.save()
        else:
            raise

    return det.detect(df)