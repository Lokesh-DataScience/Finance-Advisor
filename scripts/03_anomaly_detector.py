import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib


class AnomalyDetector:

    def __init__(self, model_path="models/anomaly_model.pkl"):
        self.model_path = model_path
        self.model = None

    # ---------------------------------------------------------
    # PREPARE FEATURES
    # ---------------------------------------------------------
    def _prepare_features(self, df):
        df = df.copy()

        # Convert date column (you may change the column name)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Hour"] = pd.to_datetime(df["time"], errors="coerce").dt.hour.fillna(0)

        # Amount column (ensure numeric)
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)

        # Replace category with numeric encoded (fallback)
        if "Predicted_Category" in df:
            df["Category_Code"] = df["Predicted_Category"].astype("category").cat.codes
        else:
            df["Category_Code"] = 0  # before Phase 3

        return df[["Amount", "DayOfWeek", "Hour", "Category_Code"]]

    # ---------------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------------
    def train(self, df):
        features = self._prepare_features(df)

        # Train Isolation Forest
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.03,   # 3% anomalies
            random_state=42
        )

        self.model.fit(features)
        print("Anomaly model trained successfully!")

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------
    def save(self):
        joblib.dump(self.model, self.model_path)
        print(f"Anomaly model saved to {self.model_path}")

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------
    def load(self):
        self.model = joblib.load(self.model_path)
        return self.model

    # ---------------------------------------------------------
    # PREDICT ANOMALIES
    # ---------------------------------------------------------
    def detect(self, df):
        if self.model is None:
            self.load()

        features = self._prepare_features(df)

        preds = self.model.predict(features)  
        scores = self.model.decision_function(features)

        df = df.copy()
        df["Anomaly"] = preds
        df["Anomaly_Score"] = scores

        # In IsolationForest:
        #   -1  = anomaly
        #    1  = normal
        df["Anomaly"] = df["Anomaly"].map({-1: "Suspicious", 1: "Normal"})

        return df


# Module-level helper
def detect_anomalies(df, model_path: str = "models/anomaly_model.pkl"):
    """Simple wrapper to detect anomalies using the AnomalyDetector class.

    Returns DataFrame with anomaly columns.
    """
    det = AnomalyDetector(model_path=model_path)
    try:
        # attempt to load existing model (if available)
        det.load()
    except Exception:
        # if load fails, train a fresh model
        det.train(df)
        det.save()

    return det.detect(df)