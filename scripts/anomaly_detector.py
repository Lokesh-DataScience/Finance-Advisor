import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Advanced anomaly detection for financial transactions using Isolation Forest.
    
    Features:
    - Robust feature engineering with multiple column name support
    - Feature scaling for improved detection
    - Configurable contamination and model parameters
    - Rolling statistics for temporal patterns
    - Model versioning and metadata tracking
    """

    def __init__(
        self,
        model_path: str = "models/anomaly_model.pkl",
        scaler_path: str = "models/scaler.pkl",
        contamination: float = 0.03,
        n_estimators: int = 200,
        random_state: int = 42
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model = None
        self.scaler = None
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_names = ["Amount", "DayOfWeek", "Hour", "Category_Code", 
                             "Amount_Abs", "Is_Weekend", "Rolling_Mean", "Rolling_Std"]
        self.metadata = {}

    # ---------------------------------------------------------
    # PREPARE FEATURES (Enhanced with additional features)
    # ---------------------------------------------------------
    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Prepare features with robust handling of various data formats.
        
        Args:
            df: Input DataFrame
            is_training: If True, calculate rolling statistics from scratch
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # --- Date Parsing ---
        date_col = self._find_column(df, ["date", "transaction_date", "posted_date", "trans_date"])
        if date_col:
            df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            df["Date"] = pd.to_datetime(df.get("Date", pd.NaT), errors="coerce")

        # Handle missing dates
        if df["Date"].isna().all():
            logger.warning("No valid dates found. Using current date as fallback.")
            df["Date"] = pd.Timestamp.now()

        df["DayOfWeek"] = df["Date"].dt.dayofweek.fillna(0).astype(int)
        df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

        # --- Time/Hour Extraction ---
        time_col = self._find_column(df, ["time", "transaction_time", "trans_time"])
        if time_col:
            df["Hour"] = pd.to_datetime(df[time_col], errors="coerce").dt.hour.fillna(0).astype(int)
        else:
            df["Hour"] = df["Date"].dt.hour.fillna(0).astype(int)

        # --- Amount Handling ---
        df = self._parse_amount(df)
        df["Amount_Abs"] = df["Amount"].abs()

        # --- Category Encoding ---
        df = self._encode_category(df)

        # --- Rolling Statistics (Temporal Features) ---
        if is_training or len(df) > 1:
            df = df.sort_values("Date")
            # Calculate rolling statistics with minimum periods
            df["Rolling_Mean"] = df["Amount"].rolling(window=7, min_periods=1).mean().fillna(df["Amount"].mean())
            df["Rolling_Std"] = df["Amount"].rolling(window=7, min_periods=1).std().fillna(0)
        else:
            # For single predictions, use global statistics if available
            df["Rolling_Mean"] = df["Amount"].mean()
            df["Rolling_Std"] = 0

        # --- Select and Order Features ---
        features = df[self.feature_names].fillna(0)
        
        # Log feature statistics
        if is_training:
            logger.info(f"Feature statistics:\n{features.describe()}")
        
        return features

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column name (case-insensitive)."""
        for col in df.columns:
            if col.lower() in candidates:
                return col
        return None

    def _parse_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse transaction amounts from various formats."""
        if "Amount" not in df.columns:
            if "Debit" in df.columns and "Credit" in df.columns:
                df["Debit"] = pd.to_numeric(df["Debit"], errors="coerce").fillna(0)
                df["Credit"] = pd.to_numeric(df["Credit"], errors="coerce").fillna(0)
                df["Amount"] = df["Credit"] - df["Debit"]
            else:
                alt_amt = self._find_column(df, ["value", "amt", "transaction_amount", "trans_amt"])
                if alt_amt:
                    df["Amount"] = pd.to_numeric(df[alt_amt], errors="coerce").fillna(0)
                else:
                    logger.warning("No amount column found. Using zeros.")
                    df["Amount"] = 0.0
        else:
            df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
        
        return df

    def _encode_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        cat_col = self._find_column(df, ["predicted_category", "category", "description", "merchant"])
        
        if cat_col:
            df["Category_Code"] = df[cat_col].astype("category").cat.codes.fillna(-1).astype(int)
        else:
            df["Category_Code"] = 0
        
        return df

    # ---------------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------------
    def train(self, df: pd.DataFrame, validate: bool = True) -> Dict:
        """
        Train the anomaly detection model with validation.
        
        Args:
            df: Training DataFrame
            validate: If True, perform validation and return metrics
            
        Returns:
            Dictionary with training metrics
        """
        if len(df) < 10:
            raise ValueError(f"Insufficient data: need at least 10 rows, got {len(df)}")

        logger.info(f"Training anomaly detector on {len(df)} transactions...")
        
        features = self._prepare_features(df, is_training=True)
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        # Train Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            max_samples='auto',
            n_jobs=-1
        )
        self.model.fit(features_scaled)

        # Store metadata
        self.metadata = {
            'training_samples': len(df),
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'feature_names': self.feature_names,
            'date_range': (df['Date'].min(), df['Date'].max()) if 'Date' in df.columns else None
        }

        metrics = {}
        if validate:
            metrics = self._validate_model(features_scaled)
        
        logger.info("Training complete!")
        return metrics

    def _validate_model(self, features_scaled: np.ndarray) -> Dict:
        """Validate model performance."""
        predictions = self.model.predict(features_scaled)
        scores = self.model.decision_function(features_scaled)
        
        anomaly_count = np.sum(predictions == -1)
        anomaly_pct = (anomaly_count / len(predictions)) * 100
        
        metrics = {
            'anomaly_count': int(anomaly_count),
            'anomaly_percentage': round(anomaly_pct, 2),
            'score_mean': float(scores.mean()),
            'score_std': float(scores.std()),
            'score_min': float(scores.min()),
            'score_max': float(scores.max())
        }
        
        logger.info(f"Validation: {anomaly_count} anomalies ({anomaly_pct:.1f}%)")
        return metrics

    # ---------------------------------------------------------
    # SAVE/LOAD MODEL
    # ---------------------------------------------------------
    def save(self, metadata: Optional[Dict] = None):
        """Save model, scaler, and metadata."""
        if self.model is None:
            raise RuntimeError("No model to save. Train a model first.")

        # Ensure directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, str(self.model_path))
        if self.scaler is not None:
            joblib.dump(self.scaler, str(self.scaler_path))
        
        # Save metadata
        if metadata:
            self.metadata.update(metadata)
        
        metadata_path = self.model_path.with_suffix('.metadata.pkl')
        joblib.dump(self.metadata, str(metadata_path))
        
        logger.info(f"Model saved to {self.model_path}")

    def load(self):
        """Load model, scaler, and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = joblib.load(str(self.model_path))
        
        # Load scaler if exists
        if self.scaler_path.exists():
            self.scaler = joblib.load(str(self.scaler_path))
        else:
            logger.warning("Scaler not found. Feature scaling may be inconsistent.")
        
        # Load metadata if exists
        metadata_path = self.model_path.with_suffix('.metadata.pkl')
        if metadata_path.exists():
            self.metadata = joblib.load(str(metadata_path))
            logger.info(f"Model loaded (trained on {self.metadata.get('training_samples', 'unknown')} samples)")
        
        return self.model

    # ---------------------------------------------------------
    # DETECT ANOMALIES
    # ---------------------------------------------------------
    def detect(
        self, 
        df: pd.DataFrame, 
        threshold_percentile: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Detect anomalies in transactions.
        
        Args:
            df: Input DataFrame
            threshold_percentile: Custom threshold (0-100). Lower scores = more anomalous.
            
        Returns:
            DataFrame with Anomaly, Anomaly_Score, and Confidence columns
        """
        if self.model is None:
            try:
                self.load()
            except FileNotFoundError:
                raise RuntimeError("No trained model available. Train or load a model first.")

        # Prepare features to compare dimensionality with the loaded model
        features = self._prepare_features(df, is_training=False)

        # If model was trained with a different feature set, raise an informative error
        if hasattr(self.model, "n_features_in_") and features.shape[1] != int(getattr(self.model, "n_features_in_")):
            raise RuntimeError(
                f"Feature mismatch: model expects {int(getattr(self.model, 'n_features_in_'))} "
                f"features but input has {features.shape[1]}.\n"
                "Possible causes: model was trained with a different pipeline or older code. "
                "Retrain the model on the current data (use retrain_model) or set `train_if_missing=True` in detect_anomalies wrapper."
            )
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            logger.warning("No scaler available. Using unscaled features.")
            features_scaled = features.values

        # Predict
        predictions = self.model.predict(features_scaled)
        scores = self.model.decision_function(features_scaled)

        # Prepare output
        out = df.copy()
        out["Anomaly_Score"] = scores
        
        # Apply custom threshold if provided
        if threshold_percentile is not None:
            threshold = np.percentile(scores, threshold_percentile)
            predictions = np.where(scores < threshold, -1, 1)
        
        out["Anomaly"] = pd.Series(predictions).map({-1: "Suspicious", 1: "Normal"})
        
        # Add confidence score (normalized between 0-1)
        score_range = scores.max() - scores.min()
        if score_range > 0:
            out["Confidence"] = ((scores - scores.min()) / score_range)
        else:
            out["Confidence"] = 0.5
        
        # Flag high-confidence anomalies
        suspicious_mask = out["Anomaly"] == "Suspicious"
        out["Risk_Level"] = "Low"
        out.loc[suspicious_mask & (out["Confidence"] < 0.3), "Risk_Level"] = "High"
        out.loc[suspicious_mask & (out["Confidence"] >= 0.3) & (out["Confidence"] < 0.5), "Risk_Level"] = "Medium"
        
        logger.info(f"Detected {suspicious_mask.sum()} suspicious transactions")
        return out

    def get_anomaly_summary(self, df_with_anomalies: pd.DataFrame) -> Dict:
        """Generate summary statistics for detected anomalies."""
        suspicious = df_with_anomalies[df_with_anomalies["Anomaly"] == "Suspicious"]
        
        summary = {
            'total_transactions': len(df_with_anomalies),
            'suspicious_count': len(suspicious),
            'suspicious_percentage': round((len(suspicious) / len(df_with_anomalies)) * 100, 2),
            'risk_breakdown': suspicious['Risk_Level'].value_counts().to_dict() if len(suspicious) > 0 else {},
            'avg_suspicious_amount': float(suspicious['Amount'].mean()) if len(suspicious) > 0 else 0,
            'total_suspicious_amount': float(suspicious['Amount'].sum()) if len(suspicious) > 0 else 0
        }
        
        return summary


# ---------------------------------------------------------
# MODULE-LEVEL HELPER FUNCTIONS
# ---------------------------------------------------------
def detect_anomalies(
    df: pd.DataFrame,
    model_path: str = "models/anomaly_model.pkl",
    train_if_missing: bool = True,
    contamination: float = 0.03,
    return_summary: bool = False
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Convenience function to detect anomalies with automatic model management.
    
    Args:
        df: Input DataFrame
        model_path: Path to saved model
        train_if_missing: Train new model if none exists
        contamination: Expected proportion of anomalies (for training)
        return_summary: If True, return summary statistics
        
    Returns:
        DataFrame with anomaly predictions, and optional summary dict
    """
    detector = AnomalyDetector(model_path=model_path, contamination=contamination)
    
    try:
        detector.load()
        logger.info("Loaded existing model")
    except FileNotFoundError:
        if train_if_missing:
            logger.info("No existing model found. Training new model...")
            detector.train(df)
            detector.save()
        else:
            raise RuntimeError("No model found and train_if_missing=False")

    # After loading, validate that the loaded model agrees with current features.
    try:
        features = detector._prepare_features(df, is_training=False)
        if detector.model is not None and hasattr(detector.model, "n_features_in_"):
            model_n = int(getattr(detector.model, "n_features_in_"))
            if features.shape[1] != model_n:
                msg = (
                    f"Loaded model expects {model_n} features but current data has {features.shape[1]} features."
                    " This typically means the model was trained with a different feature set or code version."
                )
                logger.warning(msg)
                if train_if_missing:
                    logger.info("Retraining model to match current feature set...")
                    detector.train(df)
                    detector.save()
                else:
                    raise RuntimeError(msg + " Set train_if_missing=True to retrain automatically.")
    except Exception as e:
        # If preparation or retrain failed, surface error
        raise RuntimeError(f"Error preparing features or aligning model: {e}")

    result = detector.detect(df)
    
    if return_summary:
        summary = detector.get_anomaly_summary(result)
        return result, summary
    
    return result, None


def retrain_model(
    df: pd.DataFrame,
    model_path: str = "models/anomaly_model.pkl",
    contamination: float = 0.03
) -> Dict:
    """
    Retrain the anomaly detection model on new data.
    
    Args:
        df: Training DataFrame
        model_path: Path to save model
        contamination: Expected proportion of anomalies
        
    Returns:
        Dictionary with training metrics
    """
    detector = AnomalyDetector(model_path=model_path, contamination=contamination)
    metrics = detector.train(df, validate=True)
    detector.save()
    return metrics