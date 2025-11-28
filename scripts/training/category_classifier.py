import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


class CategoryClassifier:

    def __init__(self, model_path="models/category_model.pkl"):
        self.model_path = model_path
        self.model = None

    # -----------------------------------------------------
    # TRAIN MODEL
    # -----------------------------------------------------
    def train(self, df: pd.DataFrame, text_col="Description", label_col="Category"):
        """
        Train a TF-IDF + Logistic Regression classifier on labeled transactions.
        """
        df = df.dropna(subset=[text_col, label_col])

        X = df[text_col].astype(str)
        y = df[label_col].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Pipeline = TF-IDF + LogisticRegression
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=200))
        ])

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)

        print("\n===== TRAINING REPORT =====\n")
        print(report)

        return report

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------
    def save(self):
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------
    def load(self):
        self.model = joblib.load(self.model_path)
        return self.model

    # -----------------------------------------------------
    # PREDICT CATEGORY FOR NEW DATA
    # -----------------------------------------------------
    def predict(self, df: pd.DataFrame, text_col="Description"):
        if self.model is None:
            self.load()

        df = df.copy()
        df["Predicted_Category"] = self.model.predict(df[text_col].astype(str))
        return df
