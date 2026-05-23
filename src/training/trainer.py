import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix
)
import joblib


# ── Paths ───────────────────────────────────────────────────────────────────
PROCESSED_PATH = Path("data/processed/ai4i_clean.csv")
MODEL_PATH = Path("models/random_forest.pkl")


# ── Loading ─────────────────────────────────────────────────────────────────
def load_clean_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load cleaned dataset."""
    df = pd.read_csv(path)
    print(f"✓ Data loaded : {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── Splitting ────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    """Split features and target, then train/test split."""
    # Drop failure mode columns — we predict machine_failure only
    cols_to_drop = ["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]
    X = df.drop(columns=cols_to_drop)
    y = df["machine_failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✓ Train : {X_train.shape[0]} rows | Test : {X_test.shape[0]} rows")
    print(f"✓ Failure rate in train : {y_train.mean()*100:.2f}%")
    print(f"✓ Failure rate in test  : {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


# ── Training ─────────────────────────────────────────────────────────────────
def train_model(X_train, y_train) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",  # handles class imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✓ Model trained : RandomForestClassifier")
    return model


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test) -> None:
    """Evaluate model and print metrics."""
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    print(f"\n── Evaluation ──")
    print(f"F1-Score : {f1:.4f}")
    print(f"\nClassification Report :")
    print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))
    print(f"Confusion Matrix :")
    print(confusion_matrix(y_test, y_pred))


# ── Feature Importance ───────────────────────────────────────────────────────
def print_feature_importance(model, X_train) -> None:
    """Print top features by importance."""
    importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print("\n── Feature Importance ──")
    print(importances)


# ── Saving ───────────────────────────────────────────────────────────────────
def save_model(model, path: Path = MODEL_PATH) -> None:
    """Save trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"\n✓ Model saved to {path}")


# ── Pipeline ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    print_feature_importance(model, X_train)
    save_model(model)