import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from xgboost import XGBClassifier
import joblib
import mlflow
import mlflow.sklearn

# ── Paths ───────────────────────────────────────────────────────────────────
PROCESSED_PATH = Path("data/processed/ai4i_clean.csv")
MODEL_PATH = Path("models/best_model.pkl")

# ── MLflow setup ─────────────────────────────────────────────────────────────
mlflow.set_experiment("ai-industrial-monitoring")

# ── Loading ─────────────────────────────────────────────────────────────────
def load_clean_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✓ Data loaded : {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ── Splitting ────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    cols_to_drop = ["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]
    X = df.drop(columns=cols_to_drop)
    y = df["machine_failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train : {X_train.shape[0]} rows | Test : {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test

# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "accuracy": model.score(X_test, y_test)
    }
    print(f"\n── Evaluation ──")
    for k, v in metrics.items():
        print(f"{k} : {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))
    return metrics

# ── MLflow Run ───────────────────────────────────────────────────────────────
def run_experiment(model_name: str, model, params: dict, X_train, X_test, y_train, y_test):
    """Run a single experiment tracked with MLflow."""
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)

        # Train
        model.fit(X_train, y_train)
        print(f"\n{'='*50}")
        print(f"Model : {model_name}")

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, model_name)

        return metrics["f1_score"], metrics["recall"], model

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load & split data
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # ── Define models ────────────────────────────────────────────────────────
    experiments = [
        {
            "name": "RandomForest",
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "class_weight": "balanced",
                "random_state": 42
            }
        },
        {
            "name": "XGBoost",
            "model": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=29,
                random_state=42,
                eval_metric="logloss",
                verbosity=0
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": 29
            }
        }
    ]

    # ── Run all experiments ──────────────────────────────────────────────────
    results = []
    for exp in experiments:
        f1, recall, model = run_experiment(
            exp["name"], exp["model"], exp["params"],
            X_train, X_test, y_train, y_test
        )
        results.append({
            "name": exp["name"],
            "f1": f1,
            "recall": recall,
            "model": model
        })

    # ── Compare results ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("COMPARISON")
    print(f"{'='*50}")
    for r in sorted(results, key=lambda x: x["recall"], reverse=True):
        print(f"{r['name']:<25} F1 : {r['f1']:.4f}  |  Recall : {r['recall']:.4f}")

    # ── Save best model by recall ─────────────────────────────────────────────
    best = max(results, key=lambda x: x["recall"])
    print(f"\n✓ Best model by recall : {best['name']} (Recall = {best['recall']:.4f})")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)
    print(f"✓ Best model saved to {MODEL_PATH}")