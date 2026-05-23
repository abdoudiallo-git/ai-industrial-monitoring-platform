import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ── Paths ───────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/random_forest.pkl")


# ── Loading ─────────────────────────────────────────────────────────────────
def load_model(path: Path = MODEL_PATH):
    """Load trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run trainer.py first.")
    model = joblib.load(path)
    print(f"✓ Model loaded from {path}")
    return model


# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Preprocess raw input data before prediction.
    Expects keys: type, air_temp, process_temp,
                  rotational_speed, torque, tool_wear
    """
    # Encode type
    type_mapping = {"L": 0, "M": 1, "H": 2}
    machine_type = data.get("type")
    if isinstance(machine_type, str):
        if machine_type not in type_mapping:
            raise ValueError(f"Invalid type '{machine_type}'. Must be L, M or H.")
        data["type"] = type_mapping[machine_type]

    # Compute engineered features
    air_temp = data["air_temp"]
    process_temp = data["process_temp"]
    rotational_speed = data["rotational_speed"]
    torque = data["torque"]
    tool_wear = data["tool_wear"]

    data["delta_temp"] = process_temp - air_temp
    data["power"] = torque * rotational_speed * (2 * np.pi / 60)
    data["tool_wear_torque"] = tool_wear * torque

    # Build DataFrame with correct column order
    features = [
        "type", "air_temp", "process_temp", "rotational_speed",
        "torque", "tool_wear", "delta_temp", "power", "tool_wear_torque"
    ]

    df = pd.DataFrame([data])[features]
    return df


# ── Prediction ───────────────────────────────────────────────────────────────
def predict(model, data: dict) -> dict:
    """
    Make a prediction from raw input data.
    Returns prediction, probability and status.
    """
    df = preprocess_input(data)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if probability < 0.3:
        status = "normal"
    elif probability < 0.6:
        status = "warning"
    else:
        status = "danger"

    return {
        "prediction": int(prediction),
        "failure_probability": round(float(probability), 4),
        "status": status
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = load_model()

    # Test 1 — Normal machine (values close to the averages "no failure")
    test_normal = {
        "type": "L",
        "air_temp": 300.0,
        "process_temp": 310.0,
        "rotational_speed": 1540,
        "torque": 40.0,
        "tool_wear": 107
    }

    # Test 2 — Machine at risk (values above average "failure")
    test_danger = {
        "type": "L",
        "air_temp": 300.9,
        "process_temp": 310.3,
        "rotational_speed": 1380,  
        "torque": 65.0,            
        "tool_wear": 210           
    }

    print("\n── Test 1 : Normal machine ──")
    result1 = predict(model, test_normal)
    print(result1)

    print("\n── Test 2 : Machine at risk ──")
    result2 = predict(model, test_danger)
    print(result2)