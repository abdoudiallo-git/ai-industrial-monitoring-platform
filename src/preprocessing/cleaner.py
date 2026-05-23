import pandas as pd
import numpy as np
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────
RAW_PATH = Path("data/raw/ai4i2020.csv")
PROCESSED_PATH = Path("data/processed/ai4i_clean.csv")


# ── Loading ─────────────────────────────────────────────────────────────────
def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    df = pd.read_csv(path)
    print(f"✓ Data loaded : {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── Cleaning ────────────────────────────────────────────────────────────────
def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with no ML value."""
    cols_to_drop = ["UDI", "Product ID"]
    df = df.drop(columns=cols_to_drop)
    print(f"✓ Dropped columns : {cols_to_drop}")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    df = df.rename(columns={
        "Type": "type",
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rotational_speed",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "machine_failure",
        "TWF": "twf",
        "HDF": "hdf",
        "PWF": "pwf",
        "OSF": "osf",
        "RNF": "rnf"
    })
    print("✓ Columns renamed")
    return df


def encode_type(df: pd.DataFrame) -> pd.DataFrame:
    """Encode machine type : L→0, M→1, H→2."""
    mapping = {"L": 0, "M": 1, "H": 2}
    df["type"] = df["type"].map(mapping)
    print("✓ Column 'type' encoded : L→0, M→1, H→2")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing ones."""
    # Temperature difference (key for HDF)
    df["delta_temp"] = df["process_temp"] - df["air_temp"]

    # Power in Watts (key for PWF)
    df["power"] = df["torque"] * df["rotational_speed"] * (2 * np.pi / 60)

    # Tool wear × Torque (key for OSF)
    df["tool_wear_torque"] = df["tool_wear"] * df["torque"]

    print("✓ New features added : delta_temp, power, tool_wear_torque")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Check and drop missing values if any."""
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠ {missing} missing values found — dropping rows")
        df = df.dropna()
    else:
        print("✓ No missing values")
    return df


# ── Pipeline ────────────────────────────────────────────────────────────────
def clean_pipeline(path: Path = RAW_PATH) -> pd.DataFrame:
    """Full cleaning pipeline."""
    df = load_raw_data(path)
    df = drop_useless_columns(df)
    df = rename_columns(df)
    df = encode_type(df)
    df = add_features(df)
    df = check_missing_values(df)
    print(f"\n✓ Pipeline complete : {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def save_clean_data(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    """Save cleaned dataset to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✓ Clean data saved to {path}")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_clean = clean_pipeline()
    save_clean_data(df_clean)
    print("\n── Preview ──")
    print(df_clean.head())
    print("\n── Columns ──")
    print(df_clean.columns.tolist())