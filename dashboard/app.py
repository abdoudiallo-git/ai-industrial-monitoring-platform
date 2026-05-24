import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))
from src.preprocessing.cleaner import (
    drop_useless_columns,
    rename_columns,
    encode_type,
    add_features,
    check_missing_values
)
from src.inference.predictor import load_model, predict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Industrial Monitoring",
    page_icon="🏭",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    return load_model()

model = load_ml_model()

# ── Pipeline on uploaded file ─────────────────────────────────────────────────
def run_pipeline(uploaded_file) -> pd.DataFrame:
    """Apply full cleaning pipeline on uploaded CSV."""
    df = pd.read_csv(uploaded_file)
    df = drop_useless_columns(df)
    df = rename_columns(df)
    df = encode_type(df)
    df = add_features(df)
    df = check_missing_values(df)
    return df

# ── Validate AI4I format ──────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "UDI", "Product ID", "Type",
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]",
    "Tool wear [min]", "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF"
]

def is_valid_ai4i(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in REQUIRED_COLUMNS)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏭 AI Monitoring")
st.sidebar.markdown("---")

# ── Upload section in sidebar ─────────────────────────────────────────────────
st.sidebar.markdown("### 📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload AI4I CSV file", type=["csv"])

# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    if not is_valid_ai4i(df_raw):
        st.sidebar.error("⚠ Invalid format. Please upload an AI4I 2020 compatible CSV.")
        st.session_state.df = None
    else:
        uploaded_file.seek(0)
        df_clean = run_pipeline(uploaded_file)
        st.session_state.df = df_clean
        st.session_state.df_raw = df_raw
        st.sidebar.success(f"✓ {len(df_clean):,} rows loaded")

# ── Navigation ────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")

if st.session_state.df is None:
    st.sidebar.info("Upload a dataset to unlock all pages")
    page = "🏠 Home"
else:
    page = st.sidebar.radio(
        "Go to",
        ["📊 Dashboard", "🔮 Prediction"]
    )

    # Type filter
    st.sidebar.markdown("### Filters")
    type_map = {0: "L — Low", 1: "M — Medium", 2: "H — High"}
    type_filter = st.sidebar.multiselect(
        "Machine Type",
        options=[0, 1, 2],
        default=[0, 1, 2],
        format_func=lambda x: type_map[x]
    )

# ════════════════════════════════════════════════════════════════════════════
# PAGE 0 — HOME (no dataset uploaded yet)
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.df is None:
    st.title("🏭 AI Industrial Monitoring Platform")
    st.markdown("---")
    st.info("👈 Upload an AI4I 2020 compatible CSV file in the sidebar to get started.")

    st.markdown("### What this platform does")
    col1, col2, col3 = st.columns(3)
    col1.success("📊 **Dashboard**\n\nKPIs, charts, failure distribution and correlation analysis.")
    col2.success("🔮 **Prediction**\n\nPredict machine failure probability from sensor values.")
    col3.success("⚙️ **Auto Pipeline**\n\nAutomatic data cleaning and feature engineering.")

    st.markdown("### Expected CSV format")
    st.markdown("The uploaded file must contain the following columns :")
    st.code(", ".join(REQUIRED_COLUMNS))

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    df = st.session_state.df
    df_filtered = df[df["type"].isin(type_filter)] if type_filter else df

    st.title("📊 Dashboard")
    st.markdown("Real-time monitoring of industrial machine health.")
    st.markdown("---")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total = len(df_filtered)
    failures = int(df_filtered["machine_failure"].sum())
    failure_rate = df_filtered["machine_failure"].mean() * 100
    healthy = total - failures

    col1.metric("Total Machines", f"{total:,}")
    col2.metric("Failures Detected", f"{failures:,}")
    col3.metric("Failure Rate", f"{failure_rate:.2f}%")
    col4.metric("Healthy Machines", f"{healthy:,}")

    st.markdown("---")

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    type_labels = {0: "L", 1: "M", 2: "H"}

    with col1:
        st.subheader("Machine Type Distribution")
        fig = px.pie(
            df_filtered,
            names=df_filtered["type"].map(type_labels),
            title="Machine Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Failure Rate by Machine Type")
        failure_by_type = (
            df_filtered.groupby("type")["machine_failure"]
            .mean()
            .reset_index()
        )
        failure_by_type["type"] = failure_by_type["type"].map(type_labels)
        failure_by_type["machine_failure"] *= 100
        fig = px.bar(
            failure_by_type,
            x="type", y="machine_failure",
            title="Failure Rate by Machine Type (%)",
            color="type",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"machine_failure": "Failure Rate (%)", "type": "Machine Type"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Failure Modes Distribution")
        failure_modes = ["twf", "hdf", "pwf", "osf", "rnf"]
        counts = df_filtered[failure_modes].sum().reset_index()
        counts.columns = ["Mode", "Count"]
        counts["Mode"] = counts["Mode"].str.upper()
        fig = px.bar(
            counts, x="Mode", y="Count",
            title="Failure Modes Distribution",
            color="Mode",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Correlation Matrix")
        numeric_cols = [
            "air_temp", "process_temp", "rotational_speed",
            "torque", "tool_wear", "delta_temp", "power", "tool_wear_torque"
        ]
        corr = df_filtered[numeric_cols].corr().round(2)
        fig = px.imshow(
            corr,
            title="Correlation Matrix",
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Distributions ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Sensor Distributions")
    selected_col = st.selectbox(
        "Select a variable",
        ["air_temp", "process_temp", "rotational_speed",
         "torque", "tool_wear", "delta_temp", "power", "tool_wear_torque"]
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df_filtered, x=selected_col,
            color=df_filtered["machine_failure"].map({0: "Normal", 1: "Failure"}),
            title=f"Distribution of {selected_col}",
            barmode="overlay",
            color_discrete_map={"Normal": "#00cc96", "Failure": "#EF553B"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df_filtered, x=df_filtered["machine_failure"].map({0: "Normal", 1: "Failure"}),
            y=selected_col,
            title=f"Boxplot of {selected_col} by status",
            color=df_filtered["machine_failure"].map({0: "Normal", 1: "Failure"}),
            color_discrete_map={"Normal": "#00cc96", "Failure": "#EF553B"}
        )
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    df = st.session_state.df

    st.title("🔮 Machine Failure Prediction")
    st.markdown("---")

    tab1, tab2 = st.tabs(["🎛️ Manual Input", "📋 Predict on Dataset"])

    # ── Tab 1 : Manual prediction ─────────────────────────────────────────────
    with tab1:
        st.markdown("Enter sensor values manually to get a prediction.")
        col1, col2 = st.columns(2)

        with col1:
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
            air_temp = st.slider("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1)
            process_temp = st.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1)

        with col2:
            rotational_speed = st.slider("Rotational Speed (rpm)", 1000, 2000, 1500)
            torque = st.slider("Torque (Nm)", 0.0, 80.0, 40.0, 0.1)
            tool_wear = st.slider("Tool Wear (min)", 0, 250, 100)

        st.markdown("---")
        if st.button("🔍 Predict", use_container_width=True):
            input_data = {
                "type": machine_type,
                "air_temp": air_temp,
                "process_temp": process_temp,
                "rotational_speed": rotational_speed,
                "torque": torque,
                "tool_wear": tool_wear
            }
            result = predict(model, input_data)
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", "⚠️ Failure" if result["prediction"] == 1 else "✅ Normal")
            col2.metric("Failure Probability", f"{result['failure_probability']*100:.1f}%")
            status_color = {"normal": "🟢", "warning": "🟡", "danger": "🔴"}
            col3.metric("Status", f"{status_color[result['status']]} {result['status'].upper()}")

    # ── Tab 2 : Predict on full dataset ──────────────────────────────────────
    with tab2:
        st.markdown("Run predictions on the entire uploaded dataset.")

        if st.button("🚀 Run Predictions on Dataset", use_container_width=True):
            features = [
                "type", "air_temp", "process_temp", "rotational_speed",
                "torque", "tool_wear", "delta_temp", "power", "tool_wear_torque"
            ]
            X = df[features]
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            df_results = df.copy()
            df_results["predicted_failure"] = predictions
            df_results["failure_probability"] = probabilities.round(4)
            df_results["status"] = df_results["failure_probability"].apply(
                lambda p: "danger" if p >= 0.6 else ("warning" if p >= 0.3 else "normal")
            )

            # KPIs
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Failures", f"{int(predictions.sum()):,}")
            col2.metric("Prediction Rate", f"{predictions.mean()*100:.2f}%")
            col3.metric("Avg Failure Probability", f"{probabilities.mean()*100:.2f}%")

            # Status distribution
            st.markdown("---")
            status_counts = df_results["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig = px.pie(
                status_counts, names="Status", values="Count",
                title="Machine Status Distribution",
                color="Status",
                color_discrete_map={
                    "normal": "#00cc96",
                    "warning": "#FFA15A",
                    "danger": "#EF553B"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

            # Results table
            st.markdown("### Results Table")
            st.dataframe(
                df_results[[
                    "type", "air_temp", "rotational_speed", "torque",
                    "tool_wear", "machine_failure", "predicted_failure",
                    "failure_probability", "status"
                ]].head(100),
                use_container_width=True
            )