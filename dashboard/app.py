import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))
from src.inference.predictor import load_model, predict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Industrial Monitoring",
    page_icon="🏭",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/ai4i_clean.csv")

@st.cache_resource
def load_ml_model():
    return load_model()

df = load_data()
model = load_ml_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏭 AI Monitoring")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🔮 Prediction"]
)

# ── Type filter ───────────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")
type_filter = st.sidebar.multiselect(
    "Machine Type",
    options=[0, 1, 2],
    default=[0, 1, 2],
    format_func=lambda x: {0: "L — Low", 1: "M — Medium", 2: "H — High"}[x]
)

df_filtered = df[df["type"].isin(type_filter)]

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.title("🏭 AI Industrial Monitoring Platform")
    st.markdown("Real-time monitoring of industrial machine health.")
    st.markdown("---")

    # ── KPIs ─────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    total = len(df_filtered)
    failures = df_filtered["machine_failure"].sum()
    failure_rate = df_filtered["machine_failure"].mean() * 100
    healthy = total - failures

    col1.metric("Total Machines", f"{total:,}")
    col2.metric("Failures Detected", f"{int(failures):,}")
    col3.metric("Failure Rate", f"{failure_rate:.2f}%")
    col4.metric("Healthy Machines", f"{int(healthy):,}")

    st.markdown("---")

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Machine Type Distribution")
        type_map = {0: "L", 1: "M", 2: "H"}
        fig = px.pie(
            df_filtered,
            names=df_filtered["type"].map(type_map),
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
        failure_by_type["type"] = failure_by_type["type"].map(type_map)
        failure_by_type["machine_failure"] = failure_by_type["machine_failure"] * 100
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
        st.subheader("Failure Modes")
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

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":

    st.title("🔮 Machine Failure Prediction")
    st.markdown("Enter sensor values to predict machine health status.")
    st.markdown("---")

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

        status_color = {
            "normal": "🟢",
            "warning": "🟡",
            "danger": "🔴"
        }
        col3.metric("Status", f"{status_color[result['status']]} {result['status'].upper()}")