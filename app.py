import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from prophet.serialize import model_from_json

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy Forecast Scenario Planner",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Consumption Forecast & Scenario Planner")
st.markdown(
    "Select a city to explore its baseline Prophet forecast or test 'What-If' scenarios "
    "by altering future predictors (e.g., temperature changes or new data centers)."
)

# --- Helper Functions for Data Loading ---
@st.cache_resource
def load_prophet_model(filepath):
    """Loads and deserializes a saved Prophet model from JSON."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return model_from_json(json.load(f))

@st.cache_data
def load_json_file(filepath):
    """Utility to load JSON metadata files."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

# --- Sidebar: City Selection ---
CITIES = ["Chicago", "New York", "Seattle", "Miami", "Dallas"]
selected_city = st.sidebar.selectbox("🏙️ Select City", CITIES)

city_safe = selected_city.replace(" ", "_").lower()
model_path = os.path.join("models", f"{city_safe}_prophet_model.json")
forecast_path = os.path.join("models", f"{city_safe}_forecast.json")
predictors_path = os.path.join("models", f"{city_safe}_predictors.json")

# --- Verify File Existence ---
if not (os.path.exists(model_path) and os.path.exists(forecast_path) and os.path.exists(predictors_path)):
    st.error(
        f"Missing model outputs for **{selected_city}** in the `models/` folder. "
        f"Please run `forecast_prophet()` first to generate the required JSON files."
    )
    st.stop()

# Load Model and Data
model = load_prophet_model(model_path)
baseline_forecast_df = pd.read_json(forecast_path, orient='records')
baseline_forecast_df['ds'] = pd.to_datetime(baseline_forecast_df['ds'])
predictor_meta = load_json_file(predictors_path)
selected_predictors = predictor_meta.get("best_predictors", [])

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Selected Predictors")
if selected_predictors:
    for pred in selected_predictors:
        st.sidebar.markdown(f"- `{pred}`")
else:
    st.sidebar.info("No external predictors were selected during model training for this city.")

# --- Sidebar: What-If Controls ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 'What-If' Scenario Controls")
st.sidebar.caption("Adjust values for the future forecast horizon.")

temp_delta = 0.0
dc_delta = 0
humidity_delta = 0.0

# 1. Temperature Control
if any(col in selected_predictors for col in ['Temp_Mean_C', 'Temp_Max_C', 'Temp_Min_C', 'HDD', 'CDD']):
    temp_delta = st.sidebar.slider(
        "Temperature Shift (°C)",
        min_value=-15.0,
        max_value=15.0,
        value=0.0,
        step=0.5,
        help="Applies a temperature change to the forecast period."
    )

# 2. Data Center Control
if 'total_data_center_count' in selected_predictors:
    dc_delta = st.sidebar.number_input(
        "Data Center Delta",
        min_value=-10,
        max_value=50,
        value=0,
        step=1,
        help="Add or remove data centers."
    )

# 3. Humidity Control
if 'Humidity_Pct' in selected_predictors:
    humidity_delta = st.sidebar.slider(
        "Humidity Shift (%)",
        min_value=-30.0,
        max_value=30.0,
        value=0.0,
        step=1.0
    )

# --- Scenario Simulation Engine ---
scenario_df = baseline_forecast_df.copy()

# Identify future forecast rows (where actual historical data is missing)
future_mask = scenario_df['actual_y'].isna()

# Apply Temperature shifts & recalculate derived Degree Days (HDD/CDD)
if temp_delta != 0.0:
    for col in ['Temp_Mean_C', 'Temp_Max_C', 'Temp_Min_C']:
        if col in scenario_df.columns:
            scenario_df.loc[future_mask, col] += temp_delta
    
    # Recalculate HDD & CDD dynamically if they are used by the model
    baseline_c = 18.3
    if 'Temp_Mean_C' in scenario_df.columns:
        t_mean = scenario_df.loc[future_mask, 'Temp_Mean_C']
        if 'HDD' in scenario_df.columns:
            scenario_df.loc[future_mask, 'HDD'] = (baseline_c - t_mean).clip(lower=0)
        if 'CDD' in scenario_df.columns:
            scenario_df.loc[future_mask, 'CDD'] = (t_mean - baseline_c).clip(lower=0)

# Apply Data Center modifications
if dc_delta != 0 and 'total_data_center_count' in scenario_df.columns:
    scenario_df.loc[future_mask, 'total_data_center_count'] = (
        scenario_df.loc[future_mask, 'total_data_center_count'] + dc_delta
    ).clip(lower=0)

# Apply Humidity modifications
if humidity_delta != 0.0 and 'Humidity_Pct' in scenario_df.columns:
    scenario_df.loc[future_mask, 'Humidity_Pct'] = (
        scenario_df.loc[future_mask, 'Humidity_Pct'] + humidity_delta
    ).clip(lower=0, upper=100)

# Re-run Prophet prediction for the scenario
predict_cols = ['ds'] + selected_predictors
scenario_predictions = model.predict(scenario_df[predict_cols])
scenario_df['yhat_scenario'] = scenario_predictions['yhat']

# --- KPI Metric Summary ---
st.subheader(f"📈 Scenario Impact Summary: {selected_city}")

base_future_avg = scenario_df.loc[future_mask, 'yhat'].mean()
scen_future_avg = scenario_df.loc[future_mask, 'yhat_scenario'].mean()
diff_mwh = scen_future_avg - base_future_avg
pct_diff = (diff_mwh / base_future_avg * 100) if base_future_avg != 0 else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Baseline Avg Demand", f"{base_future_avg:,.1f} MWh")
col2.metric("Scenario Avg Demand", f"{scen_future_avg:,.1f} MWh", delta=f"{diff_mwh:+,.1f} MWh")
col3.metric("Projected Impact", f"{pct_diff:+.2f}%")

# --- Interactive Plotly Chart ---
st.subheader("📊 Model Comparison Chart")

timeframe = st.radio(
    "Display Range:",
    ["Last 60 Days + Future Forecast", "Full Dataset"],
    horizontal=True
)

plot_df = scenario_df.copy()
if timeframe == "Last 60 Days + Future Forecast":
    cutoff_date = plot_df['ds'].max() - pd.Timedelta(days=60)
    plot_df = plot_df[plot_df['ds'] >= cutoff_date]

fig = go.Figure()

# Historical Actual Demand
fig.add_trace(go.Scatter(
    x=plot_df['ds'],
    y=plot_df['actual_y'],
    mode='lines',
    name='Actual Consumption',
    line=dict(color='#888888', width=1.5)
))

# Baseline Model Prediction
fig.add_trace(go.Scatter(
    x=plot_df['ds'],
    y=plot_df['yhat'],
    mode='lines',
    name='Baseline Forecast',
    line=dict(color='#1f77b4', width=2.5)
))

# What-If Scenario Prediction
fig.add_trace(go.Scatter(
    x=plot_df['ds'],
    y=plot_df['yhat_scenario'],
    mode='lines',
    name='Scenario Forecast',
    line=dict(color='#ff7f0e', width=2.5, dash='dash')
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Energy Demand (MWh)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)