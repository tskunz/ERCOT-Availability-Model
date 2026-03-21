import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import calendar
import plotly.express as px
from dataclasses import dataclass
from typing import Dict, List

# --- 1. Custom Class Definition ---
# This is necessary to load your best_hybrid_model.pkl
# If your folder 'availability_model' is present, we import from there.
try:
    from availability_model.hybrid_forecasting import HybridTrendSeasonalityForecaster
except ImportError:
    # Fallback if you prefer to define the structure here, 
    # but having the folder is better.
    pass

# --- 2. Configuration Dataclasses ---
@dataclass
class DataCenterResources:
    it_load_mw: float
    flexible_it_pct: float
    bess_power_mw: float
    bess_energy_mwh: float
    genset_capacity_mw: float

# --- 3. Load Assets ---
# Ensure these files are in the same folder as app.py
model = joblib.load("best_hybrid_model.pkl")
residuals_library = np.load("residuals_library.npy")

# --- 4. Logic Functions ---

def predict_load(temp, humidity, hour, month, day, wind_speed, precip, cloud_cover, wind_gust, year):
    # Handle tricky dates like Feb 30th
    try:
        ts = pd.Timestamp(year, int(month), int(day))
    except ValueError:
        # If invalid date (e.g., Feb 31), fallback to the last valid day of that month
        last_day = calendar.monthrange(year, int(month))[1]
        ts = pd.Timestamp(year, int(month), last_day)

    data = {
        'hour': [hour],
        'day_of_week': [ts.dayofweek], 
        'month': [month],
        'day_of_year': [ts.dayofyear],
        'year': [year],
        'weighted_temp': [temp],
        'weighted_humidity': [humidity],
        'weighted_precipitation': [precip],
        'weighted_wind_speed': [wind_speed],
        'weighted_apparent_temp': [temp],
        'weighted_dew_point': [temp - ((100 - humidity)/5)],
        'weighted_cloud_cover': [cloud_cover],
        'weighted_wind_gusts': [wind_gust]
    }
    X_input = pd.DataFrame(data)
    
    # Use a high start_idx to project the trend forward
    prediction = model.predict(X_input, start_idx=85000)
    return prediction[0]

def run_simulation(temp, humidity, hour, month, day, wind_speed, precip, cloud_cover, wind_gust, year, bess_mw, genset_mw, flex_pct, n_sims):
    base_signal = predict_load(temp, humidity, hour, month, day, wind_speed, precip, cloud_cover, wind_gust, year)
    supply_cap = 85000 # ERCOT target capacity
    
    # DC Configuration
    dc_response = bess_mw + genset_mw + (25 * flex_pct) # Assuming 25MW base DC load
    
    eens_results = []
    for _ in range(int(n_sims)):
        noise = np.random.choice(residuals_library)
        simulated_load = base_signal + noise
        net_shortfall = max(0, (simulated_load - dc_response) - supply_cap)
        eens_results.append(net_shortfall)
    
    # Generate Interactive Plotly Histogram
    df_results = pd.DataFrame({'Shortfall (MWh)': eens_results})
    fig = px.histogram(df_results, x='Shortfall (MWh)', nbins=30,
                       title="Probability Distribution of Grid Shortfall",
                       color_discrete_sequence=['coral'],
                       opacity=0.7)
    fig.update_layout(xaxis_title="MWh Shortfall", yaxis_title="Frequency", template="plotly_white")
    
    mean_eens = np.mean(eens_results)
    lolp = np.mean([1 if x > 0 else 0 for x in eens_results])
    
    return f"{mean_eens:.2f} MWh", f"{lolp:.2%}", fig

# --- 5. Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ ERCOT Reliability & Data Center Strategy Dashboard")
    gr.Markdown("Forecast grid load and simulate the reliability ROI of behind-the-meter investments.")
    
    # Hidden year input to simulate future predictions
    future_year = gr.Number(value=2026, visible=False)

    with gr.Tab("Grid Load Predictor"):
        gr.Markdown("### 🌩️ Quick Presets")
        with gr.Row():
            btn_uri = gr.Button("🌨️ Simulate Winter Storm Uri", size="sm")
            btn_summer = gr.Button("☀️ Simulate Summer Heatwave", size="sm")
            btn_mild = gr.Button("🌸 Simulate Mild Spring Day", size="sm")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📅 Date & Time")
                mo = gr.Slider(1, 12, value=8, step=1, label="Month")
                day = gr.Slider(1, 31, value=15, step=1, label="Day of Month")
                hr = gr.Slider(0, 23, value=16, step=1, label="Hour of Day")
            with gr.Column():
                gr.Markdown("### 🌡️ Basic Weather")
                t = gr.Slider(0, 110, value=95, label="Temperature (°F)")
                h = gr.Slider(0, 100, value=60, label="Humidity (%)")
        
        with gr.Accordion("Advanced Weather Options (Deep Dive)", open=False):
            with gr.Row():
                wind = gr.Slider(0, 50, value=10, step=1, label="Wind Speed (mph)")
                precip = gr.Slider(0, 100, value=0, label="Precipitation (%)")
                cloud = gr.Slider(0, 100, value=20, label="Cloud Cover (%)")
                gust = gr.Slider(0, 70, value=15, label="Wind Gusts (mph)")

        with gr.Row():
            btn_pred = gr.Button("Calculate Load", variant="primary")
            output_mw = gr.Number(label="Predicted Load (MW)")

    with gr.Tab("DC Investment Simulator"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔋 DC Resource Settings")
                bess = gr.Slider(0, 50, value=10, label="BESS Power (MW)")
                gens = gr.Slider(0, 50, value=5, label="Genset Capacity (MW)")
                flex = gr.Slider(0, 1, value=0.1, label="Flexible Load %")
                sims = gr.Radio([100, 1000, 5000], value=1000, label="Monte Carlo Iterations")
                run_btn = gr.Button("Run Reliability Sim", variant="primary")
            
            with gr.Column():
                gr.Markdown("### 📊 Risk Metrics")
                res_eens = gr.Textbox(label="Expected Energy Not Served (EENS)")
                res_lolp = gr.Textbox(label="Loss of Load Probability (LOLP)")
                plot = gr.Plot(label="Reliability Distribution")

    # Wire up the logic
    # predict_load signature: temp, humidity, hour, month, day, wind_speed, precip, cloud_cover, wind_gust, year
    weather_inputs = [t, h, hr, mo, day, wind, precip, cloud, gust, future_year]

    btn_pred.click(
        predict_load, 
        inputs=weather_inputs, 
        outputs=output_mw
    )

    run_btn.click(
        run_simulation, 
        inputs=weather_inputs + [bess, gens, flex, sims], 
        outputs=[res_eens, res_lolp, plot]
    )

    # Preset Logic Function Mappings
    def set_uri():
        return [15, 85, 8, 2, 15, 25, 20, 100, 35]
    def set_summer():
        return [105, 40, 17, 8, 15, 5, 0, 10, 10]
    def set_mild():
        return [70, 50, 12, 4, 15, 10, 0, 20, 15]

    slider_targets = [t, h, hr, mo, day, wind, precip, cloud, gust]
    btn_uri.click(set_uri, inputs=None, outputs=slider_targets)
    btn_summer.click(set_summer, inputs=None, outputs=slider_targets)
    btn_mild.click(set_mild, inputs=None, outputs=slider_targets)

    with gr.Tab("Methodology & Architecture"):
        gr.Markdown("### 📚 Research Paper")
        gr.Markdown("[Read the full research publication: Availability Model to Evaluate AI Data Centers' Role in Grid Stability](https://scholar.smu.edu/datasciencereview/vol9/iss3/3/)")
        
        gr.Markdown("### 🧠 Model Feature Importance")
        # Try to plot feature importances if the model has the method
        if hasattr(model, 'get_feature_importance'):
            fi_df = model.get_feature_importance(top_n=10)
            fi_fig = px.bar(fi_df, x='importance', y='feature', orientation='h', title="Top 10 Features Driving Predictions", color_discrete_sequence=['coral'])
            fi_fig.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
            gr.Plot(value=fi_fig, label="Top Features")
        else:
            gr.Markdown("*Feature importance not available for this model object.*")

if __name__ == "__main__":
    demo.launch(share=True, debug=True)