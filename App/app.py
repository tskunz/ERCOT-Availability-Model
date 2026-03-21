import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import calendar
import plotly.express as px
from dataclasses import dataclass
from typing import Dict, List

# --- 1. Inference Model Wrapper ---
# We define a lightweight wrapper class so we don't have to rely on 
# complex local Python paths to import the original class. This makes 
# deployment to Hugging Face or GitHub much more robust!
import xgboost as xgb

class InferenceModel:
    def __init__(self, model_dict):
        self.trend_model = model_dict['trend_model']
        self.xgb_model = model_dict['xgb_model']
        self.n_train = model_dict.get('n_train', 0)
        
    def predict(self, X: pd.DataFrame, start_idx: int = None):
        n_pred = len(X)
        if start_idx is None:
            start_idx = self.n_train
        time_index_test = np.arange(n_pred).reshape(-1, 1)
        time_index_test_shifted = time_index_test + start_idx
        trend_pred = self.trend_model.predict(time_index_test_shifted)
        
        dtest = xgb.DMatrix(X)
        seasonality_pred = self.xgb_model.predict(dtest)
        return trend_pred + seasonality_pred
        
    def get_feature_importance(self, top_n=10):
        # Extract feature importance straight from the XGBoost component
        importance = self.xgb_model.get_score(importance_type='gain')
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        return importance_df.head(top_n)

# --- 2. Configuration Dataclasses ---
@dataclass
class DataCenterResources:
    it_load_mw: float
    flexible_it_pct: float
    bess_power_mw: float
    bess_energy_mwh: float
    genset_capacity_mw: float

# --- 3. Load Assets ---
model_dict = joblib.load("best_hybrid_model.pkl")
model = InferenceModel(model_dict)
residuals_library = np.load("residuals_library.npy")

# --- 4. Logic Functions ---

def predict_availability(temp, humidity, hour, month, day, wind_speed, precip, cloud_cover, wind_gust, year):
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
    return float(prediction[0])

def run_simulation(temp, humidity, hour, month, day, wind_speed, precip, cloud_cover, wind_gust, year, simulated_demand_mw, bess_mw, genset_mw, flex_pct, n_sims):
    predicted_availability_mw = predict_availability(temp, humidity, hour, month, day, wind_speed, precip, cloud_cover, wind_gust, year)
    
    # DC Configuration
    dc_response = bess_mw + genset_mw + (25 * flex_pct) # Assuming 25MW base DC load
    
    eens_results = []
    for _ in range(int(n_sims)):
        noise = np.random.choice(residuals_library)
        simulated_available = predicted_availability_mw + noise
        net_shortfall = max(0, (simulated_demand_mw - dc_response) - simulated_available)
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
    gr.Markdown("Forecast grid availability and simulate the reliability ROI of behind-the-meter investments.")
    
    # Hidden year input to simulate future predictions
    future_year = gr.Number(value=2026, visible=False)

    with gr.Tab("Grid Capacity Predictor"):
        gr.Markdown("### 🌩️ Quick Presets")
        with gr.Row():
            btn_uri = gr.Button("🌨️ Winter Storm Uri", size="sm")
            btn_summer = gr.Button("☀️ Summer Heatwave", size="sm")
            btn_mild = gr.Button("🌸 Mild Spring Day", size="sm")
            btn_custom = gr.Button("⚙️ Reset to Custom", size="sm", variant="secondary")

        gr.Markdown("---")
        gr.Markdown("### 🛠️ Build Your Own Custom Scenario")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📅 Date & Time")
                mo = gr.Slider(1, 12, value=8, step=1, label="Month")
                day = gr.Slider(1, 31, value=15, step=1, label="Day of Month")
                hr = gr.Slider(0, 23, value=16, step=1, label="Hour of Day")
            with gr.Column():
                gr.Markdown("#### 🌡️ Basic Weather")
                t = gr.Slider(0, 110, value=95, label="Temperature (°F)")
                h = gr.Slider(0, 100, value=60, label="Humidity (%)")
        
        with gr.Accordion("Advanced Weather Options (Deep Dive)", open=False):
            with gr.Row():
                wind = gr.Slider(0, 50, value=10, step=1, label="Wind Speed (mph)")
                precip = gr.Slider(0, 100, value=0, label="Precipitation (%)")
                cloud = gr.Slider(0, 100, value=20, label="Cloud Cover (%)")
                gust = gr.Slider(0, 70, value=15, label="Wind Gusts (mph)")

        with gr.Row():
            btn_pred = gr.Button("Calculate Available Capacity", variant="primary")
            output_mw = gr.Number(label="Predicted Available Capacity (MW)")

    with gr.Tab("DC Investment Simulator"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🌩️ Grid Stress Testing")
                simulated_demand_mw = gr.Slider(30000, 100000, value=70000, step=1000, label="Simulated Peak Demand (MW)", info="ERCOT load demand to test against predicted capacity.")

                gr.Markdown("### 🔋 DC Resource Settings")
                bess = gr.Slider(0, 50, value=10, label="BESS Power (MW)")
                gens = gr.Slider(0, 50, value=5, label="Genset Capacity (MW)")
                flex = gr.Slider(0, 1, value=0.1, label="Flexible Load %")
                sims = gr.Radio([100, 1000, 5000], value=1000, label="Monte Carlo Iterations")
                run_btn = gr.Button("Run Reliability Sim", variant="primary")
            
            with gr.Column():
                gr.Markdown("### 📊 Risk Metrics")
                gr.Markdown(
                    "**💡 Metric Guide:**\n"
                    "* **EENS (Expected Energy Not Served):** The average megawatt-hours (MWh) of grid shortfall across all simulations. *Lower is better.*\n"
                    "* **LOLP (Loss of Load Probability):** The percentage chance that a grid outage or shortfall occurs at all. *Lower is better.*"
                )
                res_eens = gr.Textbox(label="Expected Energy Not Served (EENS)")
                res_lolp = gr.Textbox(label="Loss of Load Probability (LOLP)")
                plot = gr.Plot(label="Reliability Distribution")

    # Wire up the logic
    # predict_availability signature: temp, humidity, hour, month, day, wind_speed, precip, cloud_cover, wind_gust, year
    weather_inputs = [t, h, hr, mo, day, wind, precip, cloud, gust, future_year]

    btn_pred.click(
        predict_availability, 
        inputs=weather_inputs, 
        outputs=output_mw
    )

    run_btn.click(
        run_simulation, 
        inputs=weather_inputs + [simulated_demand_mw, bess, gens, flex, sims], 
        outputs=[res_eens, res_lolp, plot]
    )

    # Dynamic Month Demand Adjustment
    def update_season_demand(m):
        m = int(m)
        if m in [6, 7, 8, 9]: return 80000     # Summer Peak Demand
        elif m in [12, 1, 2]: return 60000     # Winter Normal Demand
        else: return 45000                     # Spring/Fall Normal Demand
        
    mo.change(update_season_demand, inputs=mo, outputs=simulated_demand_mw)

    # Preset Logic Function Mappings
    # Format: [temp, humidity, hour, month, day, wind, precip, cloud, gust, simulated_demand_mw]
    def set_uri():
        return [15, 85, 8, 2, 15, 25, 20, 100, 35, 74000]  # Uri Demand
    def set_summer():
        return [105, 40, 17, 8, 15, 5, 0, 10, 10, 85000]   # Summer Peak
    def set_mild():
        return [70, 50, 12, 4, 15, 10, 0, 20, 15, 45000]   # Mild Day
    def set_custom():
        return [95, 60, 16, 8, 15, 10, 0, 20, 15, 70000]   # Default Custom Set

    slider_targets = [t, h, hr, mo, day, wind, precip, cloud, gust, simulated_demand_mw]
    btn_uri.click(set_uri, inputs=None, outputs=slider_targets)
    btn_summer.click(set_summer, inputs=None, outputs=slider_targets)
    btn_mild.click(set_mild, inputs=None, outputs=slider_targets)
    btn_custom.click(set_custom, inputs=None, outputs=slider_targets)

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
