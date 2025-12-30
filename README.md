# ERCOT-Availability-Model
Availability Model to Evaluate AI Data Centers’ Role in Grid Stability

This repository contains the codebase and methodology for evaluating the impact of large-scale AI data centers on the ERCOT power grid. The research focuses on the "First-Principled Hybrid Model," which outperforms standard machine learning approaches by combining deterministic linear trends with residual-based XGBoost learning.

## Quick Start
To run the model immediately without setting up API credentials or downloading years of raw data, follow these steps:

Clone the Repository:

git clone https://github.com/tskunz/ERCOT-Availability-Model.git
cd ERCOT-Availability-Model

Install Dependencies:
pip install -r requirements.txt

Get the Data:

Download the pre-processed dataset ZIP from the Data Sources Folder. Download "final_master_hourly_data.zip" (the other files are the raw files if you choose to run the formatting as well).

Extract the contents into the /data folder at the root of the project.

Run the Analysis:

Open ERCOT_Availability_Model.ipynb in Jupyter or Google Colab (GPU strongly recommended) to train the models and view the stability results.

## Modeling Methodology
1. The STL-XGBoost Failure Case
The repository includes the implementation of an STL-XGBoost model to demonstrate its limitations in grid forecasting. While effective for seasonal patterns, this model fails to extrapolate the aggressive, non-linear load growth associated with new AI data center clusters.

2. The First-Principled Hybrid Model (Primary)
The core contribution of this work is a hybrid architecture designed to handle structural shifts in the grid:

Linear Regression Base: Captures the deterministic long-term growth trends of the grid.

XGBoost Residual Compensation: Instead of predicting the load directly, the XGBoost model is trained on the residuals of the linear model. It learns to "compensate" for violations of linear assumptions by using weather, time-of-day, and regional weights as features to predict the error term.

Reconstruction: The final forecast is the sum of the linear trend and the learned error compensation.

## Repoository Contents

ERCOT_Availability_Model.ipynb: Full pipeline from training to Monte Carlo reliability analysis.

availability_model/: Implementation of the STLXGBoostForecaster and hybrid logic.

preprocess_weather_weights.ipynb: Calculates population-weighted weather metrics for the ERCOT region.

download_weather_colab.ipynb: Automated weather data retrieval for model features.

## License & Citation
This project is licensed under the MIT License.

Research Paper: Availability Model to Evaluate AI Data Centers’ Role in Grid Stability (SMU, 2026). The paper is currently approved for print and scheduled for official publication next month.

