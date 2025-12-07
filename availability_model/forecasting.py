"""
GPU-accelerated load forecasting using STL decomposition + XGBoost.
Implements the forecasting methodology from the research paper.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from pathlib import Path
import pickle

# Time series decomposition
from statsmodels.tsa.seasonal import STL

# Linear regression for trend extrapolation
from sklearn.linear_model import LinearRegression

# XGBoost with GPU support
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    model_config,
    gpu_config,
    MODELS_DIR,
    RESULTS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class STLXGBoostForecaster:
    """
    Two-stage forecasting model:
    1. STL decomposition to separate trend, seasonal, and residual components
    2. XGBoost regression on residuals with GPU acceleration
    """

    def __init__(
        self,
        seasonal_period: int = None,
        trend_window: int = None,
        use_gpu: bool = True
    ):
        """
        Initialize forecaster

        Args:
            seasonal_period: Period for seasonal decomposition (default: 24 hours)
            trend_window: Window for trend extraction (default: 168 hours = 1 week)
            use_gpu: Whether to use GPU acceleration for XGBoost
        """
        self.seasonal_period = seasonal_period or model_config.stl_seasonal_period
        self.trend_window = trend_window or model_config.stl_trend_window
        self.use_gpu = use_gpu and gpu_config.use_gpu

        # Model components
        self.stl_model = None
        self.xgb_model = None
        self.trend_component = None
        self.seasonal_component = None
        self.trend_regression_model = None  # Linear regression for trend extrapolation
        self.n_trend_samples = None  # Number of trend samples for time index

        # Metadata
        self.feature_names = None
        self.train_stats = {}

    def decompose_time_series(
        self,
        y_train: pd.Series,
        timestamps: pd.Series = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform STL decomposition

        Args:
            y_train: Training time series
            timestamps: Optional timestamps for plotting

        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        logger.info("Performing STL decomposition...")

        # Ensure we have enough data
        min_required = 2 * self.seasonal_period
        if len(y_train) < min_required:
            raise ValueError(f"Need at least {min_required} samples for STL decomposition")

        # Validate STL parameters: trend must be odd and > period
        trend_window = self.trend_window
        if trend_window <= self.seasonal_period:
            trend_window = self.seasonal_period + 2
            logger.warning(f"trend_window ({self.trend_window}) must be > period ({self.seasonal_period}). Using {trend_window}")

        if trend_window % 2 == 0:
            trend_window += 1
            logger.warning(f"trend_window must be odd. Using {trend_window}")

        # Reset index to ensure proper time series (STL needs integer index)
        y_train_reset = y_train.reset_index(drop=True)

        # Fit STL - use 'period' parameter explicitly
        self.stl_model = STL(
            y_train_reset,
            period=self.seasonal_period,  # Period for seasonal decomposition
            trend=trend_window,  # Must be odd and > period
            robust=True  # Robust to outliers
        )

        result = self.stl_model.fit()

        # Extract components
        components = {
            'trend': result.trend.values,
            'seasonal': result.seasonal.values,
            'residual': result.resid.values,
            'observed': y_train.values
        }

        # Store for later use
        self.trend_component = components['trend']
        self.seasonal_component = components['seasonal']

        # Log statistics
        logger.info(f"STL Decomposition Statistics:")
        logger.info(f"  Trend range: [{components['trend'].min():.2f}, {components['trend'].max():.2f}]")
        logger.info(f"  Seasonal range: [{components['seasonal'].min():.2f}, {components['seasonal'].max():.2f}]")
        logger.info(f"  Residual std: {components['residual'].std():.2f}")

        return components

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        plot_decomposition: bool = True
    ):
        """
        Fit the complete forecasting model

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            plot_decomposition: Whether to plot STL decomposition
        """
        logger.info("Fitting STL-XGBoost forecaster...")

        # Store feature names
        self.feature_names = X_train.columns.tolist()

        # Step 1: STL Decomposition
        components = self.decompose_time_series(y_train)

        # Step 1.5: Fit trend regression model for proper extrapolation
        logger.info("Fitting linear regression model for trend extrapolation...")
        self.n_trend_samples = len(self.trend_component)
        time_index = np.arange(self.n_trend_samples).reshape(-1, 1)
        self.trend_regression_model = LinearRegression()
        self.trend_regression_model.fit(time_index, self.trend_component)
        trend_slope = float(self.trend_regression_model.coef_[0])
        trend_intercept = float(self.trend_regression_model.intercept_)
        logger.info(f"  Trend equation: y = {trend_intercept:.2f} + {trend_slope:.6f} * t")

        # Plot decomposition if requested
        if plot_decomposition:
            self._plot_decomposition(components)

        # Step 2: Train XGBoost on residuals
        logger.info("Training XGBoost model on residuals...")

        # Prepare XGBoost parameters (XGBoost 3.1+ compatible with regularization)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': model_config.xgb_max_depth,
            'learning_rate': model_config.xgb_learning_rate,
            'subsample': model_config.xgb_subsample,  # Subsample 80% to prevent overfitting
            'colsample_bytree': model_config.xgb_colsample_bytree,  # Use 80% of features
            'reg_alpha': model_config.xgb_reg_alpha,  # L1 regularization
            'reg_lambda': model_config.xgb_reg_lambda,  # L2 regularization
            'tree_method': 'hist',  # Use 'hist' for both CPU and GPU (device parameter controls GPU usage)
            'device': 'cuda' if self.use_gpu else 'cpu',  # XGBoost 3.1+ uses 'device' instead of 'gpu_id'
            'random_state': model_config.random_seed,
            'verbosity': 1
        }

        # Create DMatrix for efficient training
        dtrain = xgb.DMatrix(X_train, label=components['residual'])

        # Prepare validation set if provided
        eval_set = []
        if X_val is not None and y_val is not None:
            # Decompose validation target using trend regression
            # Validation trend continues from where training ended
            val_time_index = np.arange(self.n_trend_samples, self.n_trend_samples + len(y_val)).reshape(-1, 1)
            val_trend = self.trend_regression_model.predict(val_time_index)

            val_seasonal = np.tile(
                self.seasonal_component[:self.seasonal_period],
                len(y_val) // self.seasonal_period + 1
            )[:len(y_val)]
            val_residual = y_val.values - val_trend - val_seasonal

            dval = xgb.DMatrix(X_val, label=val_residual)
            eval_set = [(dtrain, 'train'), (dval, 'val')]

        # Train model
        evals_result = {}
        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=model_config.xgb_n_estimators,
            evals=eval_set if eval_set else [(dtrain, 'train')],
            early_stopping_rounds=model_config.xgb_early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=50
        )

        # Store training statistics
        self.train_stats = {
            'train_rmse': evals_result['train']['rmse'][-1] if 'train' in evals_result else None,
            'val_rmse': evals_result['val']['rmse'][-1] if 'val' in evals_result else None,
            'best_iteration': self.xgb_model.best_iteration,
            'n_features': len(self.feature_names)
        }

        logger.info(f"Training complete. Best iteration: {self.xgb_model.best_iteration}")
        if self.train_stats['val_rmse']:
            logger.info(f"Validation RMSE: {self.train_stats['val_rmse']:.2f}")

    def predict(
        self,
        X: pd.DataFrame,
        forecast_steps: int = None
    ) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Feature matrix
            forecast_steps: Number of steps to forecast (for extrapolating trend/seasonal)

        Returns:
            Predictions array
        """
        if self.xgb_model is None:
            raise ValueError("Model must be fitted before prediction")

        # Predict residuals
        dtest = xgb.DMatrix(X)
        residual_pred = self.xgb_model.predict(dtest)

        # Reconstruct trend and seasonal components
        n_pred = len(X)

        # Extrapolate trend using LINEAR REGRESSION (proper extrapolation)
        # Time indices continue from where training ended
        time_index_pred = np.arange(self.n_trend_samples, self.n_trend_samples + n_pred).reshape(-1, 1)
        trend_pred = self.trend_regression_model.predict(time_index_pred)

        # Repeat seasonal pattern
        seasonal_pred = np.tile(
            self.seasonal_component[:self.seasonal_period],
            n_pred // self.seasonal_period + 1
        )[:n_pred]

        # Combine components
        y_pred = trend_pred + seasonal_pred + residual_pred

        return y_pred

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        plot: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test target
            plot: Whether to plot predictions

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set...")

        # Make predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }

        logger.info("Test Set Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Plot if requested
        if plot:
            self._plot_predictions(y_test.values, y_pred)

        return metrics

    def _plot_decomposition(self, components: Dict[str, np.ndarray]):
        """Plot STL decomposition components"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 10))

        components_to_plot = ['observed', 'trend', 'seasonal', 'residual']
        titles = ['Observed', 'Trend', 'Seasonal', 'Residual']

        for ax, comp_name, title in zip(axes, components_to_plot, titles):
            ax.plot(components[comp_name], linewidth=1)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('MW')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time Index')
        plt.tight_layout()

        # Save plot
        plot_path = RESULTS_DIR / 'stl_decomposition.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved decomposition plot to {plot_path}")
        plt.close()

    def _plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot actual vs predicted values"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Time series plot (first 7 days)
        n_plot = min(24 * 7, len(y_true))  # 1 week
        axes[0].plot(y_true[:n_plot], label='Actual', linewidth=2, alpha=0.7)
        axes[0].plot(y_pred[:n_plot], label='Predicted', linewidth=2, alpha=0.7)
        axes[0].set_title('Load Forecast vs Actual (First Week)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Hour')
        axes[0].set_ylabel('Load (MW)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.3, s=10)
        axes[1].plot([y_true.min(), y_true.max()],
                    [y_true.min(), y_true.max()],
                    'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_title('Predicted vs Actual Load', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Actual Load (MW)')
        axes[1].set_ylabel('Predicted Load (MW)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = RESULTS_DIR / 'forecast_predictions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction plot to {plot_path}")
        plt.close()

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from XGBoost model

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance scores
        """
        if self.xgb_model is None:
            raise ValueError("Model must be fitted first")

        # Get importance scores
        importance = self.xgb_model.get_score(importance_type='gain')

        # Convert to dataframe
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save_model(self, filename: str = "stl_xgboost_model.pkl"):
        """Save model to disk"""
        filepath = MODELS_DIR / filename

        model_data = {
            'xgb_model': self.xgb_model,
            'trend_component': self.trend_component,
            'seasonal_component': self.seasonal_component,
            'seasonal_period': self.seasonal_period,
            'trend_window': self.trend_window,
            'feature_names': self.feature_names,
            'train_stats': self.train_stats
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved model to {filepath}")

    def load_model(self, filename: str = "stl_xgboost_model.pkl"):
        """Load model from disk"""
        filepath = MODELS_DIR / filename

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.xgb_model = model_data['xgb_model']
        self.trend_component = model_data['trend_component']
        self.seasonal_component = model_data['seasonal_component']
        self.seasonal_period = model_data['seasonal_period']
        self.trend_window = model_data['trend_window']
        self.feature_names = model_data['feature_names']
        self.train_stats = model_data['train_stats']

        logger.info(f"Loaded model from {filepath}")


if __name__ == "__main__":
    print("="*60)
    print("STL-XGBoost Forecasting Demo")
    print("="*60)

    # This will be run with actual data
    logger.info("This module provides GPU-accelerated load forecasting")
    logger.info("Import and use with preprocessed ERCOT data")
