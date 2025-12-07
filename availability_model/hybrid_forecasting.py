"""
Hybrid forecasting model combining Linear Trend + XGBoost Seasonality.

This model addresses XGBoost's poor extrapolation by decomposing the problem:
1. Linear Regression on time index for trend (extrapolates well)
2. XGBoost on detrended residuals for seasonality (captures non-linear patterns)
3. Combines both for final forecast: Prediction = Trend + Detrended_XGBoost

This approach gives the best of both worlds:
- Linear trend that projects accurately into the future
- XGBoost seasonality that learns complex patterns
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from pathlib import Path
import pickle

# Linear regression for trend
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


class HybridTrendSeasonalityForecaster:
    """
    Hybrid forecasting model combining Linear Trend + XGBoost Seasonality.

    Architecture:
    1. Train a simple Linear Regression on time index [0, 1, 2, ..., n-1]
       to capture long-term trend that extrapolates well
    2. Compute detrended residuals: y_detrended = y_train - trend_pred_train
    3. Train XGBoost on detrended residuals using rich features
    4. Final prediction: y_pred = trend_pred + xgb_pred
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the hybrid forecaster.

        Args:
            use_gpu: Whether to use GPU acceleration for XGBoost
        """
        self.use_gpu = use_gpu and gpu_config.use_gpu

        # Model components
        self.trend_model = None  # Linear regression for trend
        self.xgb_model = None    # XGBoost for detrended residuals

        # Training data info
        self.n_train = None
        self.trend_slope = None
        self.trend_intercept = None

        # Metadata
        self.feature_names = None
        self.train_stats = {}

    def _create_time_index(self, n_samples: int) -> np.ndarray:
        """Create time index [0, 1, 2, ..., n-1] for trend modeling."""
        return np.arange(n_samples).reshape(-1, 1)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        plot_decomposition: bool = True,
        val_start_idx: int = None
    ):
        """
        Fit the hybrid forecasting model.

        Args:
            X_train: Training features
            y_train: Training target (load values)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            plot_decomposition: Whether to plot trend and seasonality components
            val_start_idx: Starting time index for validation data (if None, assumes end of X_train)
        """
        logger.info("Fitting Hybrid Trend+Seasonality forecaster...")

        # Store feature names and training size
        self.feature_names = X_train.columns.tolist()
        self.n_train = len(X_train)

        # ============================================================================
        # STEP 1: Fit Linear Trend Model
        # ============================================================================
        logger.info("Step 1: Fitting linear trend model...")

        time_index = self._create_time_index(len(y_train))
        self.trend_model = LinearRegression()
        self.trend_model.fit(time_index, y_train.values)

        # Get trend predictions for training data
        trend_pred_train = self.trend_model.predict(time_index)

        # Store slope and intercept for reference
        self.trend_slope = float(self.trend_model.coef_[0])
        self.trend_intercept = float(self.trend_model.intercept_)

        logger.info(f"  Trend equation: y = {self.trend_intercept:.2f} + {self.trend_slope:.6f} * t")
        logger.info(f"  Trend range: [{trend_pred_train.min():.2f}, {trend_pred_train.max():.2f}] MW")

        # ============================================================================
        # STEP 2: Compute Detrended Residuals
        # ============================================================================
        logger.info("Step 2: Computing detrended residuals...")

        y_detrended = y_train.values - trend_pred_train

        logger.info(f"  Detrended mean: {y_detrended.mean():.4f} MW (should be ≈0)")
        logger.info(f"  Detrended std: {y_detrended.std():.2f} MW")
        logger.info(f"  Detrended range: [{y_detrended.min():.2f}, {y_detrended.max():.2f}] MW")

        # ============================================================================
        # STEP 3: Train XGBoost on Detrended Residuals
        # ============================================================================
        logger.info("Step 3: Training XGBoost on detrended residuals...")

        # Prepare XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': model_config.xgb_max_depth,
            'learning_rate': model_config.xgb_learning_rate,
            'subsample': model_config.xgb_subsample,
            'colsample_bytree': model_config.xgb_colsample_bytree,
            'reg_alpha': model_config.xgb_reg_alpha,
            'reg_lambda': model_config.xgb_reg_lambda,
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu',
            'random_state': model_config.random_seed,
            'verbosity': 1
        }

        # Create DMatrix for training
        dtrain = xgb.DMatrix(X_train, label=y_detrended)

        # Prepare validation set if provided
        eval_set = []
        if X_val is not None and y_val is not None:
            # Determine validation start index (temporal position in full dataset)
            if val_start_idx is None:
                # If not provided, assume validation comes at END of training
                # This is incorrect! Validation should be part of training data
                val_start_idx = self.n_train - len(y_val)
                logger.warning(f"val_start_idx not provided. Assuming validation at indices [{val_start_idx}:{self.n_train}]")

            # Predict trend for validation set at CORRECT time indices
            time_index_val = self._create_time_index(len(y_val))
            time_index_val_corrected = time_index_val + val_start_idx  # Use CORRECT temporal position
            trend_pred_val = self.trend_model.predict(time_index_val_corrected)

            logger.info(f"Validation time indices: [{val_start_idx}:{val_start_idx + len(y_val)}]")

            # Compute detrended validation target
            y_detrended_val = y_val.values - trend_pred_val

            dval = xgb.DMatrix(X_val, label=y_detrended_val)
            eval_set = [(dtrain, 'train'), (dval, 'val')]

        # Train XGBoost model
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
            'n_features': len(self.feature_names),
            'trend_slope': self.trend_slope,
            'trend_intercept': self.trend_intercept
        }

        logger.info(f"Training complete. Best iteration: {self.xgb_model.best_iteration}")
        if self.train_stats['val_rmse']:
            logger.info(f"Validation RMSE (on detrended): {self.train_stats['val_rmse']:.2f}")

        # Plot decomposition if requested
        if plot_decomposition:
            self._plot_decomposition(y_train.values, trend_pred_train, y_detrended)

    def predict(self, X: pd.DataFrame, start_idx: int = None) -> np.ndarray:
        """
        Make predictions combining trend and seasonality.

        Args:
            X: Feature matrix
            start_idx: Starting time index (if None, assumes continuation from training)

        Returns:
            Predictions array: trend_pred + xgb_pred
        """
        if self.trend_model is None or self.xgb_model is None:
            raise ValueError("Model must be fitted before prediction")

        n_pred = len(X)

        # If start_idx not provided, assume continuation from training data
        if start_idx is None:
            start_idx = self.n_train

        # ============================================================================
        # PREDICT TREND: Simple linear extrapolation
        # ============================================================================
        time_index_test = self._create_time_index(n_pred)
        time_index_test_shifted = time_index_test + start_idx
        trend_pred = self.trend_model.predict(time_index_test_shifted)

        # ============================================================================
        # PREDICT SEASONALITY: XGBoost on features
        # ============================================================================
        dtest = xgb.DMatrix(X)
        seasonality_pred = self.xgb_model.predict(dtest)

        # ============================================================================
        # COMBINE: Final Prediction = Trend + Seasonality
        # ============================================================================
        y_pred = trend_pred + seasonality_pred

        return y_pred

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        plot: bool = True,
        start_idx: int = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test target
            plot: Whether to plot predictions
            start_idx: Starting time index for test set

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set...")

        # Make predictions
        y_pred = self.predict(X_test, start_idx=start_idx)

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

    def _plot_decomposition(
        self,
        y_observed: np.ndarray,
        y_trend: np.ndarray,
        y_detrended: np.ndarray
    ):
        """Plot observed, trend, and detrended components."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        # Plot 1: Observed
        axes[0].plot(y_observed, linewidth=1, color='steelblue', label='Observed')
        axes[0].plot(y_trend, linewidth=2, color='red', alpha=0.7, label='Linear Trend')
        axes[0].set_title('Observed Load vs Linear Trend', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Load (MW)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Trend
        axes[1].plot(y_trend, linewidth=1.5, color='red')
        axes[1].set_title('Linear Trend Component', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Trend (MW)')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Detrended (Seasonality + Noise)
        axes[2].plot(y_detrended, linewidth=1, color='green', alpha=0.7)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_title('Detrended Component (Seasonality + Noise)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Detrended Load (MW)')
        axes[2].set_xlabel('Time Index')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = RESULTS_DIR / 'hybrid_decomposition.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved decomposition plot to {plot_path}")
        plt.close()

    def _plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot actual vs predicted values."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Time series plot (first 7 days)
        n_plot = min(24 * 7, len(y_true))  # 1 week
        axes[0].plot(y_true[:n_plot], label='Actual', linewidth=2, alpha=0.7, color='steelblue')
        axes[0].plot(y_pred[:n_plot], label='Predicted', linewidth=2, alpha=0.7, color='coral')
        axes[0].set_title('Load Forecast vs Actual (First Week)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Hour')
        axes[0].set_ylabel('Load (MW)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
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
        plot_path = RESULTS_DIR / 'hybrid_forecast_predictions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction plot to {plot_path}")
        plt.close()

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from XGBoost model.

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

    def save_model(self, filename: str = "hybrid_forecaster.pkl"):
        """Save model to disk."""
        filepath = MODELS_DIR / filename

        model_data = {
            'trend_model': self.trend_model,
            'xgb_model': self.xgb_model,
            'n_train': self.n_train,
            'trend_slope': self.trend_slope,
            'trend_intercept': self.trend_intercept,
            'feature_names': self.feature_names,
            'train_stats': self.train_stats
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved model to {filepath}")

    def load_model(self, filename: str = "hybrid_forecaster.pkl"):
        """Load model from disk."""
        filepath = MODELS_DIR / filename

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.trend_model = model_data['trend_model']
        self.xgb_model = model_data['xgb_model']
        self.n_train = model_data['n_train']
        self.trend_slope = model_data['trend_slope']
        self.trend_intercept = model_data['trend_intercept']
        self.feature_names = model_data['feature_names']
        self.train_stats = model_data['train_stats']

        logger.info(f"Loaded model from {filepath}")


if __name__ == "__main__":
    print("="*60)
    print("Hybrid Trend+Seasonality Forecasting Demo")
    print("="*60)
    logger.info("This module provides hybrid load forecasting")
    logger.info("Import and use with preprocessed ERCOT data")
