"""
Time Series Cross-Validation

Implements proper cross-validation for time series data with expanding window
and no future data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TimeSeriesSplit:
    """
    Time series cross-validator with expanding training window.

    For each fold:
    - Training set expands from start to time t
    - Test set is contiguous period from t to t+size
    - No future data in training set
    """

    def __init__(self, n_splits=5):
        """
        Initialize TimeSeriesSplit.

        Parameters
        ----------
        n_splits : int
            Number of folds
        """
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like
            Data to split
        y : array-like, optional
            Target variable (not used, for sklearn compatibility)
        groups : array-like, optional
            Group labels (not used, for sklearn compatibility)

        Yields
        ------
        train : ndarray
            Training indices for the fold
        test : ndarray
            Test indices for the fold
        """
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)

        for fold in range(1, self.n_splits + 1):
            # Training indices: from start to fold_size * fold
            train_end = fold_size * fold
            train_indices = np.arange(0, train_end)

            # Test indices: contiguous block after training
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)
            test_indices = np.arange(test_start, test_end)

            # Skip fold if test set is empty
            if len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator."""
        return self.n_splits


class TimeSeriesCV:
    """
    Time Series Cross-Validator with metrics calculation.

    Evaluates model across multiple time series folds and computes average metrics.
    """

    def __init__(self, n_splits=5):
        """
        Initialize TimeSeriesCV.

        Parameters
        ----------
        n_splits : int
            Number of folds for cross-validation
        """
        self.n_splits = n_splits
        self.splitter = TimeSeriesSplit(n_splits=n_splits)

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like
            Data to split
        y : array-like, optional
            Target variable (not used, for sklearn compatibility)
        groups : array-like, optional
            Group labels (not used, for sklearn compatibility)

        Yields
        ------
        train : ndarray
            Training indices for the fold
        test : ndarray
            Test indices for the fold
        """
        for train_indices, test_indices in self.splitter.split(X, y, groups):
            yield train_indices, test_indices

    def evaluate(self, model_func, X, y, **fit_params):
        """
        Evaluate model using cross-validation.

        Parameters
        ----------
        model_func : callable
            Function that takes (X_train, y_train) and returns a model with
            .predict() method
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        **fit_params : dict
            Additional parameters for model_func

        Returns
        -------
        results : dict
            Dictionary with average and std metrics across folds
        """
        metrics_list = {
            'rmse': [],
            'mae': [],
            'mape': [],
            'r2': []
        }

        for fold_idx, (train_indices, test_indices) in enumerate(self.split(X, y), 1):
            # Get train/test data for this fold
            X_fold_train = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
            y_fold_train = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
            X_fold_test = X.iloc[test_indices] if hasattr(X, 'iloc') else X[test_indices]
            y_fold_test = y.iloc[test_indices] if hasattr(y, 'iloc') else y[test_indices]

            # Train model
            model = model_func(X_fold_train, y_fold_train, **fit_params)

            # Make predictions
            y_pred = model.predict(X_fold_test)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_fold_test, y_pred))
            mae = mean_absolute_error(y_fold_test, y_pred)
            mape = np.mean(np.abs((y_fold_test - y_pred) / y_fold_test)) * 100
            r2 = r2_score(y_fold_test, y_pred)

            metrics_list['rmse'].append(rmse)
            metrics_list['mae'].append(mae)
            metrics_list['mape'].append(mape)
            metrics_list['r2'].append(r2)

        # Compute statistics
        results = {
            'rmse_mean': np.mean(metrics_list['rmse']),
            'rmse_std': np.std(metrics_list['rmse']),
            'mae_mean': np.mean(metrics_list['mae']),
            'mae_std': np.std(metrics_list['mae']),
            'mape_mean': np.mean(metrics_list['mape']),
            'mape_std': np.std(metrics_list['mape']),
            'r2_mean': np.mean(metrics_list['r2']),
            'r2_std': np.std(metrics_list['r2']),
            'fold_rmse': metrics_list['rmse'],
            'fold_mae': metrics_list['mae'],
            'fold_mape': metrics_list['mape'],
            'fold_r2': metrics_list['r2']
        }

        return results

    def plot_results(self, results, figsize=(14, 8)):
        """
        Plot cross-validation results.

        Parameters
        ----------
        results : dict
            Results from evaluate()
        figsize : tuple
            Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plots")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: RMSE across folds
        ax = axes[0, 0]
        folds = np.arange(1, len(results['fold_rmse']) + 1)
        ax.bar(folds, results['fold_rmse'], color='steelblue', alpha=0.7, edgecolor='black')
        ax.axhline(results['rmse_mean'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {results['rmse_mean']:.2f}")
        ax.set_xlabel('Fold')
        ax.set_ylabel('RMSE (MW)')
        ax.set_title('RMSE Across Folds')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Plot 2: All metrics
        ax = axes[0, 1]
        metrics = ['RMSE', 'MAE', 'MAPE', 'R²']
        means = [results['rmse_mean'], results['mae_mean'], results['mape_mean'],
                 results['r2_mean']]
        stds = [results['rmse_std'], results['mae_std'], results['mape_std'],
                results['r2_std']]
        ax.bar(metrics, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7,
               edgecolor='black')
        ax.set_ylabel('Value')
        ax.set_title('Average Metrics ± Std Dev')
        ax.grid(axis='y', alpha=0.3)

        # Plot 3: R² across folds
        ax = axes[1, 0]
        ax.plot(folds, results['fold_r2'], marker='o', linestyle='-', linewidth=2,
                markersize=8, color='green', alpha=0.7)
        ax.axhline(results['r2_mean'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {results['r2_mean']:.4f}")
        ax.set_xlabel('Fold')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Across Folds')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 4: Metric variation
        ax = axes[1, 1]
        variation = [
            (results['rmse_std'] / results['rmse_mean']) * 100,
            (results['mae_std'] / results['mae_mean']) * 100,
            (results['mape_std'] / results['mape_mean']) * 100,
            (results['r2_std'] / results['r2_mean']) * 100 if results['r2_mean'] > 0 else 0
        ]
        colors = ['green' if v < 15 else 'orange' for v in variation]
        ax.bar(metrics, variation, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(15, color='red', linestyle='--', linewidth=2, label='15% threshold')
        ax.set_ylabel('Variation (%)')
        ax.set_title('Metric Stability Across Folds')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig
