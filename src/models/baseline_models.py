# src/models/baseline_models.py
"""
Baseline statistical models for COVID-19 forecasting comparison.

This module provides simple baseline models that can be used to benchmark
the deep learning models. These baselines help establish whether the
complexity of neural networks is justified.

Reference:
    Ajao-olarinoye et al., "Deep Learning Based Forecasting of COVID-19 
    Hospitalisation in England: A Comparative Analysis", ICMLA 2023
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class BaselineModel(ABC):
    """Abstract base class for baseline forecasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, y: np.ndarray) -> 'BaselineModel':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, n_periods: int) -> np.ndarray:
        """Generate forecasts for n_periods ahead."""
        pass
    
    def fit_predict(self, y: np.ndarray, n_periods: int) -> np.ndarray:
        """Convenience method to fit and predict in one step."""
        self.fit(y)
        return self.predict(n_periods)


class NaiveModel(BaselineModel):
    """
    Naive baseline: predicts the last observed value for all future periods.
    
    This is the simplest possible baseline and often works surprisingly well
    for short-term forecasting of slowly-changing time series.
    """
    
    def __init__(self):
        super().__init__("Naive")
        self.last_value = None
    
    def fit(self, y: np.ndarray) -> 'NaiveModel':
        """Store the last observed value."""
        self.last_value = y[-1]
        self._is_fitted = True
        return self
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Predict the last value for all future periods."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(n_periods, self.last_value)


class SeasonalNaiveModel(BaselineModel):
    """
    Seasonal Naive baseline: predicts values from the same period in the previous season.
    
    For COVID-19 data, this captures weekly patterns where weekends might
    have different reporting patterns than weekdays.
    """
    
    def __init__(self, season_length: int = 7):
        super().__init__(f"SeasonalNaive_{season_length}")
        self.season_length = season_length
        self.last_season = None
    
    def fit(self, y: np.ndarray) -> 'SeasonalNaiveModel':
        """Store the last season's values."""
        self.last_season = y[-self.season_length:]
        self._is_fitted = True
        return self
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Predict using the corresponding value from the last season."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = np.zeros(n_periods)
        for i in range(n_periods):
            predictions[i] = self.last_season[i % self.season_length]
        return predictions


class MovingAverageModel(BaselineModel):
    """
    Moving Average baseline: predicts the average of the last k observations.
    
    This smooths out short-term fluctuations and can be effective for
    capturing the underlying trend.
    """
    
    def __init__(self, window: int = 7):
        super().__init__(f"MA_{window}")
        self.window = window
        self.ma_value = None
    
    def fit(self, y: np.ndarray) -> 'MovingAverageModel':
        """Calculate the moving average of the last window observations."""
        self.ma_value = np.mean(y[-self.window:])
        self._is_fitted = True
        return self
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Predict the moving average value for all future periods."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(n_periods, self.ma_value)


class DriftModel(BaselineModel):
    """
    Drift model: extrapolates the trend from the training data.
    
    This model assumes the time series will continue changing at the
    same average rate as observed in the training data.
    """
    
    def __init__(self):
        super().__init__("Drift")
        self.last_value = None
        self.drift_per_period = None
    
    def fit(self, y: np.ndarray) -> 'DriftModel':
        """Calculate the average drift per period."""
        self.last_value = y[-1]
        # Drift = (last value - first value) / (n - 1)
        self.drift_per_period = (y[-1] - y[0]) / (len(y) - 1)
        self._is_fitted = True
        return self
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Predict using drift extrapolation."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = np.zeros(n_periods)
        for i in range(n_periods):
            predictions[i] = self.last_value + (i + 1) * self.drift_per_period
        return predictions


class ExponentialSmoothingModel(BaselineModel):
    """
    Simple Exponential Smoothing baseline.
    
    This model gives more weight to recent observations and less weight
    to older ones, controlled by the smoothing parameter alpha.
    """
    
    def __init__(self, alpha: float = 0.3):
        super().__init__(f"ExpSmooth_{alpha}")
        self.alpha = alpha
        self.smoothed_value = None
    
    def fit(self, y: np.ndarray) -> 'ExponentialSmoothingModel':
        """Apply exponential smoothing to training data."""
        # Initialize with first observation
        self.smoothed_value = y[0]
        
        # Apply exponential smoothing
        for obs in y[1:]:
            self.smoothed_value = self.alpha * obs + (1 - self.alpha) * self.smoothed_value
        
        self._is_fitted = True
        return self
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Predict the smoothed value for all future periods."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(n_periods, self.smoothed_value)


def evaluate_baseline_models(
    train: np.ndarray,
    test: np.ndarray,
    models: Optional[list] = None
) -> pd.DataFrame:
    """
    Evaluate multiple baseline models and return comparison metrics.
    
    Args:
        train: Training time series data
        test: Test time series data
        models: List of BaselineModel instances (default: all standard baselines)
    
    Returns:
        DataFrame with evaluation metrics for each model
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    
    if models is None:
        models = [
            NaiveModel(),
            SeasonalNaiveModel(season_length=7),
            MovingAverageModel(window=7),
            MovingAverageModel(window=14),
            DriftModel(),
            ExponentialSmoothingModel(alpha=0.3),
            ExponentialSmoothingModel(alpha=0.5),
        ]
    
    results = []
    n_test = len(test)
    
    for model in models:
        # Fit and predict
        model.fit(train)
        predictions = model.predict(n_test)
        
        # Calculate metrics
        mae = mean_absolute_error(test, predictions)
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(test, predictions)
        
        results.append({
            'Model': model.name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        })
    
    return pd.DataFrame(results)


# Convenience function for quick baseline comparison
def get_baseline_forecasts(
    train: np.ndarray,
    n_periods: int
) -> pd.DataFrame:
    """
    Get forecasts from all baseline models.
    
    Args:
        train: Training time series data
        n_periods: Number of periods to forecast
    
    Returns:
        DataFrame with forecasts from each baseline model
    """
    models = [
        NaiveModel(),
        SeasonalNaiveModel(season_length=7),
        MovingAverageModel(window=7),
        DriftModel(),
        ExponentialSmoothingModel(alpha=0.3),
    ]
    
    forecasts = {}
    for model in models:
        model.fit(train)
        forecasts[model.name] = model.predict(n_periods)
    
    return pd.DataFrame(forecasts)
