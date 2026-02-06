import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def coverage_probability(y_true, lower, upper):
    """Fraction of true values lying within the interval."""
    return np.mean((y_true >= lower) & (y_true <= upper))

def interval_width(lower, upper):
    """Average width of prediction intervals."""
    return np.mean(upper - lower)

def evaluate_point_forecast(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred)
    }

def evaluate_intervals(y_true, lower, upper):
    return {
        "Coverage": coverage_probability(y_true, lower, upper),
        "Width": interval_width(lower, upper)
    }