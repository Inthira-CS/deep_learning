import numpy as np
from data.generate_data import generate_time_series

from data.preprocess import create_sequences
from training.train_lstm import train_lstm
from training.train_mc_dropout import train as train_mc
from evaluation.evaluate_models import evaluate_point_forecast, evaluate_intervals
from models.baseline_sarima import train_sarima
from visualization.helpers import mc_dropout_predict
from visualization.plotting import plot_forecast


# Generate dataset
df = generate_time_series()
values = df["value"].values

# Prepare sequences
seq_length = 10
X, y = create_sequences(values, seq_length)
X = X[..., np.newaxis]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train LSTM
lstm_model = train_lstm(X_train, y_train)
y_pred = lstm_model.predict(X_test).flatten()

print("LSTM Metrics:", evaluate_point_forecast(y_test, y_pred))
# Train and evaluate SARIMA baseline
sarima_pred = train_sarima(values, split)

print("SARIMA Metrics:", evaluate_point_forecast(y_test, sarima_pred))

# Train MC Dropout LSTM
mc_model = train_mc(X_train, y_train)
mean_pred, std_pred = mc_dropout_predict(mc_model, X_test)
# Using 1.96 multiplier assuming approximate Gaussian predictive distribution (95% CI)

lower = mean_pred - 1.96 * std_pred
upper = mean_pred + 1.96 * std_pred

print("MC Dropout Metrics:", evaluate_point_forecast(y_test, mean_pred))
print("Interval Metrics:", evaluate_intervals(y_test, lower, upper))
print("Prediction Interval Width:", (upper - lower).mean())

plot_forecast(y_test, mean_pred, lower, upper)
