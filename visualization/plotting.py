import matplotlib.pyplot as plt


def plot_forecast(y_true, y_pred, lower=None, upper=None):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Prediction")
    if lower is not None and upper is not None:
        plt.fill_between(range(len(lower)), lower, upper, alpha=0.3, label="Confidence Interval")
    plt.legend()
    plt.title("Forecast with Uncertainty")
    plt.show()
