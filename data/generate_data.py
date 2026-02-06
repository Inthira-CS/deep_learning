import numpy as np
import pandas as pd

print("generate_data.py loaded")
def generate_time_series(n_samples=1500):
    np.random.seed(42)
    time = np.arange(n_samples)

    trend = time * 0.01
    seasonality = 10 * np.sin(2 * np.pi * time / 50)
    noise = np.random.normal(0, 2, n_samples)

    series = trend + seasonality + noise
    df = pd.DataFrame({"value": series})
    return df


if __name__ == "__main__":
    df = generate_time_series()
    df.to_csv("data/synthetic_series.csv", index=False)
    print("Synthetic dataset saved.")

