import pandas as pd


def load_dataset(path="data/synthetic_series.csv"):
    return pd.read_csv(path)
