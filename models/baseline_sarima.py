from statsmodels.tsa.statespace.sarimax import SARIMAX


def train_sarima(series):
    model = SARIMAX(series, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
    fitted = model.fit(disp=False)
    return fitted
