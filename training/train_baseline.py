from models.baseline_sarima import train_sarima

def train_(series):
    model = train_sarima(series)
    return model
