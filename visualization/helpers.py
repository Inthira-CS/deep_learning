import numpy as np


def mc_dropout_predict(model, X, n_samples=100):
    preds = [model(X, training=True).numpy().flatten() for _ in range(n_samples)]
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)
