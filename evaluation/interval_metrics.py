import numpy as np


def coverage_probability(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))


def interval_width(lower, upper):
    return np.mean(upper - lower)
