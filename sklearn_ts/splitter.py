import numpy as np
import pandas as pd


# TODO gap for example each month for half a year
def custom_split(data, h, n_splits, gap):
    """
    Splitting data into time-CV parts
    :param data: assumed to be already ordered
    :param h: forecasting horizon
    :param n_splits: number of splits
    :param gap: gap between performing forecasts, f.e. [n-h, n), [n-2(h+gap), n-h+gap), ...
    :return: splitted datasets, splitted indices (preferably: dates)
    """
    splits = []
    dates = []
    for i in range(0, n_splits):
        train = np.arange(data.shape[0] - h - i * gap)
        test = np.arange((data.shape[0] - h - i * gap), (data.shape[0] - i * gap))

        splits.append((train, test))
        dates.append((data.iloc[train].index, data.iloc[test].index))

    return splits, dates


def separate_data(dataset, target, features, categorical_features):
    y = dataset[target]
    X = dataset[features]
    X_dummies = pd.get_dummies(dataset[features], columns=categorical_features)
    return y, X, X_dummies


def split(dataset, h):
    train = dataset[:(dataset.shape[0]-h)].copy()
    test = dataset[(dataset.shape[0]-h):].copy()
    return train, test
