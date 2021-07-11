from sklearn.base import BaseEstimator, RegressorMixin
from sktime.forecasting.naive import NaiveForecaster
import numpy as np
import pandas as pd


class NaiveRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, strategy='last', sp=7):
        self.strategy = strategy
        self.sp = sp

        self.model = None
        self.y_ = None

    def fit(self, X, y):
        forecaster = NaiveForecaster(strategy=self.strategy, sp=self.sp)

        # TODO inference frequency
        y = y.asfreq(pd.infer_freq(y.index))
        self.y_ = y
        forecaster.fit(y)
        self.model = forecaster
        return self

    def predict(self, X):
        # TODO not valid for fitting to train
        return self.model.predict(np.arange(X.shape[0]) + 1)

    def get_params(self, deep=True):
        return {'strategy': self.strategy, 'sp': self.sp}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
