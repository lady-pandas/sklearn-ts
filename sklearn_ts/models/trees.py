import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

from sklearn_ts.models.base import TimeSeriesModel


class RandomForestTimeSeriesModel(BaseEstimator, RegressorMixin, TimeSeriesModel):
    # https: // scikit - learn.org / stable / modules / generated / sklearn.ensemble.RandomForestRegressor.html

    def __init__(self, n_estimators=100, criterion='mae', max_depth=None, min_samples_leaf=1, random_state=None,
                 coverage=0.9, features=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.coverage = coverage

        self.features = features

        self.model = None
        self.residuals = None
        self.predictions = None

    def fit(self, X, y):
        model = RandomForestRegressor()
        model.fit(X, y)
        self.model = model

        predictions = self.model.predict(X)
        self.residuals = predictions - y

        return self

    def predict(self, X):
        point_forecast = self.model.predict(X)
        quantiles = self.residuals.quantile( [(1.0 - self.coverage)/2, 1.0 - (1.0 - self.coverage)/2] )  # not dependent on horizon
        self.predictions = pd.DataFrame({
            'pi_lower': point_forecast + quantiles.iloc[0],
            'pi_upper': point_forecast + quantiles.iloc[1]
        })
        return point_forecast

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators, 'criterion': self.criterion,
            'max_depth': self.max_depth, 'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state, 'coverage': self.coverage,
            'features': self.features,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_features(self):
        return pd.DataFrame({
                'feature': self.features,
                'impact': self.model.feature_importances_,
        })
