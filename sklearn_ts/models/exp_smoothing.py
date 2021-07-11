import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


class ExpSmoothingRegressor(BaseEstimator, RegressorMixin):
    # https://www.statsmodels.org/devel/generated/statsmodels.tsa.exponential_smoothing.ets.ETSModel.html

    def __init__(self, error='add', trend=None, damped_trend=False, seasonal=None, seasonal_periods=None):
        self.error = error
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

        self.model = None
        self.predictions= None

    def fit(self, X, y):
        model = ETSModel(y, error=self.error, trend=self.trend, seasonal=self.seasonal,
                         damped_trend=self.damped_trend, seasonal_periods=self.seasonal_periods)
        self.model = model.fit(maxiter=10000)
        return self

    def predict(self, X):
        X = np.ndarray.flatten(X)
        self.predictions = self.model.get_prediction(start=min(X), end=max(X)).summary_frame(alpha=0.05)
        return self.predictions['mean']

    def get_params(self, deep=True):
        return {"error": self.error, 'trend': self.trend, 'damped_trend': self.damped_trend, 'seasonal': self.seasonal,
                'seasonal_periods': self.seasonal_periods}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
