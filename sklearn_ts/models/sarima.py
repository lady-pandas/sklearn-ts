from numpy.linalg import LinAlgError
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn_ts.models.base import TimeSeriesModel
import pandas as pd


class SARIMAXTimeSeriesModel(BaseEstimator, RegressorMixin, TimeSeriesModel):
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    def __init__(self, coverage=0.9, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), exog=None,
                 trend=[1]):
        self.coverage = coverage
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog = exog
        self.trend = trend

        self.model = None
        self.predictions = None

    def fit(self, X, y):
        model = SARIMAX(endog=y,
                        order=self.order, seasonal_order=self.seasonal_order,
                        exog=self.exog, trend=self.trend)

        try:
            model_fit = model.fit(disp=False, maxiter=200)
        except LinAlgError as e:
            model_fit = model.fit(disp=False, maxiter=200,  initialization='approximate_diffuse')

        self.model = model_fit

        # if model_fit.mle_retvals['converged']:
        #     self.model = model_fit
        # else:
        #     model_fit = model.fit(disp=False, maxiter=200)
        #     if model_fit.mle_retvals['converged']:
        #         self.model = model_fit
        #     #else:
        #     #    raise Exception("Convergence problem")

        return self

    def predict(self, X):
        X = np.ndarray.flatten(X)
        predictions = self.model.get_prediction(start=min(X), end=max(X)).summary_frame(alpha=1-self.coverage)
        self.predictions = predictions[['mean_ci_lower', 'mean_ci_upper']]. \
            rename(columns={'mean_ci_lower': 'pi_lower', 'mean_ci_upper': 'pi_upper'})
        return predictions['mean'].values

    def get_params(self, deep=True):
        return {
            "coverage": self.coverage, 'trend': self.trend,
            'order': self.order, 'seasonal_order': self.seasonal_order, 'exog': self.exog
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_features(self):
        return pd.concat([pd.DataFrame(self.model.params).reset_index().\
            rename(columns={'index': 'feature', 0: 'impact'}), pd.DataFrame({'feature': ['converedg'], 'impact': [int(self.model.mle_retvals['converged'])]})])
