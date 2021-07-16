from fbprophet import Prophet
from sklearn.base import BaseEstimator, RegressorMixin
from fbprophet.utilities import regressor_coefficients
import pandas as pd

from sklearn_ts.models.base import TimeSeriesModel


class ProphetModel(BaseEstimator, RegressorMixin, TimeSeriesModel):
    # https://facebook.github.io/prophet/docs/quick_start.html#python-api

    def __init__(self, target='new_cases', features=['date'], regressors=[], **kwargs):
        self.target = target
        self.features = features
        self.regressors = regressors
        self.kwargs = kwargs

        self.model = None
        self.predictions = None
        self.feature_importances_ = None

    def fit(self, X, y):
        df = pd.DataFrame(X, columns=self.features)

        # Necessary for prophet algo:
        df['ds'] = df['date']
        df['y'] = y.values

        m = Prophet(**self.kwargs)
        for regressor in self.regressors:
            m.add_regressor(regressor)
        m.fit(df)

        self.model = m
        if len(self.regressors) > 1:
            self.feature_importances_ = [None] + regressor_coefficients(m)['coef'].tolist()  # place for date

        return self

    def predict(self, X):
        df = pd.DataFrame(X, columns=self.features)
        df['ds'] = df['date']  # necessary for prophet
        predictions = self.model.predict(df)
        self.predictions = predictions[['ds', 'yhat_lower', 'yhat_upper']].\
            rename(columns={'yhat_lower': 'pi_lower', 'yhat_upper': 'pi_upper'})
        return predictions['yhat'].values

    def get_params(self, deep=True):
        return {
            **{"target": self.target, 'regressors': self.regressors, 'features': self.features},
            **self.kwargs
        }

    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         setattr(self, parameter, value)
    #     return self
