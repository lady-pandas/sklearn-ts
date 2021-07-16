import unittest

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from sklearn_ts.datasets.covid import load_covid, generate_arma_dataset
from sklearn_ts.models.prophet import ProphetModel
from sklearn_ts.models.sarima import SARIMAXModel
from sklearn_ts.models.trees import RandomForestTimeSeriesModel
from sklearn_ts.validator import check_model, pi_coverage
import pandas as pd
import numpy as np
from scipy.stats import t



class ValidatorTestCase(unittest.TestCase):

    def test_rf(self):
        def norm(size):
            #return np.random.normal(loc=0.0, scale=5.0, size=size)
            return t.rvs(2, size=size)

        # -0.8, -0.5, 0.3, -0.1
        dataset = generate_arma_dataset(level=100, ar=[0,0,0,0,0,0,0.7], ma=[0], nsample=100, distrvs=norm)
        dataset['date'] = pd.to_datetime(pd.date_range(start='2021-01-01', periods=100, freq='D'))
        dataset.index = dataset['date']

        dataset['lag'] = dataset['y'].shift(7)

        params = {'coverage': [0.9], 'features': [['lag']]}
        regressor = RandomForestTimeSeriesModel()

        results = check_model(
            regressor, params, dataset,
            target='y', features=['lag'], categorical_features=[], user_transformers=[],
            h=7, n_splits=10, gap=7,
            plotting=True
        )


        self.assertEqual(len(results.keys()), 9)


    def test_sarimax(self):
        def norm(size):
            #return np.random.normal(loc=0.0, scale=5.0, size=size)
            return t.rvs(2, size=size)

        # -0.8, -0.5, 0.3, -0.1
        dataset = generate_arma_dataset(level=100, ar=[0,0,0,0,0,0,0.7], ma=[0], nsample=100, distrvs=norm)
        dataset['date'] = pd.to_datetime(pd.date_range(start='2021-01-01', periods=100, freq='D'))
        dataset.index = dataset['date']

        params = {'coverage': [0.9], 'order': [(7, 0, 0)]} #, 'seasonal_order': [(0, 0, 0, 7)], 'trend': [None], 'exog': [None]}
        regressor = SARIMAXModel()

        results = check_model(
            regressor, params, dataset,
            target='y', features=['date'], categorical_features=[], user_transformers=[],
            h=7, n_splits=10, gap=7,
            plotting=True
        )


        self.assertEqual(len(results.keys()), 9)

    def test_prophet(self):
        dataset = load_covid()['dataset']

        dataset['date'] = pd.to_datetime(dataset.index)
        params = {'features': [['date', 'month']], 'daily_seasonality': [True]}
        regressor = ProphetModel(target='new_cases', features=['date', 'month'], regressors=[],
                                 daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True,
                                 growth='linear'
                                 )

        results = check_model(
            regressor, params, dataset,
            target='new_cases', features=['date', 'month'], categorical_features=[], user_transformers=[],
            h=30, n_splits=5, gap=30,
            plotting=True
        )

        self.assertEqual(len(results.keys()), 9)

    def test_regression(self):
        dataset = load_covid()['dataset']

        params = {'fit_intercept': [False]}
        regressor = LinearRegression(fit_intercept=False)

        results = check_model(
            regressor, params, dataset,
            target='new_cases', features=['month'], categorical_features=[], user_transformers=[],
            h=30, n_splits=5, gap=30,
            plotting=True
        )

        self.assertEqual(len(results.keys()), 9)
        self.assertAlmostEqual(results['mape_cv'], 0.71713, 5)

    def test_SVR(self):
        dataset = load_covid()['dataset']

        params = {'C': [1.0]}
        regressor = SVR()

        results = check_model(
            regressor, params, dataset,
            target='new_cases', features=['month'], categorical_features=[], user_transformers=[('mm', MinMaxScaler())],
            h=14, n_splits=2, gap=14,
            plotting=True
        )

        self.assertEqual(len(results.keys()), 9)
        self.assertAlmostEqual(results['mape_cv'], 0.20885, 5)
