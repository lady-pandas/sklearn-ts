import unittest

import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from sklearn_ts.datasets.covid import load_covid
from sklearn_ts.models.prophet import ProphetModel
from sklearn_ts.models.sarima import SARIMAXTimeSeriesModel
from sklearn_ts.models.trees import RandomForestTimeSeriesModel
from sklearn_ts.validator import check_model


class ValidatorTestCase(unittest.TestCase):

    def test_rf(self):
        dataset = pd.read_parquet('../tests/sarima_AR5_monthly.parquet')
        dataset = dataset[(dataset['distr'] == 'normal') & (dataset['mc'] == 0) & (dataset['batch'] == 'monthly5_3ahead')]
        dataset.index = dataset['date']
        dataset['lag'] = dataset['y'].shift(7)

        params = {'coverage': [0.9], 'features': [['lag']]}
        regressor = RandomForestTimeSeriesModel()

        results = check_model(
            regressor, params, dataset,
            target='y', features=['lag'], categorical_features=[], user_transformers=[],
            h=3, n_splits=10, gap=6,
            plotting=True
        )

        self.assertEqual(len(results.keys()), 10)

    def test_sarimax(self):
        dataset = pd.read_parquet('../tests/sarima_AR.parquet')
        dataset = dataset[(dataset['distr'] == 'normal') & (dataset['mc'] == 0) & (dataset['batch'] == 'daily_7ahead')]
        dataset.index = dataset['date']

        params = {'coverage': [0.8], 'order': [(0, 0, 0)], 'seasonal_order': [(1, 0, 0, 7)], 'trend': [[1]]}
        regressor = SARIMAXTimeSeriesModel()

        results = check_model(
            regressor, params, dataset,
            target='y', features=['date'], categorical_features=[], user_transformers=[],
            h=7, n_splits=12, gap=7,
            plotting=True
        )

        self.assertEqual(len(results.keys()), 10)

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

        self.assertEqual(len(results.keys()), 10)

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

        self.assertEqual(len(results.keys()), 10)
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

        self.assertEqual(len(results.keys()), 10)
        self.assertAlmostEqual(results['mape_cv'], 0.20885, 5)
