import unittest

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from sklearn_ts.datasets.covid import load_covid
from sklearn_ts.validator import check_model


class ValidatorTestCase(unittest.TestCase):

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

        self.assertEqual(len(results.keys()), 8)

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

        self.assertEqual(len(results.keys()), 8)
