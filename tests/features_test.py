import unittest

from sklearn.linear_model import LinearRegression

from sklearn_ts.datasets.covid import load_covid
from sklearn_ts.features.explainer import plot_features
from sklearn_ts.validator import check_model


class FeaturesTestCase(unittest.TestCase):

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

        features_df = plot_features(results['model'], results['features'])

        self.assertEqual(features_df.shape[0], 1)
