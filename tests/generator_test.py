import unittest

from sklearn_ts.datasets.sarimax_generator import generate_multiple_arma_datasets


class GeneratorsTestCase(unittest.TestCase):

    def test_generate_sarima(self):
        mc = 100

        details = {
            'daily_7ahead': {'level': mc, 'freq': 'D', 'nsample': 7 * 52 * 2,
                             'ar': [0, 0, 0, 0, 0, 0, 0.7], 'ma': []},

            'monthly_1ahead': {'level': mc, 'freq': 'M', 'nsample': 12 * 6,
                               'ar': [0]*2 + [0.5, -0.3] + [0]*7 + [0.7] + [0]*11 + [0.3], 'ma': []}  # every quarter
        }

        datasets = generate_multiple_arma_datasets(name='AR', MC=mc, details=details)

        self.assertEqual(datasets.shape[0], (7 * 52 * 2 + 12 * 6) * mc * 3)
