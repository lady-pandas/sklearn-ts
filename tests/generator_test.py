import unittest

from sklearn_ts.datasets.sarimax_generator import generate_multiple_arma_datasets


class GeneratorsTestCase(unittest.TestCase):

    def test_generate_sarima(self):
        mc = 100

        details = {
            'daily_7ahead2': {'level': mc, 'freq': 'D', 'nsample': 7 * 52 * 2,
                             'ar': [0, 0, 0, 0, 0, 0, 0.2], 'ma': []},

            # 'daily_7ahead': {'level': mc, 'freq': 'D', 'nsample': 7 * 52 * 2,
            #                  'ar': [0, 0, 0, 0, 0, 0, 0.7], 'ma': []},

            # 'monthly5_3ahead': {'level': mc, 'freq': 'M', 'nsample': 12 * 12,
            #                    'ar': [0] * 2 + [0.5, -0.4, -0.2], 'ma': []}

            # 'monthly_3ahead': {'level': mc, 'freq': 'M', 'nsample': 12 * 6,
            #                    'ar': [0]*2 + [0.8, 0, 0.2] + [0]*7 + [0.2], 'ma': []}  # every quarter
        }

        datasets = generate_multiple_arma_datasets(name='AR_daily2', MC=mc, details=details)

        self.assertEqual(datasets.shape[0], (7 * 52 * 2) * mc * 3)
