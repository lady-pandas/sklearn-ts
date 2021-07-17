import unittest

from sklearn_ts.datasets.sarimax_generator import generate_multiple_arma_datasets


class GeneratorsTestCase(unittest.TestCase):

    def test_generate_sarima(self):
        datasets = generate_multiple_arma_datasets(
            MC=100, level=0, freq='D',
            nsample=7*52*2, ar=[0, 0, 0, 0, 0, 0, 0.7], ma=[]
        )

        self.assertEqual(datasets.shape[0], 7*52*2 * 100 * 2)
