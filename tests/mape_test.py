import unittest

import pandas as pd

from sklearn_ts.validator import mean_absolute_percentage_error


class MAPETestCase(unittest.TestCase):
    def test_MAPE_equals(self):
        y_true = pd.Series([0, 1])
        y_pred = pd.Series([0, 1])
        mape = mean_absolute_percentage_error(y_true, y_pred)
        self.assertEqual(mape, 0)

    def test_MAPE_zeros(self):
        y_true = pd.Series([0, 0, 1])
        y_pred = pd.Series([0.2, 0.4, 1])
        mape = mean_absolute_percentage_error(y_true, y_pred)
        self.assertEqual(round(mape, 10), 0.2)

    def test_MAPE_zeros_epsilon(self):
        y_true = pd.Series([0, 0, 1])
        y_pred = pd.Series([0.2, 0.4, 1])
        mape = mean_absolute_percentage_error(y_true, y_pred, zeros_strategy='epsilon')
        self.assertGreater(mape, 10**6)
