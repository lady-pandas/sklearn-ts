import unittest

import pandas as pd
from matplotlib import pyplot as plt

from sklearn_ts.features.seasonality import generate_fourier, add_fourier_to_X


class FourierTestCase(unittest.TestCase):
    def test_Fourier_simple(self):
        fourier = generate_fourier(0, 2, periods=[7], N=[1], with_intercept=False)
        fourier_ref = pd.DataFrame({
            'fourier_sin_7_1': [0.0, 0.78183],
            'fourier_cos_7_1': [1.0, 0.62349],
        })
        self.assertIsNone(pd.testing.assert_frame_equal(fourier, fourier_ref, check_less_precise=True))

    def test_Fourier_add_toX(self):
        X = pd.DataFrame({'trend': [0, 1]}, index=['2021-07-01', '2021-07-02'])
        X_with_fourier = add_fourier_to_X(X, periods=[7], N=[1], with_intercept=False)
        X_with_fourier_ref = pd.DataFrame({
            'trend': [0, 1],
            'fourier_sin_7_1': [0.0, 0.78183],
            'fourier_cos_7_1': [1.0, 0.62349],
        }, index=['2021-07-01', '2021-07-02'])

        self.assertIsNone(pd.testing.assert_frame_equal(X_with_fourier, X_with_fourier_ref,
                                                        check_less_precise=True))

    def test_Fourier_plotting(self):
        fourier = generate_fourier(0, 24, periods=[7, 12], N=[1, 1], with_intercept=False)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        fourier.plot.line(ax=axes,
                          color=['orange', 'orange', 'blue', 'blue'],
                          style=['-', '--', '-', '--'],
                          title='Fourier components'
                          )
        fig.savefig(f'Fourier.png')

        self.assertEqual(fourier.shape[1], 4)
