import numpy as np
import pandas as pd


def add_fourier_to_X(X, t_start=0, periods=[7], N=[10], with_intercept=False):
    fourier = generate_fourier(t_start, X.shape[0] + t_start, periods, N, with_intercept)
    fourier.index = X.index
    return X.join(fourier)


def generate_fourier(t_start, t_end, periods=[7], N=[10], with_intercept=False):
    t = np.arange(t_start, t_end)

    fourier_dict = {}
    for i, period in enumerate(periods):
        n_vec = np.arange(start=0 if with_intercept else 1, stop=N[i] if with_intercept else N[i] + 1)
        for n in n_vec:
            x = 2 * np.pi * n * t / period
            fourier_dict[f'fourier_sin_{period}_{n}'] = np.sin(x)
            fourier_dict[f'fourier_cos_{period}_{n}'] = np.cos(x)

    return pd.DataFrame(fourier_dict)
