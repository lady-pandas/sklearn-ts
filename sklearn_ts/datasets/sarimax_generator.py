import datetime
import math

import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.tsa.arima_process import arma_generate_sample


DEGREES = 2.1


def normal(size):
    return np.random.normal(loc=0.0, scale=math.sqrt(DEGREES/(DEGREES-1)), size=size)


def tstud_finite(size):
    return t.rvs(DEGREES, size=size)


def generate_one_arma_dataset(level, freq='D', **kwargs):
    kwargs['ar'] = [1] + [-el for el in kwargs['ar']]
    kwargs['ma'] = [1] + kwargs['ma']
    dataset = pd.DataFrame({'y': arma_generate_sample(**kwargs) + level})

    dataset['date'] = pd.to_datetime(
        pd.date_range(end=datetime.datetime.today().date(), periods=dataset.shape[0], freq=freq))
    return dataset


def generate_multiple_arma_datasets(MC, level, freq, **kwargs):
    ts_array = []
    i = 0
    for mc in range(MC):
        for distr in [normal, tstud_finite]:
            ts = generate_one_arma_dataset(level, freq=freq, distrvs=distr, **kwargs)
            ts['distr'] = distr.__name__
            ts['mc'] = mc
            ts['id'] = i
            ts_array.append(ts)

            i += 1

    datasets = pd.concat(ts_array)
    datasets.to_parquet(f'sarima_level{level}.parquet')
    return datasets
