import datetime
import math

import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.tsa.arima_process import arma_generate_sample


DEGREES = 2.1


def normal(size):
    return np.random.normal(loc=0.0, scale=math.sqrt(DEGREES/(DEGREES-2)), size=size)


def tstud_finite(size):
    return t.rvs(DEGREES, size=size)


def tstud_infinite(size):
    return t.rvs(1, size=size)


def generate_one_arma_dataset(level, freq='D', **kwargs):
    kwargs['ar'] = [1] + [-el for el in kwargs['ar']]
    kwargs['ma'] = [1] + kwargs['ma']
    dataset = pd.DataFrame({'y': arma_generate_sample(**kwargs) + level})

    dataset['date'] = pd.to_datetime(
        pd.date_range(end=datetime.datetime.today().date(), periods=dataset.shape[0], freq=freq))
    return dataset


def generate_multiple_arma_datasets(name, MC, details):
    ts_array = []
    i = 0
    for key, value in details.items():
        print(key, value)
        # read params
        level = value['level']
        freq = value['freq']
        nsample = value['nsample']
        ar = value['ar']
        ma = value['ma']

        for distr in [normal, tstud_finite, tstud_infinite]:
            print(freq, level, distr)

            for mc in range(MC):
                ts = generate_one_arma_dataset(level, freq=freq, distrvs=distr,
                                               nsample=nsample, ar=ar, ma=ma)
                ts['distr'] = distr.__name__
                ts['mc'] = mc
                ts['id'] = i

                ts['batch'] = key
                ts['level'] = level
                ts['freq'] = freq
                ts['nsample'] = nsample
                ts['ar'] = ','.join([str(el) for el in ar])
                ts['ma'] = ','.join([str(el) for el in ma])

                ts_array.append(ts)

                i += 1

        datasets = pd.concat(ts_array)
        datasets.to_parquet(f'sarima_{name}.parquet')

    datasets = pd.concat(ts_array)
    datasets.to_parquet(f'sarima_{name}.parquet')
    #pd.read_parquet(f'sarima_AR.parquet')
    return datasets
