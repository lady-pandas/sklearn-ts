import pandas as pd

from sklearn_ts.models.sarima import SARIMAXTimeSeriesModel
from sklearn_ts.models.trees import RandomForestTimeSeriesModel
from sklearn_ts.validator import check_model

import warnings

warnings.filterwarnings("ignore")


def run_model(dataset, params, regressor, name, h, n_splits, gap, lags, features, performance, errors):
    if type(regressor).__name__ == 'SARIMAXTimeSeriesModel':
        dataset.index = pd.to_datetime(dataset['date'])
        features_list = ['date']
    else:
        features_list = []
        for lag in lags:
            dataset[f'lag_{lag}'] = dataset['y'].shift(lag)
            features_list.append(f'lag_{lag}')

    try:
        results = check_model(
            regressor, params, dataset,
            target='y', features=features_list, categorical_features=[], user_transformers=[],
            h=h, n_splits=n_splits, gap=gap,
            plotting=True
        )

        results['performance_cv']['id'] = i
        results['features_importance']['id'] = i
        results['performance_cv']['model_details'] = name
        results['features_importance']['model_details'] = name
        results['performance_cv']['data'] = 'AR'
        results['features_importance']['data'] = 'AR'

        features.append(results['features_importance'])
        performance.append(results['performance_cv'])
    except Exception as e:
        errors.append({'error': str(e), 'id': i, 'model': type(regressor).__name__, 'data': 'AR'})

    return features, performance, errors



datasets = pd.read_parquet(f'sarima_AR.parquet')

features_array = []
performance_array = []
errors_array = []

details = {
    'daily_7ahead': {'order': (0, 0, 0), 'seasonal_order': (1, 0, 0, 7), 'trend': [1], 'lags': [7],
                     'h': 7, 'n_splits': 12, 'gap': 60, 'coverage': 0.8},

    'monthly_1ahead': {'order': (4, 0, 0), 'seasonal_order': (2, 0, 0, 12), 'trend': [1], 'lags': [3, 4, 12, 24],
                       'h': 1, 'n_splits': 20, 'gap': 3, 'coverage': 0.8},
}

for i in datasets['id'].unique():
    print(i)

    dataset = datasets[datasets['id'] == i].copy()
    config = details[dataset['batch'].max()]
    print(dataset['batch'].max())

    params = {'coverage': [config['coverage']], 'order': [config['order']],
              'seasonal_order': [config['seasonal_order']], 'trend': [config['trend']]}
    regressor = SARIMAXTimeSeriesModel()
    run_model(dataset, params, regressor,
              dataset['batch'].max(), config['h'], config['n_splits'], config['gap'], config['lags'],
              features_array, performance_array, errors_array)

    params = {'coverage': [config['coverage']], 'features': [[f'lag_{lag}' for lag in config['lags']]]}
    regressor = RandomForestTimeSeriesModel()
    run_model(dataset, params, regressor,
              dataset['batch'].max(), config['h'], config['n_splits'], config['gap'], config['lags'],
              features_array, performance_array, errors_array)

    print('done')

pd.concat(features_array).to_parquet('features.parquet')
pd.concat(performance_array).to_parquet('performance.parquet')
pd.DataFrame(errors_array).to_parquet('errors.parquet')
