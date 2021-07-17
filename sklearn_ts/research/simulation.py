import pandas as pd

from sklearn_ts.models.sarima import SARIMAXTimeSeriesModel
from sklearn_ts.models.trees import RandomForestTimeSeriesModel
from sklearn_ts.validator import check_model

import warnings
warnings.filterwarnings("ignore")


def run_model(dataset, params, regressor, model_details, features, performance, errors):
    if type(regressor).__name__ == 'SARIMAXTimeSeriesModel':
        dataset.index = pd.to_datetime(dataset['date'])
        features_list = ['date']
    else:
        dataset['lag'] = dataset['y'].shift(7)
        features_list = ['lag']

    try:
        results = check_model(
            regressor, params, dataset,
            target='y', features=features_list, categorical_features=[], user_transformers=[],
            h=7, n_splits=12, gap=60,
            plotting=True
        )

        results['performance_cv']['id'] = i
        results['features_importance']['id'] = i
        results['performance_cv']['model_details'] = model_details
        results['features_importance']['model_details'] = model_details

        features.append(results['features_importance'])
        performance.append(results['performance_cv'])
    except Exception as e:
        errors.append({'error': str(e), 'id': i, 'model': type(regressor).__name__})

    return features, performance, errors


level = 0
datasets = pd.read_parquet(f'sarima_level{level}.parquet')

features_array = []
performance_array = []
errors_array = []
# %%
for i in datasets['id'].unique():
    print(i)
    if i <= 1000:
        dataset = datasets[datasets['id'] == i].copy()

        params = {'coverage': [0.8], 'order': [(0, 0, 0)], 'seasonal_order': [(1, 0, 0, 7)], 'trend': [None]}
        regressor = SARIMAXTimeSeriesModel()
        run_model(dataset, params, regressor, 'SARIMA(0,0,0)(1,0,0)7', features_array, performance_array, errors_array)

        params = {'coverage': [0.8], 'order': [(7, 0, 0)], 'seasonal_order': [(0, 0, 0, 0)], 'trend': [None]}
        regressor = SARIMAXTimeSeriesModel()
        run_model(dataset, params, regressor, 'SARIMA(1,0,0)(0,0,0)0', features_array, performance_array, errors_array)

        params = {'coverage': [0.9], 'features': [['lag']]}
        regressor = RandomForestTimeSeriesModel()
        run_model(dataset, params, regressor, 'RF', features_array, performance_array, errors_array)


pd.concat(features_array).to_parquet('features.parquet')
pd.concat(performance_array).to_parquet('performance.parquet')
pd.DataFrame(errors_array).to_parquet('errors.parquet')
