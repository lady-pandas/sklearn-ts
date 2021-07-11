import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sklearn_ts.splitter import split, separate_data, custom_split


def mean_absolute_percentage_error(y_true, y_pred, zeros_strategy='mae'):
    """
    Similar to sklearn https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    with options for behaviour for around zeros
    :param y_true:
    :param y_pred:
    :param zeros_strategy:
    :return:
    """
    epsilon = np.finfo(np.float64).eps
    if zeros_strategy == 'epsilon':
        ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    elif zeros_strategy == 'mae':
        ae = np.abs(y_pred - y_true)
        ape = ae / np.maximum(np.abs(y_true), epsilon)
        # When true values are very small, we take MAE
        small_y_mask = y_true < epsilon
        ape = np.where(small_y_mask, ae, ape)
    else:
        raise ValueError(f'Undefined zeros_strategy {zeros_strategy}')

    return np.mean(ape)


def plot_results(plotting, train, test, X_dummies_train, target, gs, model, model_name, i, mae_test):
    if plotting:
        print('Plot')
        # Features
        # TODO SHAP
        # no_features_imp = [
        #     'SVR', 'ExpSmoothingRegressor', 'NaiveRegressor',
        #     'MLPRegressor', 'ANNRegressor', 'LSTMRegressor', 'TCNRegressor'
        # ]
        # if model_name not in no_features_imp:
        #     if model_name == "LinearRegression":
        #         impact = model.named_steps['regressor'].coef_
        #     else:
        #         impact = model.named_steps['regressor'].feature_importances_
        #
        #     # not_dummies = [i for el in model.named_steps['preprocessor'].transformers_
        #     # if hasattr(el[1], 'get_feature_names') for i in el[1].get_feature_names()]
        #     features_df = pd.DataFrame({
        #         'impact': impact,
        #         'feature': list(X_dummies_train.columns),
        #     })
        #
        #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
        #     features_df.sort_values('impact', ascending=True).plot.barh(x='feature', y='impact', ax=ax)
        #     fig.tight_layout(pad=4.0)
        #     fig.savefig(f'{model_name}_features.png')


        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

        # CV errors
        pred_cv_features = [f'pred_cv_{j}' for j in range(i)]
        mae_cv = abs(gs.best_score_)
        subset_cv = train.loc[train['pred_cv'].notnull(), [target] + pred_cv_features]
        subset_cv.plot(y=[target] + pred_cv_features,
                       title='CV MAPE: {0:.0%}'.format(mae_cv), ax=axes[0][0])

        # Train errors:
        train['pred'] = model.predict(X_dummies_train)
        mae = mean_absolute_percentage_error(train[target], train['pred'])
        subset_train = train[[target, 'pred']]
        subset_train.plot(y=[target, 'pred'],
                          title='Train MAPE: {0:.0%}'.format(mae), ax=axes[0][1])

        # Test errors zoomed
        subset_test = test[[target, 'pred', 'pi_lower', 'pi_upper']]
        subset_test.plot(y=[target, 'pred', 'pi_lower', 'pi_upper'],
                         title='Testing MAPE: {0:.0%}'.format(mae_test), ax=axes[1][0])

        # Test errors
        rejoined = subset_train.rename(columns={target: 'train'})[['train']].join(
            subset_test.rename(columns={target: 'test'})[['test', 'pred', 'pi_lower', 'pi_upper']], how='outer')
        rejoined.plot(y=['train', 'pred', 'test'],
                      title='Testing MAPE: {0:.0%}'.format(mae_test), ax=axes[1][1])

        for ax_sub in axes:
            for ax in ax_sub:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        fig.tight_layout(pad=4.0)
        fig.savefig(f'{model_name}.png')

        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
        # rejoined.plot(y=['train', 'pred', 'test', 'pi_lower', 'pi_upper'],
        #               title='Testing MAPE: {0:.0%}'.format(mae_test), ax=ax)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # fig.tight_layout(pad=4.0)
        # fig.savefig(f'{model_name}_prediction.png')

        return rejoined


# noinspection PyDefaultArgument
def check_model(regressor, params, dataset,
                target='new_cases', features=['date'], categorical_features=[], user_transformers=[],
                h=14, n_splits=5, gap=14,
                plotting=True
                ):
    """
    Check model
    :param gap:
    :param regressor:
    :param params:
    :param dataset:
    :param target:
    :param features:
    :param categorical_features:
    :param user_transformers:
    :param h:
    :param n_splits:
    :param plotting:
    :return:
    """
    # TODO info about dropped
    y, X, X_dummies = separate_data(dataset.dropna(), target, features, categorical_features)
    train, test = split(dataset.dropna(), h)
    X_dummies_train, X_dummies_test = split(X_dummies, h)
    y_train, y_test = split(y, h)

    cv = custom_split(train, h, n_splits=n_splits, gap=gap)

    # https://newbedev.com/sklearn-pipeline-get-feature-names-after-onehotencode-in-columntransformer
    # https://stackoverflow.com/questions/57528350/can-you-consistently-keep-track-of-column-labels-using-sklearns-transformer-api/57534118#57534118
    # https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
    # categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[('num', FunctionTransformer(), list(X_dummies.columns))] +
                     [(ut[0], ut[1], list(X_dummies.columns)) for ut in user_transformers]
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', regressor)])
    # rename params:
    params = {f'regressor__{k}': v for k, v in params.items()}

    # search parameters space
    # TODO change to log
    print('Grid search')
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        cv=cv[0]
    )
    # print(X_dummies_train.head())
    gs.fit(X_dummies_train, y_train)
    model = gs.best_estimator_

    # check cv results
    i = 0
    train['pred_cv'] = None
    for train_split, test_split in cv[1]:
        train[f'pred_cv_{i}'] = None
        # TODO prevent retraining
        model.fit(X_dummies_train.loc[train_split, :], y_train.loc[train_split])
        train.loc[test_split, f'pred_cv_{i}'] = model.predict(X_dummies_train.loc[test_split, :])
        train.loc[test_split, 'pred_cv'] = model.predict(X_dummies_train.loc[test_split, :])
        i += 1

    # fit model to train
    print('Fitting to train')
    model.fit(X_dummies_train, y_train)

    # check performance on test dataset
    test['pred'] = model.predict(X_dummies_test)
    if hasattr(model.named_steps["regressor"], 'predictions'):
        test['pi_lower'] = model.named_steps["regressor"].predictions['pi_lower']
        test['pi_upper'] = model.named_steps["regressor"].predictions['pi_upper']
    else:
        test['pi_lower'] = None
        test['pi_upper'] = None

    mape_test = mean_absolute_percentage_error(test[target], test['pred'])
    mae_test = mean_absolute_error(test[target], test['pred'])
    rmse_test = math.sqrt(mean_squared_error(test[target], test['pred']))

    model_name = type(model.named_steps["regressor"]).__name__
    rejoined = plot_results(plotting, train, test, X_dummies_train, target, gs, model, model_name, i, mape_test)

    # TODO remove regressor_ from best_params

    return {
        'model_name': type(pipeline.named_steps["regressor"]).__name__,
        'model': model,
        'mape_cv': abs(gs.best_score_),
        'std_cv': pd.DataFrame(gs.cv_results_).sort_values('rank_test_score')['std_test_score'].values[0],
        'performance_test': {'MAE': mae_test, 'MAPE': mape_test, 'RMSE': rmse_test},
        'best_params': gs.best_params_,
        'cv_results': pd.DataFrame(gs.cv_results_),
        'rejoined': rejoined,
    }


