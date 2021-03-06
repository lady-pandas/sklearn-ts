import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pingouin import normality

from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sklearn_ts.splitter import split, separate_data, custom_split


def pi_coverage(y_true, predictions):
    within_pi = (y_true.values >= predictions['pi_lower']) & (y_true.values <= predictions['pi_upper'])
    return sum(within_pi) / len(y_true)


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


def plot_results(plotting, measure_to_plot, train, test, X_dummies_train, target, gs, model, model_name, i, measures):
    if plotting:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

        # CV errors
        pred_cv_features = [f'pred_cv_{j}' for j in range(1, i+1)]
        mae_cv = measures.loc[~measures['fold'].isin(['test', 'train']), measure_to_plot].mean()
        subset_cv = train.loc[train['pred_cv'].notnull(), [target] + pred_cv_features]
        subset_cv.plot(y=[target] + pred_cv_features,
                       title=f'CV {measure_to_plot.upper()}: {mae_cv: .00%}', ax=axes[0][0])

        # Train errors:
        train['pred'] = model.predict(X_dummies_train)
        mae = mean_absolute_percentage_error(train[target], train['pred'])
        subset_train = train[[target, 'pred']]
        # subset_train.plot(y=[target, 'pred'],
        #                   title='Train MAPE: {0:.0%}'.format(mae), ax=axes[0][1])
        train['error'] = train[target] - train['pred_cv']

        is_normal = normality(train['error'][train['error'].notnull()].astype(float))['normal'].iloc[0]
        train['error'].plot.hist(ax=axes[0][1], title=f'NORMAL: {"True" if is_normal else "False"}')

        # Test errors zoomed
        mape_test = measures.loc[measures['fold'].isin(['test']), measure_to_plot].mean()

        subset_test = test[[target, 'pred', 'pi_lower', 'pi_upper']]
        subset_test.plot(y=[target, 'pred'],
                         title=f'Test {measure_to_plot.upper()}: {mape_test:.00%}', ax=axes[1][0])
        (axes[1][0]).fill_between(x=test.index, y1=test['pi_lower'],
                                  y2=test['pi_upper'], zorder=3, color='grey', alpha=0.2)

        # Test errors
        rejoined = subset_train.rename(columns={target: 'train'})[['train']].join(
            subset_test.rename(columns={target: 'test'})[['test', 'pred', 'pi_lower', 'pi_upper']], how='outer')
        rejoined.plot(y=['train', 'pred', 'test'],
                      title=f'Test {measure_to_plot.upper()}: {mape_test:.00%}', ax=axes[1][1])

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

        return rejoined, is_normal


# noinspection PyDefaultArgument
def check_model(regressor, params, dataset,
                target='new_cases', features=['date'], categorical_features=[], user_transformers=[],
                h=14, n_splits=5, gap=14,
                plotting=True, measure_to_plot='pi',
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
    # print('Grid search')
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
    i = n_splits
    train['pred_cv'] = None
    features_array = []
    pi = None
    performance_cv = []
    for train_split, test_split in cv[1]:
        train[f'pred_cv_{i}'] = None
        # TODO prevent retraining

        model.fit(X_dummies_train.loc[train_split, :], y_train.loc[train_split])
        is_normal = \
        normality(train.loc[train_split, target] - model.predict(X_dummies_train.loc[train_split, :]))['normal'].iloc[0]

        train.loc[test_split, f'pred_cv_{i}'] = model.predict(X_dummies_train.loc[test_split, :])
        mape = mean_absolute_percentage_error(train.loc[test_split, target], train.loc[test_split, f'pred_cv_{i}'])

        features = model.named_steps["regressor"].get_features()
        if features is not None:
            features['fold'] = str(i)
            features['model'] = type(model.named_steps["regressor"]).__name__
            features['n'] = len(train_split)
            features_array.append(features)

        # TODO pi
        if hasattr(model.named_steps["regressor"], 'predictions'):
            pi = pi_coverage(train.loc[test_split, target],
                             model.named_steps["regressor"].predictions[['pi_upper', 'pi_lower']])

        train.loc[test_split, 'pred_cv'] = model.predict(X_dummies_train.loc[test_split, :])

        performance_cv.append({
            'mape': mape,
            'pi': pi,
            'fold': str(i),
            'n': len(train_split),
            'is_normal': is_normal,
        })

        i -= 1

    # fit model to train
    # print('Fitting to train')
    model.fit(X_dummies_train, y_train)

    features = model.named_steps["regressor"].get_features()
    if features is not None:
        features['fold'] = 'test'
        features['model'] = type(model.named_steps["regressor"]).__name__
        features['n'] = X_dummies_train.shape[0]
        features_array.append(features)

    # check performance on test dataset
    test['pred'] = model.predict(X_dummies_test)
    if hasattr(model.named_steps["regressor"], 'predictions'):
        test['pi_lower'] = model.named_steps["regressor"].predictions['pi_lower'].values
        test['pi_upper'] = model.named_steps["regressor"].predictions['pi_upper'].values
        pi_coverage_test = pi_coverage(test[target], test[['pi_lower', 'pi_upper']])
    else:
        test['pi_lower'] = None
        test['pi_upper'] = None

    mape_test = mean_absolute_percentage_error(test[target], test['pred'])
    mae_test = mean_absolute_error(test[target], test['pred'])
    rmse_test = math.sqrt(mean_squared_error(test[target], test['pred']))
    is_normal = normality(train[target] - model.predict(X_dummies_train))['normal'].iloc[0]

    performance_cv.append({
        'mape': mape_test,
        'pi': pi_coverage_test,
        'fold': 'test',
        'n': X_dummies_train.shape[0],
        'is_normal': is_normal,
    })

    model_name = type(model.named_steps["regressor"]).__name__
    performance_cv = pd.DataFrame(performance_cv)
    performance_cv['model'] = model_name

    rejoined, is_normal = plot_results(plotting, measure_to_plot, train, test, X_dummies_train, target, gs, model, model_name, n_splits,
                            performance_cv)

    # TODO remove regressor_ from best_params
    # choose measure to print on charts

    return {
        'model_name': type(pipeline.named_steps["regressor"]).__name__,
        'model': model,
        'performance_cv': performance_cv,
        'std_cv': pd.DataFrame(gs.cv_results_).sort_values('rank_test_score')['std_test_score'].values[0],
        'performance_test': {'MAE': mae_test, 'MAPE': mape_test, 'RMSE': rmse_test,
                             'PI_COVERAGE': pi_coverage_test},
        'best_params': gs.best_params_,
        'cv_results': pd.DataFrame(gs.cv_results_),
        'rejoined': rejoined,
        'features': list(X_dummies.columns),
        'features_importance': pd.concat(features_array) if len(features_array) > 0 else pd.DataFrame(),
    }
