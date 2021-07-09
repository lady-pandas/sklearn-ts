import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sklearn_ts.splitter import split, separate_data, custom_split


# If sklearn version without it
def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.mean(mape)


def plot_results(plotting, train, test, X_dummies_train, target, gs, model, model_name, i, mae_test):
    if plotting:
        print('Plot')

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

        # CV errors
        pred_cv_features = [f'pred_cv_{j}' for j in range(i)]
        mae_cv = abs(gs.best_score_)
        subset_cv = train.loc[train['pred_cv'].notnull(), ['date', target] + pred_cv_features]
        subset_cv.plot(x='date', y=[target] + pred_cv_features,
                       title='CV MAPE: {0:.0%}'.format(mae_cv), ax=axes[0][0])

        # Train errors:
        train['pred'] = model.predict(X_dummies_train)
        if hasattr(model.named_steps["regressor"], 'predictions'):
            train['pi_lower'] = model.named_steps["regressor"].predictions['pi_lower']
        else:
            train['pi_lower'] = None
        mae = mean_absolute_percentage_error(train[target], train['pred'])
        subset_train = train[['date', target, 'pred', 'pi_lower']]
        subset_train.plot(x='date', y=[target, 'pred', 'pi_lower'],
                          title='Train MAPE: {0:.0%}'.format(mae), ax=axes[0][1])

        # Test errors
        subset_test = test[['date', target, 'pred']]
        subset_test.plot(x='date', y=[target, 'pred'],
                         title='Testing MAPE: {0:.0%}'.format(mae_test), ax=axes[1][0])

        # Features
        # TODO SHAP
        no_features_imp = [
            'SVR', 'ExpSmoothingRegressor',
            'MLPRegressor', 'ANNRegressor', 'LSTMRegressor', 'TCNRegressor'
        ]
        if model_name not in no_features_imp:
            if model_name == "LinearRegression":
                impact = model.named_steps['regressor'].coef_
            else:
                impact = model.named_steps['regressor'].feature_importances_

            # not_dummies = [i for el in model.named_steps['preprocessor'].transformers_
            # if hasattr(el[1], 'get_feature_names') for i in el[1].get_feature_names()]
            features_df = pd.DataFrame({
                'impact': impact,
                'feature': list(X_dummies_train.columns),
            })
            features_df.sort_values('impact', ascending=True).plot.barh(x='feature', y='impact', ax=axes[1][1])

        fig.tight_layout(pad=4.0)
        fig.savefig(f'{model_name}.png')


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
    mae_test = mean_absolute_percentage_error(test[target], test['pred'])

    model_name = type(model.named_steps["regressor"]).__name__
    plot_results(plotting, train, test, X_dummies_train, target, gs, model, model_name, i, mae_test)

    # TODO remove regressor_ from best_params

    return {
        'model_name': type(pipeline.named_steps["regressor"]).__name__,
        'model': model,
        'mae_cv': abs(gs.best_score_),
        'std_cv': pd.DataFrame(gs.cv_results_).sort_values('rank_test_score')['std_test_score'].values[0],
        'mae_test': mae_test,
        'best_params': gs.best_params_,
        'cv_results': pd.DataFrame(gs.cv_results_)
    }
