import pandas as pd
import matplotlib.pyplot as plt


# list(X_dummies_train.columns)
def plot_features(model, features):
    model_name = type(model.named_steps["regressor"]).__name__
    features_df = None

    no_features_imp = [
        'SVR', 'ExpSmoothingRegressor', 'NaiveRegressor',
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
            'feature': features,
        })

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
        features_df.sort_values('impact', ascending=True).plot.barh(x='feature', y='impact', ax=ax)
        fig.tight_layout(pad=4.0)
        fig.savefig(f'{model_name}_features.png')

    return features_df
