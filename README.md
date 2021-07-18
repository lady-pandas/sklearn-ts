# Welcome to sklearn-ts

Testing time series forecasting models made easy :)
This package leverages [scikit-learn](https://github.com/scikit-learn/scikit-learn), simply tuning it where needed for time series specific purposes.

Main features include:
- Moving window time split
    - train-test split
    - CV on moving window time splits
- Model wrappers:
    - Neural networks
    
Other python packages in the time series domain:
- [sktime](https://github.com/alan-turing-institute/sktime)
- [sktime-dl](https://github.com/sktime/sktime-dl)
- [darts](https://github.com/unit8co/darts)

# Installation

```bash
pip install sklearn-ts
```

# Quickstart
## Forecasting COVID-19 with Linear Regression
```python
from sklearn_ts.datasets.covid import load_covid
from sklearn.linear_model import LinearRegression
from sklearn_ts.validator import check_model

dataset = load_covid()['dataset']
dataset['month'] = dataset['date'].dt.month

params = {'fit_intercept': [False, True]}
regressor = LinearRegression()

results = check_model(
    regressor, params, dataset,
    target='new_cases', features=['month'], categorical_features=[], user_transformers=[],
    h=14, n_splits=2, gap=14,
    plotting=True
)
```

![alt text](tests\LinearRegression.png)

# Forecasting models

| Model family | Model | Univariate |
| ------------- |:-------------:| -----:|
| Benchmark | Naive | 1 |
| Exponential Smoothing | SES | 1 |
| Exponential Smoothing | Holt's linear | 1 |
| Exponential Smoothing | Holt-Winter | 1 |
| - | Prophet |  |
| Neural networks | ANN |  |
| Neural networks | LSTM |  |
| Neural networks | TCN |  |

# Documentation
Tutorial notebooks:
- [neural networks](https://colab.research.google.com/drive/1wSZPydSkIoGYh9VANgP_wTQe-wrhzY1w#scrollTo=_W2QP0dhCKFV)

# Development roadmap
- TCN przewaga
- Regularization
- XGBoost drawing
- FEATURES + SHAP
- x13
- prettier plot
- Handling many observations per date
- Constant window for forecasting
- For NN - chart of how it learned
- Logging
- Read the docs
- prod
- save picture optional
- PI Coverage
- Watermark
- OLS pi
- AIC / BIC
penalizing coefficients / weights
param vs hypreparams
reg l1 l2, drop out, data augment, eartly stopping
- one step ahead forecast and again forecast etc


JOURNAL
- residuals normality as part of performance evaluation
- decide which measure to show
- those without features and pi still working
- czasem się nie przelicza - co wtedy? Zliczać błędne / 100?
