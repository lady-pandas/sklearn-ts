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
| Neural networks | ANN | 1 |
| Neural networks | LSTM | 1 |
| Neural networks | TCN | 1 |

# Documentation
Tutorial notebook preparation in progress.

# Development roadmap
- New repo
- Remove old deploy from test
- Pypi
- Exploding MAPE
- Handling many observations per date
- Constant window for forecasting
- Tutorial notebooks