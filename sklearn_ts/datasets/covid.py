import pandas as pd


def load_covid():
    covid = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv")

    target = 'new_cases'
    date = 'date'

    dataset = covid[(covid['location'] == 'World')].copy()[[target, date]]
    dataset[date] = pd.to_datetime(dataset[date])
    dataset.index = dataset[date]

    dataset['month'] = dataset['date'].dt.month
    dataset = dataset.drop(columns=['date'])

    return {
        'target': target,
        'dataset': dataset,
    }
