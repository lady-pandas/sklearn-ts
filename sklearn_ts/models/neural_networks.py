from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from tcn import TCN
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class ANNRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, learning_rate=0.001, loss='mean_squared_error'):
        self.loss = loss
        self.learning_rate = learning_rate

        self.history = None
        self.model = None

    def fit(self, X, y):
        model = Sequential()
        model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        adam = Adam(learning_rate=self.learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam)

        self.history = model.fit(X, y, epochs=100, batch_size=10, verbose=0, validation_split=0.1)

        self.model = model
        return self

    def predict(self, X):
        return np.squeeze(self.model.predict(X))

    def get_params(self, deep=True):
        return {'loss': self.loss, 'learning_rate': self.learning_rate}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class LSTMRegressor(BaseEstimator, RegressorMixin):
    # TODO automatic feature creation
    # X_pd = pd.DataFrame(X)
    # cols = X_pd.columns
    # lags = [1, 2, 3]
    # lagged_features = []
    # for lag in lags:
    #   for feature in cols:
    #     X_pd[f'{feature}_{lag}_lag'] = X_pd[feature].shift(lag)
    #     lagged_features.append(f'{feature}_{lag}_lag')
    # trainX = np.reshape(X_pd[lagged_features].values, (X.shape[0], len(lags), len(cols)))
    # X_pd.head()

    def __init__(self, learning_rate=0.001, loss='mean_squared_error'):
        self.loss = loss
        self.learning_rate = learning_rate

        self.history = None
        self.model = None

    def fit(self, X, y):
        model = Sequential()
        # model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(32, activation='relu'))  # , input_shape=(X.shape[1], 1)
        model.add(Dense(1))

        adam = Adam(learning_rate=self.learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam)

        # reshape input to be [samples, time steps, features]
        reshaped_X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        reshaped_y = np.reshape(y.values, (y.shape[0], 1))

        es = EarlyStopping(monitor='val_loss', patience=20)
        self.history = model.fit(reshaped_X, reshaped_y,
                                 epochs=400, batch_size=1, verbose=0, callbacks=[es], validation_split=0.1)

        self.model = model
        return self

    def predict(self, X):
        # TODO train and predict several times to have average
        reshaped_X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return np.squeeze(self.model.predict(reshaped_X))

    def get_params(self, deep=True):
        return {'loss': self.loss, 'learning_rate': self.learning_rate}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# https://github.com/philipperemy/keras-tcn
class TCNRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, learning_rate=0.001, loss='mean_squared_error'):
        self.loss = loss
        self.learning_rate = learning_rate

        self.history = None
        self.model = None

    def fit(self, X, y):
        tcn_layer = TCN(input_shape=(X.shape[1], 1))

        m = Sequential([
            tcn_layer,
            Dense(1)
        ])

        adam = Adam(learning_rate=self.learning_rate)
        m.compile(optimizer=adam, loss='mse')

        reshaped_X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        reshaped_y = np.reshape(y.values, (y.shape[0], 1))

        es = EarlyStopping(monitor='val_loss', patience=20)
        self.history = m.fit(reshaped_X, reshaped_y, epochs=500, validation_split=0.1, verbose=0, callbacks=[es])

        self.model = m
        return self

    def predict(self, X):
        reshaped_X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return np.squeeze(self.model.predict(reshaped_X))

    def get_params(self, deep=True):
        return {'loss': self.loss, 'learning_rate': self.learning_rate}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
