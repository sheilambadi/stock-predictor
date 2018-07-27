import quandl
import pandas as pd
import numpy as np 
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

QUANDL_API_KEY = 'VTqCNpqhApPnXzS6Ksot'

quandl.ApiConfig.api_key = QUANDL_API_KEY

df = quandl.get("WIKI/AMZN")
# print(df.tail())

df = df[['Adj. Close']]

forecast_out = int(30)

df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
# print(df.tail())

# defining features and labels
X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)
# print(X)

X_forecast = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(df['Prediction'])
y = y[:-forecast_out]

# prediction
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# training
clf = LinearRegression()
clf.fit(X_train, y_train)

# testing
confidence = clf.score(X_test, y_test)
print('Confidence', confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)