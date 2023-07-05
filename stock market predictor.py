import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt

# Override pandas_datareader's get_data_yahoo() method
yf.pdr_override()

# Fetch data
df = pdr.get_data_yahoo('BTC-USD', start='2018-01-01', end='2023-01-01')

# Use 'Close' price for prediction
df = df[['Close']]

# Predict for the next 30 days
forecast_out = 30

# Create another column (the target) shifted 'n' units up
df['Prediction'] = df[['Close']].shift(-forecast_out)

# Create the independent and dependent data sets
X = df.drop(['Prediction'], 1)[:-forecast_out]
y = df['Prediction'][:-forecast_out]

# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the Linear Regression Model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Test the model using score
lr_confidence = lr.score(x_test, y_test)

# Set x_forecast equal to the last 30 rows of the original data set from Close column
x_forecast = df.drop(['Prediction'], 1)[:-forecast_out]
x_forecast = x_forecast.tail(forecast_out)

# Print linear regression model predictions for the next 'n' days
lr_prediction = lr.predict(x_forecast)

# Plotting the predictions
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Close Price History')
plt.plot(df.index[-len(lr_prediction):], lr_prediction, label='Predicted Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()
