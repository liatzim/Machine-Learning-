# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # only opening price matters to us 

'''The data contains 4 years of trading days in the Google stock, taking this into account
we would predict the first month of 2017'''

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
'''each step look 60 days before time t, based on the trend try to predict the 
next output'''
'''60 is an experimental number, some cause overfitting and some miss catching importent trends'''
'''60 coressponds to the last 3 months of trading'''

X_train = [] # 60 previous financial open prices
y_train = [] # price at time t 
'''must start the loop at day 60 to get the previous 60 days'''
'''1258 the last day in our data set'''
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
'''shape[0] is the number of lines in the data
shape[1] number of columns 
1 is the number of predictors, for now its only the opening price'''
'''Creating 3 dimentions:
    1. The number of stock prices.
    2. Number of timesteps
    3. Number of indicators
'''
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout # to avoid overfitting

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation

'''
1. number of units in the LSTM layer = 50 , high dimentionality to catch all trends
2. Return seuqences when building a stacked layer
3. 3 dimentions shape , only the last two dimentions 
3. Droput = percentage of neurons to ignore, since 20% of 50 is 10 neurons, thats the amount that will be dropped out
'''
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
'''no need to mention sequences since its the last layer , no more sequences'''
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
'''Dense class to make the full connection, 1 unit the number of neurons in 
the output layer'''
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
'''to get the predicted price of each trading day in January 2017, 
we would need have the previous 60 days of each day, need both the 
training set and the test set, because we need December 2016''' 
'''if we concatenated the data as is, we would've had to normalize the test
results, but we are not allowed to do it'''
'''use the original train set to scale it correctly''' 
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    '''80 is upper bound because you only have 20 financial trading days in the test set''' 
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
