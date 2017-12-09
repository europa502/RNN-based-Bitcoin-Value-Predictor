# RNN-based-Bitcoin-Value-Predictor 
## Introduction

Recurrent Neural Networks are excellent to use along with time series analysis to predict stock prices. What is time series analysis? Time series analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. Time series forecasting is the use of a model to predict future values based on previously observed values.

An example is this. Would today affect the stock prices of tomorrow? Would last week affect the stock prices of tomorrow? How about last month? Last year? Seasons or fiscal quarters? Decades? Although stock advisors may have different opinions, recurrent neural networks uses every single case and finds the best method to predict.

Problem: Client wants to know when to invest to get largest return in Dec 2017.

Data: 7 years of Bitcoin prices. (2010-2017)

Solution: Use recurrent neural networks to predict Bitcoin prices in the first week of December 2017 using data from 2010-2017.

## Data

You can get up-to-date bitcoin prices in proper csv format from https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD

## Visualising The Data
```python
#Importing preprocessing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


training_set=pd.read_csv('BTCtrain.csv')      #reading csv file
training_set.head()                           #print first five rows

```
## Preprocessing the data
```python
training_set1=training_set.iloc[:,1:2]        #selecting the second column
training_set1.head()                          #print first five rows
training_set1=training_set1.values            #converting to 2d array
training_set1                                 #print the whole data

#Scaling the data

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()                           #scaling using normalisation 
training_set1 = sc.fit_transform(training_set1)
xtrain=training_set1[0:2694]                  #input values of rows [0-2694]		   
ytrain=training_set1[1:2695]                  #input values of rows [1-2695]

today=pd.DataFrame(xtrain[0:5])               #taking first file elements of the row from xtrain
tomorrow=pd.DataFrame(ytrain[0:5])            #taking first file elements of the row from ytrain
ex= pd.concat([today,tomorrow],axis=1)        #concat two columns 
ex.columns=(['today','tomorrow'])
xtrain = np.reshape(xtrain, (2694, 1, 1))     #Reshaping into required shape for Keras

```
## Building the RNN

```python
#importing keras and its packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


regressor=Sequential()                                                      #initialize the RNN

regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))      #adding input layerand the LSTM layer 

regressor.add(Dense(units=1))                                               #ading output layers

regressor.compile(optimizer='adam',loss='mean_squared_error')               #compiling the RNN

regressor.fit(xtrain,ytrain,batch_size=32,epochs=2000)                      #fitting the RNN to the training set  
```
## Making Predictions 

```python
# Reading CSV file into test set
test_set = pd.read_csv('BTCtest.csv')
test_set.head()


real_stock_price = test_set.iloc[:,1:2]         #selecting the second column

real_stock_price = real_stock_price.values      #converting to 2D array

#getting the predicted BTC value of the first week of Dec 2017  
inputs = real_stock_price			
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (8, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
```
## visualising the result 
```python

plt.plot(real_stock_price, color = 'red', label = 'Real BTC Value')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted BTC Value')
plt.title('BTC Value Prediction')
plt.xlabel('Days')
plt.ylabel('BTC Value')
plt.legend()
plt.show()
```

## Results 
![screenshot from 2017-12-09 05-40-01](https://user-images.githubusercontent.com/26405791/33792957-186dc214-dca6-11e7-9b11-2ffcc70027a4.png)


You can surely increase the accuracy up to a limit by increasing the epochs. 
```python 
regressor.fit(xtrain,ytrain,batch_size=32,epochs=2000)                      #fitting the RNN to the training set 
```

## Reference- 

[kimanalytics](https://github.com/kimanalytics/Recurrent-Neural-Network-to-Predict-Stock-Prices) - Recurrent Neural Network to Predict Tesla's Stock Prices
