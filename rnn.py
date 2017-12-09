#Imprting preprocessing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


training_set=pd.read_csv('BTCtrain.csv')   #reading csv file
training_set.head()			   #print first five rows

training_set1=training_set.iloc[:,1:2] 	   #selecting the second column
training_set1.head()			   #print first five rows
training_set1=training_set1.values	   #converting to 2d array
training_set1				   #print the whole data

sc = MinMaxScaler()			   #scaling using normalisation 
training_set1 = sc.fit_transform(training_set1)

xtrain=training_set1[0:2694]		   #input values of rows [0-2694]		   
ytrain=training_set1[1:2695]		   #input values of rows [1-2695]

today=pd.DataFrame(xtrain[0:5])		   #taking first file elements of the row from xtrain
tomorrow=pd.DataFrame(ytrain[0:5])         #taking first file elements of the row from ytrain
ex= pd.concat([today,tomorrow],axis=1)	   #concat two columns 
ex.columns=(['today','tomorrow'])

# Reshaping into required shape for Keras
xtrain = np.reshape(xtrain, (2694, 1, 1))

#importing keras and its packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


regressor=Sequential()			#initialize the RNN

regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))			#adding input layerand the LSTM layer 

regressor.add(Dense(units=1))		#ading output layers

regressor.compile(optimizer='adam',loss='mean_squared_error') 		#compiling the RNN

regressor.fit(xtrain,ytrain,batch_size=32,epochs=2000)		#fitting the RNN to the training set  




# Reading CSV file into test set
test_set = pd.read_csv('BTCtest.csv')
test_set.head()


real_stock_price = test_set.iloc[:,1:2]		#selecting the second column

real_stock_price = real_stock_price.values	#converting to 2D array

#getting the predicted BTC value of the first week of Dec 2017  
inputs = real_stock_price			
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (8, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualising the result 

plt.plot(real_stock_price, color = 'red', label = 'Real BTC Value')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted BTC Value')
plt.title('BTC Value Prediction')
plt.xlabel('Days')
plt.ylabel('BTC Value')
plt.legend()
plt.show()

