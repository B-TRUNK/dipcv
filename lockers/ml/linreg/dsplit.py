"""
This Code is to split data sets some portion for training
while the other portion is for predicting
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split

#read a csv file using pandas
df = pd.read_csv('../../../data_files/cells.csv')
print(df)

x_df = df[['time']]
y_df = df.cells

#splitting_CSV
X_train ,X_test ,y_train ,y_test = train_test_split(x_df ,y_df ,test_size=0.4 ,random_state=10)
#random_state : is a random rate of keeping same data each time code excuted,
#  if == none then you will get completely data sets every time you run the code

#creates an instance of the model
reg = linear_model.LinearRegression()
#fit the line(train data)
reg.fit(X_train ,y_train)

#data to be predicted
prediction_test = reg.predict(X_test)
print(y_test ,prediction_test)

print("Mean Square Error between y_test and predicted = " ,np.mean(prediction_test - y_test)**2)

#Residual Plots
plt.scatter(prediction_test ,(prediction_test - y_test))
plt.hlines(y = 0 ,xmin = 200 ,xmax = 310)


plt.show()

