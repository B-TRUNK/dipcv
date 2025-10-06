#this code is a practical example to linear regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

#read a csv file using pandas
df = pd.read_csv('../../data_files/cells.csv')
cells_predict_df = pd.read_csv('../../data_files/cells_predict.csv')
#print(df)

#plt.xlabel('time')
#plt.ylabel('cells')
#plt.scatter(df.time ,df.cells ,color="red" ,marker="+")

#in order to draw the line for a linear regression : 
# 1 - define the u=instance ,
# 2 - train the data to fit the line,
# 3 - predict some unknown values to test the model

# x-axis is the (independent variable - time)
# y-axis is the (dependent variable) - we predict y(y-hat)

#x_df = df.drop('cells' ,axis='columns') #method two, if only you have 1 variable on x, so you disclude the y

x_df = df[['time']] #the right method for any number of x-variables
y_df = df.cells


#creates an instance of the model
reg = linear_model.LinearRegression()
#fit the line(train data)
reg.fit(x_df ,y_df)

#data to be predicted
predicted_value = pd.DataFrame([[2.3]] ,columns = ['time'])
#predicted_value = reg.predict(np.array([[2.3]]))


#print(reg.score(x_df ,y_df))
m = reg.coef_
b = reg.intercept_
#print('Coefficient(m):' ,reg.coef_)
#print('Intercept(b):' ,reg.intercept_)

#print predicted value (method_1)
print('Pridected # cells ...' ,reg.predict(predicted_value))
#print predicted value (method_2)
print('Pridected # cells using y-hat = mx + b = ' ,((m*2.3) + b))


#================================================================
#predict for a bunch of data from an excel cheet

predicted_list = reg.predict(cells_predict_df)
print(predicted_list)

#add a new columns
cells_predict_df['cells'] = predicted_list
print(cells_predict_df)

#export to csv
cells_predict_df.to_csv('../../data_files/exported.csv')



plt.show()
