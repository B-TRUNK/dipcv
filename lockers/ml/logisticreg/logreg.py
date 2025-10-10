from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd

#=======================================
#Step_1 : Data Reading and Understancing
#=======================================
df = pd.read_excel('../../../data_files/images_analyzed_productivity.xlsx')
#print(df.head())
#plt.scatter(df.Age ,df.Productivity ,marker='+' ,color='Red')
#sizes = df['Productivity'].value_counts(sort=1)
#plt.pie(sizes ,shadow=True ,autopct='%1.1f%%')

#=============================
#Step_2 : Drop Irrelevant Data
#=============================
df.drop(['Images_Analyzed'],axis=1 ,inplace=True)
df.drop(['User'],axis=1 ,inplace=True)
#print(df.head())

#=================================
#Step_3 : Deal with Missing Values
#=================================
df = df.dropna()

#Step_4 : Convert Binary Data to Numeric Values 1, 0
#df.Productivity[df.Productivity == 'Good'] = 1
#df.Productivity[df.Productivity == 'Bad'] = 0
df.loc[df.Productivity == 'Good', "Productivity"] = 1
df.loc[df.Productivity == 'Bad', "Productivity"] = 0

#==============================================================================
#Step_5 : Prepare The Data for Learning(Define Dependent/Independent Variables)
#==============================================================================
Y = df['Productivity'].values #Object
Y = Y.astype('int') #int
X = df.drop(labels=['Productivity'] ,axis=1)
#print(X.head())
#print(df.head())

#================================================
#Step_6 : Split Data into Training & Testing Sets
#================================================
X_train ,X_test ,Y_train ,Y_test = train_test_split(X ,Y ,test_size=0.1 ,random_state=20)
#print(X_test)

#============================
#Step_7 : Define the ML Model
#============================
model = LogisticRegression()
model.fit(X_train ,Y_train)

#=======================
#Step_8 : Test the Model
#=======================
prediction_test = model.predict(X_test)

#=========================
#Step_9 : Verify the Model
#=========================
print("Accuracy = " ,metrics.accuracy_score(Y_test ,prediction_test)) 
# Accuracy = 0.68 with test_size=0.4 , 0.85 with test_size=0.1 ,note 0.05!!!
"""
if accuracy is low , get more data for training ,so decrease test size from .4 to .1
"""

#============================
#Step_10 : Weights Evaluation
#=================================
weights = pd.Series(model.coef_[0] ,index=X.columns.values)
print(weights)
#print(model.coef_)

plt.show()