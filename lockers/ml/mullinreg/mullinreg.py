import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


df = pd.read_excel('../../../data_files/images_analyzed.xlsx')
#print(df.head())

"""
sns.lmplot(x = 'Time' ,y = 'Images_Analyzed' ,data = df ,hue = 'Age')
sns.lmplot(x = 'Coffee' ,y = 'Images_Analyzed' ,data = df ,hue = 'Age')
"""

reg = linear_model.LinearRegression()
reg.fit(df[ ['Time' ,'Coffee' ,'Age']] ,df.Images_Analyzed)

print(reg.coef_)
print(reg.predict([[13 ,2 ,34]]))

plt.show()