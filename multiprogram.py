'''
1.import the library
2.importing data set
3.one hot encoding to convert in binary state name
4.Training and tseting
5.training  the multi linear regression model
6.test the multi linear regression model
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d1=pd.read_csv('50_Startups.csv')
X=d1.iloc[:,:-1].values
y=d1.iloc[:,-1].values
# print(X)
# print(y)
#one hot encoding to convert in binary state name
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct= ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder="passthrough")
X=np.array(ct.fit_transform(X))
# print(X)
#training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#training  the multi linear regression model
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(X_train,y_train)

#test the multi linear regression model
y_pred=regressor.predict(X_test)
np.set_printoptions(precision=2)#precision use for .2 decimal show only
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))#axis=1