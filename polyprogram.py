
'''
1.import the library
2.importing data set
3.feature scaling
4.Training the supporting vector regression model
5.predicting
6.visualising the Decision tree regression result
7.visualising the Decision tree regression resul(high resolution
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
d1=pd.read_csv('Position_Salaries.csv')
X=d1.iloc[:,1:-1].values
y=d1.iloc[:,-1].values
# print(X)
# print(y)
#training the linear regression model on the whole data set
from sklearn.linear_model import LinearRegression
lin_regr=LinearRegression()
lin_regr.fit(X,y)
#training the polynomial regression model on the whole data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)               #we have two time regression  degree 2,3,4-------n
X_poly=poly_reg.fit_transform(X)
lin_regr_2=LinearRegression()
lin_regr_2.fit(X_poly,y)
#visualing the linear
# plt.scatter(X,y,color='red')
# plt.plot(X,lin_regr.predict(X),color='blue')
# plt.title('Truth salary table with year of work knowledge')
# plt.xlabel('position')
# plt.ylabel('salary')
# plt.show()
# #visualing the polynomial regression
# plt.scatter(X,y,color='red')
# plt.plot(X,lin_regr_2.predict(poly_reg.fit_transform(X)),color='blue')
# plt.title('Truth salary table with year of work knowledge')
# plt.xlabel('position')
# plt.ylabel('salary')
# plt.show()
#prediction in linear and polynomial
k=lin_regr.predict([[6.5]])
#polynomial answer
z=lin_regr_2.predict(poly_reg.fit_transform([[6.5]]))
print(f"this is the salary given by linear{k}")
print(f"this is the salary given by polynomial{z}")