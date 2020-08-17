'''
1.import the library
2.importing data set
3.spliting the data set into the trainset and test set
4.Training the simple linear model
5.predicting
6.visualising
7.visualising
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
#import data csv

d1=pd.read_csv('Salary_Data.csv')
X=d1.iloc[:,:-1].values
y=d1.iloc[:,-1].values
#spliting the data set into the trainset and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#train the simple linear model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#prediction the test result
y_pred=regressor.predict(X_test)

#visulising
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience')
plt.xlabel('year of experience')
plt.ylabel('Salary')
plt.show()

#visualising test set result
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience')
plt.xlabel('year of experience')
plt.ylabel('Salary')
plt.show()
