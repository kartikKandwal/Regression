import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d1=pd.read_csv('Position_Salaries.csv')
X=d1.iloc[:,1:-1].values
y=d1.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,y)

regressor.predict([[6.5]])

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color = 'red')
plt.plot(X_grid,regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()