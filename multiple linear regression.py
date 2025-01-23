import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\RAMYA\Downloads\Investment.csv")

x = dataset.iloc[:,:-1]

y = dataset.iloc[:,4]

x = pd.get_dummies(x, dtype= int)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

m = regressor.coef_ #slope (#6 independent variables,hence 6 slopes)
print(m)

c = regressor.intercept_ #constant
print(c)

X = np.append(arr = np.ones((50,1)).astype(int), values = x , axis = 1)

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]] #index 6 is removed
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit() #ordinary least squares
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5]] #index 4 is removed as p value is high(reject the null hypothesis) #recursive feature elimination
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit() #ordinary least squares
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3]] #index 5 is removed as p value is high(reject the null hypothesis)
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit() #ordinary least squares
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,3]] #index 2 is removed as p value is high(reject the null hypothesis)
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit() #ordinary least squares
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1]] #index 3 is removed as p value is high(reject the null hypothesis)
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit() #ordinary least squares
regressor_OLS.summary()

bias = regressor.score(x_train,y_train)
print(bias)

variance = regressor.score(x_test,y_test)
variance















