from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

## Uploading our data and checking.
temp_datas = pd.read_csv("data.csv")

print(temp_datas.head())
print(temp_datas.describe())
print(temp_datas.isnull().sum())




## Taking the columns which will work for us and eliminating overstatement values.
t2data = temp_datas.iloc[:,1:13]
data1 = t2data[t2data['price'] < 1000000 ]
data = data1[data1['price'] > 0 ]
## Seperating the columns as our x and y.

result = data.iloc[:,0:1]
reasons = data.iloc[:,1:]




## Seperating the data for train and  test.

x_train, x_test,y_train,y_test = train_test_split(reasons,result,test_size=0.3, random_state=0)


regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)


## Backward elimination for better result.
X = np.append(arr = np.ones((4207,1)).astype(int), values=reasons, axis=1 )
X_l = reasons.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]].values
r_ols = sm.OLS(endog = result, exog =X_l)
r = r_ols.fit()
print(r.summary())

X = np.append(arr = np.ones((4207,1)).astype(int), values=reasons, axis=1 )
X_l = reasons.iloc[:,[0,1,2,3,4,5,6,7,8,9]].values
r_ols = sm.OLS(endog = result, exog =X_l)
r = r_ols.fit()
print(r.summary())

X = np.append(arr = np.ones((4207,1)).astype(int), values=reasons, axis=1 )
X_l = reasons.iloc[:,[0,2,3,4,5,6,7,8,9]].values
r_ols = sm.OLS(endog = result, exog =X_l)
r = r_ols.fit()
print(r.summary())


## After Backward Elimination, dropping some columns for better results.
x_train = pd.concat([x_train.iloc[:,0:1],x_train.iloc[:,2:10]],axis = 1)
x_test =  pd.concat([x_test.iloc[:,0:1],x_test.iloc[:,2:10]],axis = 1)

regressor.fit(x_train,y_train)
y_pred2 = regressor.predict(x_test)

## Visualization


plt.scatter(y_pred,y_test, color="red")
plt.xlabel("x-label")
plt.ylabel("y-label")


plt.scatter(y_pred2,y_test, color="red")
plt.xlabel("x-label2")
plt.ylabel("y-label2")

