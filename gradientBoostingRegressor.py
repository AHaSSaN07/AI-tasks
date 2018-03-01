import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#reading data

data = pd.read_csv("abalone.csv",index_col = None)
data.columns = ['sex','length','diameter','height','whole weight','shucked weight','viscera weight','shell weight','rings']

#converting  Categorical data(sex only)

le = LabelEncoder()
le.fit(data.sex)
data.sex = le.transform(data.sex)
#removing outliers 

for j in range(1, len(data.columns)):
    data = data[np.abs(data[data.columns[j]]-data[data.columns[j]].mean())<=(3*data[data.columns[j]].std())]#SOF Rocks! 

#spliting into input and output

X =data.iloc[:,:8]
Y =data.iloc[:,8]

#splitting into train and test sets 

xtrain,xtest,ytrain,ytest =train_test_split(X,Y,test_size =.33,random_state =42)


##applying linear regression (out->2.16609908149)
linreg=LinearRegression()
linreg.fit(xtrain,ytrain)
result3 =linreg.predict(xtest)



#applying Decision Trees
from sklearn.tree import DecisionTreeRegressor
desction_tree =DecisionTreeRegressor()
desction_tree.fit(xtrain,ytrain)
result2 =desction_tree.predict(xtest)


#applying random Forest

from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor(n_estimators =500,criterion ='mse')
Regressor.fit(xtrain,ytrain)
result =Regressor.predict(xtest)

##applying XGB
#from xgboost.sklearn import XGBRegressor
##
#XGBoostRegressor = XGBRegressor()
#XGBoostRegressor.fit(xtrain,ytrain)
#XGBoostRegressorResult = XGBoostRegressor.predict(xtest)
##

############################################################
from sklearn.ensemble import GradientBoostingRegressor

gradientBoostingRegressor = GradientBoostingRegressor()
gradientBoostingRegressor.fit(xtrain,ytrain)
gradientBoostingRegressorResult = gradientBoostingRegressor.predict(xtest)




#error
print("Random forest regression error:",np.sqrt(mean_squared_error(ytest,result)))
print("linear regression error:",np.sqrt(mean_squared_error(ytest,result3)))
print("Decision Trees error:",np.sqrt(mean_squared_error(ytest,result2)))
print ("gradientBoostingRegressor:",np.sqrt(mean_squared_error(ytest,gradientBoostingRegressorResult)))



#output
#Random forest regression error: 2.04120748873
#linear regression error: 2.06632655797
#Decision Trees error: 2.6786120086
#gradientBoostingRegressor error: 2.01407899865











