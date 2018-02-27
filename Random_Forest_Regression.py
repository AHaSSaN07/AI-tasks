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

#spliting into input and output

X =data.iloc[:,:8]
Y =data.iloc[:,8]

#splitting into train and test sets 

xtrain,xtest,ytrain,ytest =train_test_split(X,Y,test_size =.5,random_state =0)


##applying linear regression (out>>2.16609908149)
#linreg=LinearRegression()
#linreg.fit(xtest,ytest)
#result =linreg.predict(xtest)



#applying Decision Trees
#from sklearn.tree import tree
#desction_tree =tree.DecisionTreeClassifier()
#desction_tree.fit(xtest,ytest)
#result =desction_tree.predict(xtest)


#applying random Forest
from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor()
Regressor.fit(xtest,ytest)
result =Regressor.predict(xtest)



#error
print(np.sqrt(mean_squared_error(ytest,result)))

