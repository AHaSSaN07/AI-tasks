import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

data = pd.read_csv("aps_failure_training_set_processed_8bit.csv",index_col = None)


"""for j in range(1, len(data.columns)):
    data = data[np.abs(data[data.columns[j]]-data[data.columns[j]].mean())<=(3*data[data.columns[j]].std())]#SOF Rocks! 
"""
"""def transform(c):
    le = LabelEncoder()
    if(c.dtype == "object"):
        le.fit(c)
        c = le.transform(c)
    return c
"""
    

X = data.iloc[:, 1:]
Y = data.iloc[:, 0]

Y[Y>0] = 1
Y[Y<0] = 0

#xtrain,xtest,ytrain,ytest =train_test_split(X,Y,test_size =.33)


nparr = np.zeros(Y.shape[0])
skf = StratifiedKFold(n_splits =5)
#
for train ,test in skf.split(X,Y):
    xtrain = X.iloc[train,:]
    xtest = X.iloc[test,:]
    ytrain = Y[train]
    #ytest = Y[test]
    gradientboosting = GradientBoostingClassifier()
    gradientboosting.fit(xtrain, ytrain)
    nparr[test] = gradientboosting.predict(xtest)
   
    
print(accuracy_score(Y,nparr))
            
    

    
