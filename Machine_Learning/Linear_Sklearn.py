import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

dataset=pd.read_csv('linear_regression_dataset.csv')

dependent_variable='TOTCHG'
independent_variable=dataset.columns.tolist()
independent_variable.remove(dependent_variable)

X=dataset[independent_variable].values

y=dataset[dependent_variable].values



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)

# Transform X_test using the trained imputer
X_test_imputed = imputer.transform(X_test)



regressor=LinearRegression()
regressor.fit(X_train,y_train)


y_pred=regressor.predict(X_test_imputed )



math.sqrt(mean_squared_error(y_test,y_pred))

print(r2_score(y_test,y_pred))


