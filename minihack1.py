# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 19:08:49 2016

@author: abzooba
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error
from math import  sqrt
import random
import xgboost as xgb

random.seed(786)

rcParams['figure.figsize'] = 15, 6


#data = pd.read_csv('Train_JPXjxg6.csv')
#print data.head()
#print '\n Data Types:'
#print data.dtypes

dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%Y %H:%M')
data = pd.read_csv('Train_JPXjxg6.csv', parse_dates='Datetime', index_col='Datetime',date_parser=dateparse)
test = pd.read_csv('Test_mvj827l.csv', parse_dates='Datetime', index_col='Datetime',date_parser=dateparse)


# add dummmy
test['Count'] = -1

# merge the training and test sets
n = data.shape[0]
combined = pd.concat( [ data, test ] )

combined['trend'] = range(1, len(combined) + 1)
combined['trend'] = combined['trend']**2
#combined['dayofyear'] = [int(x.strftime('%j')) for x in combined.index ] 
combined['year'] = combined.index.year
combined['month'] = combined.index.month
combined['dayofweek'] = combined.index.dayofweek
combined['hour'] = combined.index.hour
#combined['weekofyear'] = combined.index.weekofyear
#combined['weekend'] = 0
#combined.loc[combined['dayofweek'] >= 5, 'weekend' ] = 1

# separate into training and test sets
training = combined.head(n)
testing = combined.drop(training.index)

# drop metrics from the testing set
testing.drop(['Count'], axis=1, inplace=True)

features = list(testing.columns.values)
target = 'Count'
#features.remove(target)
X = training[features].values
x_test = testing.values
y = training[target]

dx = xgb.DMatrix(X, label=y)        
dtest = xgb.DMatrix(x_test ) 

kf = KFold(n, 4, False)

# setup parameters for xgboost
param = {}
param['objective'] = 'reg:linear'
param['eval_metric'] = 'rmse'
# scale weight of positive examples
param['eta'] = 0.3
param['max_depth'] = 4
param['silent'] = 1
param['subsample'] = 0.5
param['colsample_bytree'] = 0.6  

fold = 1
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    rf = RandomForestRegressor(n_estimators=300, criterion='mse', max_depth=5, max_features='auto', oob_score=False, n_jobs=1, verbose=0, warm_start=False)
    rf_fit = rf.fit(X_train, y_train)
    
    y_hat_rf = rf_fit.predict(X_test)
    print '*'*5 + str(fold) + '*'*5 
    print 'RF = ' + str( sqrt(mean_squared_error(y_test, y_hat_rf)) )
    
    gbr = GradientBoostingRegressor()
    gbr_fit = gbr.fit(X_train, y_train)
    y_hat_gbr = gbr_fit.predict(X_test)
    print 'GBR = ' + str( sqrt(mean_squared_error(y_test, y_hat_gbr)) )
    
    dtrain = xgb.DMatrix(X_train, label=y_train )
    dval = xgb.DMatrix(X_test, label=y_test ) # 
    
    watchlist = [ (dtrain,'train'), (dval, 'test') ]
    num_round = 500
    clf = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=30, verbose_eval=False) # 
    print 'XGB = ' + str( clf.best_score )
    
    fold += 1

#rf_whole = rf.fit(X, y)
#predictions = rf_whole.predict(x_test)
#test[target] = predictions

gbr_whole = gbr.fit(X, y)
predictions = gbr_whole.predict(x_test)
test[target] = predictions

#xgb_clf = xgb.train(param, dx, 80)
#predictions = xgb_clf.predict(dtest)
#test[target] = predictions

#test['Datetime'] = test.index.apply(lambda x: x.strftime('%d-%m-%Y %H:%M'))
test['Datetime'] = test.index.strftime('%d-%m-%Y %H:%M')
test.to_csv('submission4.csv', index=False, header = True)