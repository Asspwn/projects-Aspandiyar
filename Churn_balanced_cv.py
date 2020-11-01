# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import pandas as pd
from sklearn.metrics import accuracy_score


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


os.chdir("D:/Dropbox/Lehre/Digital Analytics/Excercises/Classification")
data_ori = pd.read_csv('churn_balanced.csv')
print(data_ori.shape)
# types
print(data_ori.dtypes)
# feature names
print(list(data_ori))
# head
print(data_ori.head(6))
# descriptions, change precision to 2 places
print(data_ori.describe())

X_ori = data_ori.drop('CHURN', axis = 1)
Y = data_ori['CHURN']

# standardize data = (data_ori-data_ori.mean())/data_ori.std()
X = (X_ori-X_ori.min())/(X_ori.max()-X_ori.min())
print(X.head(6))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)  #random_state=0
print(Y_train.value_counts())
print(Y_test.value_counts())

#create report dataframe
report = pd.DataFrame(columns=['Model','Mean Acc. Training','Standard Deviation','Acc. Test'])


#######################
# Logistic Regression #
#######################

from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression()
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(lrmodel, X_train, Y_train, scoring='accuracy', cv = 10)
print("Accuracies = ", accuracies)
print("Mean = ", accuracies.mean())
print("SD = ", accuracies.std())
lrmodel.fit(X_train, Y_train)
Y_test_pred = lrmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['Logistic Regression', accuracies.mean(), accuracies.std(), accte]
print(report.loc[len(report)-1])


###############
# Naive Bayes #
###############

from sklearn.naive_bayes import GaussianNB
nbmodel = GaussianNB()
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(nbmodel, X_train, Y_train, scoring='accuracy', cv = 10)
print("Accuracies = ", accuracies)
nbmodel.fit(X_train, Y_train)
Y_test_pred = nbmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['Naive Bayes', accuracies.mean(), accuracies.std(), accte]
print(report.loc[len(report)-1])


#################
# Random Forest #
#################

from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(random_state=0)
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'max_depth': [ 4.,  5.,  6.,  7.,  8.],
    'n_estimators': [ 10,  50,  100, 150, 200]
}
CV_rfmodel = GridSearchCV(estimator=rfmodel, param_grid=param_grid, cv=10)
CV_rfmodel.fit(X_train, Y_train)
print(CV_rfmodel.best_params_)
#use the best parameters
rfmodel = rfmodel.set_params(**CV_rfmodel.best_params_)
rfmodel.fit(X_train, Y_train)
Y_test_pred = rfmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['Random Forest (grid)', 
                          CV_rfmodel.cv_results_['mean_test_score'][CV_rfmodel.best_index_], 
                          CV_rfmodel.cv_results_['std_test_score'][CV_rfmodel.best_index_], accte]
print(report.loc[len(report)-1])


################################
# Gradient Boosting Classifier #
################################

from sklearn.ensemble import GradientBoostingClassifier
gbmodel = GradientBoostingClassifier(random_state=0)
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'max_depth': [ 3., 4., 5.],
    'subsample': [0.7, 0.8, 0.9],
    'n_estimators': [50, 100,150],
    'learning_rate': [0.1, 0.2, 0.3]
}
CV_gbmodel = GridSearchCV(estimator=gbmodel, param_grid=param_grid, cv=10)
CV_gbmodel.fit(X_train, Y_train)
print(CV_gbmodel.best_params_)
#use the best parameters
gbmodel = gbmodel.set_params(**CV_gbmodel.best_params_)
gbmodel.fit(X_train, Y_train)
Y_test_pred = gbmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['Gradient Boosting (grid)', 
                          CV_gbmodel.cv_results_['mean_test_score'][CV_gbmodel.best_index_], 
                          CV_gbmodel.cv_results_['std_test_score'][CV_gbmodel.best_index_], accte]
print(report.loc[len(report)-1])





################
# Final Report #
################

print(report)

