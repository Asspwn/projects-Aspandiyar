# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 01:09:18 2019

@author: acer
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:51:45 2019

@author: acer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import copy

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import VotingRegressor

# =============================================================================
# Import data
# =============================================================================

bm_df = pd.read_csv("BigMart.csv")

#%%

# =============================================================================
# Data Exploration
# =============================================================================

# First look at data
print(bm_df.head())

# Checking for NA's
print(bm_df.info())

# Outlet_Size vs Outlet_Type
outlet_size_crosstab = pd.crosstab(bm_df['Outlet_Size'], bm_df['Outlet_Type'])
print(outlet_size_crosstab)

# Looking at numerical data for any oddities
print(bm_df.describe())

# Univariate distribution of our target variable
sns.distplot(bm_df['Item_Outlet_Sales'])

# Pair plot of all numerical variables
sns.pairplot(bm_df[['Item_Weight',
                    'Item_Visibility',
                    'Item_MRP',
                    'Item_Outlet_Sales']])

# Confirming above Pairplot with a correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(bm_df.corr(),annot=True)

# Investigating Item_Visibility vs Item_Outlet_Sales
plt.figure(figsize=(12,7))
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Visibility and Item_Outlet_Sales Analysis")
plt.plot(bm_df.Item_Visibility, bm_df["Item_Outlet_Sales"],'.', alpha = 0.3)

# Check correlation between Visibility and Outlet_Size and Outlet_Type
# (diese beiden Zeilen einzeln ausführen)
sns.boxplot(x = 'Outlet_Size', y = 'Item_Visibility', data = bm_df)
sns.boxplot(x = 'Outlet_Type', y = 'Item_Visibility', data = bm_df)

# Distribution of Item_Fat_Content
sns.countplot(bm_df.Item_Fat_Content)

# Distribution of Item Type
sns.countplot(bm_df.Item_Type)
plt.xticks(rotation = 90)

# Distribution of Outlet_Identifier
sns.countplot(bm_df.Outlet_Identifier)
plt.xticks(rotation = 90)


ax = sns.catplot(x="Outlet_Identifier", 
                 y = "Item_Outlet_Sales", 
                 data=bm_df,
                 height=5,
                 aspect=2,
                 kind="bar")

#%%

# =============================================================================
# Data Cleaning and Feature Engineering
# =============================================================================

print('Starting Cleaning')

# Replacing Item_Weight NANs with Avg of Item_weight for resp. Item_Identifier
bm_df['Item_Weight'] = bm_df.groupby('Item_Identifier')\
                            .transform(lambda x: x.fillna(x.mean()))

# Right now, 4 Item_Weights are not assigned yet; therefore, we use the .mean(Item_Type)
bm_df['Item_Weight'] = bm_df.groupby('Item_Type')\
                            .transform(lambda x: x.fillna(x.mean()))

# Normalizing Item_Visibility by Outlet_Type
outlet_type_visibility_max = bm_df.pivot_table(values='Item_Visibility',
                                     columns='Outlet_Type',
                                     aggfunc=lambda x: x.max())

def normalize_visibility(cols):
    visibility = cols[0]
    outlet_type = cols[1]
    
    return visibility/outlet_type_visibility_max.loc['Item_Visibility']\
                     [outlet_type_visibility_max.columns == outlet_type][0]

bm_df['Item_Visibility'] = bm_df[['Item_Visibility', 'Outlet_Type']]\
                                 .apply(normalize_visibility, axis = 1)


# Replacing Item_Visibility = 0 with average visibility for that item
visibility_item_avg = bm_df.pivot_table(values='Item_Visibility',
                                       index='Item_Identifier')

def impute_visibility_mean(cols):
    visibility = cols[0]
    item = cols[1]
    if visibility == 0:
        return visibility_item_avg['Item_Visibility']\
                                  [visibility_item_avg.index == item]
    else:
        return visibility
    
bm_df['Item_Visibility'] = \
bm_df[['Item_Visibility','Item_Identifier']]\
.apply(impute_visibility_mean,axis=1).astype(float)

sns.distplot(bm_df['Item_Visibility'])

# Box-Cox transforming (aka log transforming) Item_Visibility
bm_df['Item_Visibility'], item_vis_lambda = boxcox(bm_df['Item_Visibility'])

# Replacing Item_Identifier with column of its first two letters
bm_df['Item_Identifier'] = bm_df['Item_Identifier'].apply(lambda x: x[0:2])

# Rename them to clearer categories:
bm_df['Item_Identifier'] = \
bm_df['Item_Identifier'].map({'FD': 'Food',
                              'NC': 'Non-Consumable',
                              'DR': 'Drinks'})

print(bm_df['Item_Identifier'].value_counts())

# Replacing Outlet Size NANs with Mode of each row's Outlet_Type
outlet_size_mode = bm_df.pivot_table(values='Outlet_Size',
                                     columns='Outlet_Type',
                                     aggfunc=lambda x: x.mode())

def impute_size_mode(cols):
    Size = cols[0]
    Type = cols[1]
    if pd.isnull(Size):
        return outlet_size_mode.loc['Outlet_Size']\
                                   [outlet_size_mode.columns == Type][0]
    else:
        return Size

bm_df['Outlet_Size'] = bm_df[['Outlet_Size', 'Outlet_Type']]\
                       .apply(impute_size_mode, axis=1)

# Converting Outlet_Establishment_Year to age of store
bm_df['Outlet_Age'] = 2013 - bm_df['Outlet_Establishment_Year']
bm_df = bm_df.drop(columns=['Outlet_Establishment_Year'])

# Homogenizing Item_Fat_Content
bm_df['Item_Fat_Content'] = bm_df['Item_Fat_Content'].replace({
                                                               'LF': 'Low Fat',
                                                               'reg': 'Regular',
                                                               'low fat': 'Low Fat',
                                                               })

    
# Replacing Item_Fat_Content of nonfood with 'Non-Consumable'
bm_df.loc[bm_df['Item_Identifier']=="Non-Consumable",'Item_Fat_Content'] = "Non-Consumable"    

print(bm_df['Item_Fat_Content'].value_counts())

# Finally, dropping NAs (of which there are only 4)
bm_df = bm_df.dropna()

# One-hot encoding all categorical variables except Item_Type
bm_df = pd.get_dummies(bm_df, columns=['Item_Identifier',
                                       'Item_Fat_Content',
                                       'Outlet_Identifier',
                                       'Outlet_Location_Type',
                                       'Outlet_Type',
                                       'Outlet_Size'])
 
# Dropping one of each one-hot vector, as well as Item_Type
bm_df = bm_df.drop(['Item_Identifier_Food',
                    'Item_Fat_Content_Non-Consumable',
                    'Outlet_Identifier_OUT010',
                    'Outlet_Location_Type_Tier 1',
                    'Outlet_Type_Grocery Store',
                    'Outlet_Size_Small',
                    'Item_Type'], axis=1)
    
# Split into X and Y
bm_X = bm_df.drop('Item_Outlet_Sales', axis = 1)
bm_Y = bm_df['Item_Outlet_Sales']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(bm_X,
                                                    bm_Y,
                                                    test_size = 0.2,
                                                    random_state = 0)

# Gathering means before we transform the target variable and normalize
Y_train_mean = Y_train.mean()
Y_train_meandev = sum((Y_train - Y_train_mean) ** 2)
Y_test_mean = Y_test.mean()
Y_test_meandev = sum((Y_test - Y_train_mean) ** 2)

# Box-Cox transforming Y_train (note: we get the lambda value here, which is
# the parameter for B-C transforms. We use this lambda to do the inverse transform
# on our test set predictions later. We can't transform the test set's Y-values but we can 
# do the inverse of the transform to our predicted values, which is the same thing)
Y_train_pretransform = copy.copy(Y_train)
Y_train, boxcox_lambda = boxcox(Y_train)

sns.distplot(Y_train)

# Normalizing Data
nscaler = MinMaxScaler()
X_train = nscaler.fit_transform(X_train)
X_test = nscaler.transform(X_test)

#%%

# =============================================================================
# =============================================================================
# Analysis
# =============================================================================
# =============================================================================

print('Starting Analysis')

def grid_search(clf_, grid_values_, X_train_, Y_train_, cv_ = 5):

    grid_result = GridSearchCV(estimator = clf_, 
                               param_grid = grid_values_,
                               cv = cv_,
                               n_jobs = -1,
                               verbose = True)
    
    grid_result.fit(X_train_, Y_train_)
    
    print("Best: %f using %s" % (grid_result.best_score_, 
                                 grid_result.best_params_))
    
    return grid_result

def evaluate(model, X_train_, X_test_, Y_train_, Y_test_):

    Y_train_pred = inv_boxcox(model.predict(X_train_), boxcox_lambda)#*X_train_MRPs
    Y_train_dev = sum((Y_train_pretransform  - Y_train_pred) ** 2)
    r2 = 1 - Y_train_dev / Y_train_meandev

    Y_test_pred = inv_boxcox(model.predict(X_test_), boxcox_lambda)#*X_test_MRPs
    Y_test_dev = sum((Y_test_ - Y_test_pred) ** 2)
    pseudor2 = 1 - Y_test_dev / Y_test_meandev

    return r2, pseudor2

# =============================================================================
# Dummy Classifier
# =============================================================================  

clf = DummyRegressor()
clf.fit(X_train, Y_train)
r2, pseudor2 = evaluate(clf, X_train, X_test, Y_train, Y_test)
print('Dummy Regression: ', r2, pseudor2)

# =============================================================================
# Linear Regression
# =============================================================================

lr = LinearRegression()
lr.fit(X_train, Y_train)
r2, pseudor2 = evaluate(lr, X_train, X_test, Y_train, Y_test)
print('Linear Regression: ', r2, pseudor2)

# =============================================================================
# Ridge Regression
# =============================================================================

print('Starting Ridge')

clf = Ridge()
grid_values = {'alpha': np.linspace(0, 2, 20)}
ridgereg = grid_search(clf, grid_values, X_train, Y_train)

# Best: 0.304482 using {'alpha': 2.0}
# Done 100 out of 100 | elapsed:    2.4s finished

r2, pseudor2 = evaluate(ridgereg, X_train, X_test, Y_train, Y_test)
print('Ridge: ', r2, pseudor2)

# =============================================================================
# Support Vector Regression
# =============================================================================

print('Starting SVR')

clf = SVR()
grid_values = [
  {'C': [1, 10, 100, 1000],
   'kernel': ['linear']},
  {'C': [1, 10, 100, 1000],
   'epsilon': [0.1, 0.01, 0.001, 0.0001],
   'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001],
   'kernel': ['rbf']},
 ]

SVReg = grid_search(clf, grid_values, X_train, Y_train)

r2, pseudor2 = evaluate(SVReg, X_train, X_test, Y_train, Y_test)
print('SVR (best of linear and rbf): ', r2, pseudor2)

# =============================================================================
# Neural Network
# =============================================================================

print('Starting Neural Network')

clf = MLPRegressor(solver='lbfgs', random_state=0)
grid_values = {'hidden_layer_sizes': np.arange(15,25,1), 
               'alpha': [0.1, 0.01, 0.001, 0.0001]}
NNetReg = grid_search(clf, grid_values, X_train, Y_train)

# Best: 0.573892 using {'alpha': 0.0010000000000000002, 'hidden_layer_sizes': (15,)}
# Done 165 out of 165 | elapsed:   35.4s finished

r2, pseudor2 = evaluate(NNetReg, X_train, X_test, Y_train, Y_test)
print('Neural Network: ', r2, pseudor2)

# =============================================================================
# Random Forest
# =============================================================================

clf = RandomForestRegressor(random_state=0)
grid_values = {'max_depth': [k for k in range (7,9)], 
               'n_estimators': [k*25 for k in range(19,21)]}
RForreg = grid_search(clf, grid_values, X_train, Y_train, 2)

# Best: 0.520450 using {'max_depth': 8, 'n_estimators': 500}
# Done   8 out of   8 | elapsed:   40.7s finished

r2, pseudor2 = evaluate(RForreg, X_train, X_test, Y_train, Y_test)
print('Random Forest: ', r2, pseudor2)

# =============================================================================
# Gradient Boosting
# =============================================================================

clf = GradientBoostingRegressor(random_state=0)
grid_values = {'max_depth': [k for k in range (4,6)], 
               'learning_rate': np.linspace(0.01, 0.1, 10)}
GBoostreg = grid_search(clf, grid_values, X_train, Y_train)

# Best: 0.548547 using {'learning_rate': 0.2, 'max_depth': 4}
# Done  30 out of  30 | elapsed:   21.1s finished

r2, pseudor2 = evaluate(GBoostreg, X_train, X_test, Y_train, Y_test)
print('Gradient Boosting: ', r2, pseudor2)

# =============================================================================
# XGBoost
# =============================================================================

grid_values = {'n_estimators': [100, 1000, 5000],
               
        
        }

XGBReg = XGBRegressor(n_estimators = 1000, learning_rate = 0.05)
XGBReg.fit(X_train, Y_train)
r2, pseudor2 = evaluate(XGBReg, X_train, X_test, Y_train, Y_test)
print('XGB Regression: ', r2, pseudor2)

#%%

# =============================================================================
# Fitting all optimized models (just run this to save time)
# =============================================================================

lr = LinearRegression()
lr.fit(X_train, Y_train)
r2, pseudor2 = evaluate(lr, X_train, X_test, Y_train, Y_test)
print('Linear Regression: ', r2, pseudor2)

# alpha in [0, 0.1, ..., 2] (optimiert)
ridgereg = Ridge(alpha = 1.368)
ridgereg.fit(X_train, Y_train)
r2, pseudor2 = evaluate(ridgereg, X_train, X_test, Y_train, Y_test)
print('Ridge: ', r2, pseudor2)

# C in [1,10,100,1000] (nicht optimiert)
# epsilon in [0.1, 0.01, 0.001, 0.0001] (nicht optimiert)
# gamma in [0.1, 0.01, 0.001, 0.0001, 0.00001] (nicht optimiert)
SVReg = SVR(C = 1000, epsilon = 0.1, gamma = 0.1, kernel = 'rbf')
SVReg.fit(X_train, Y_train)
r2, pseudor2 = evaluate(SVReg, X_train, X_test, Y_train, Y_test)
print('SVR (best of linear and rbf): ', r2, pseudor2)

# hidden_layer_sizes in [15, 16, ..., 25] (optimiert)
# alpha in [0.1, 0.01, 0.001, 0.0001] (in order of magnitude optimiert aber nicht exakt)
NNetReg = MLPRegressor(solver='lbfgs',
                       random_state=0,
                       alpha = 0.01,
                       hidden_layer_sizes = 17)
NNetReg.fit(X_train, Y_train)
r2, pseudor2 = evaluate(NNetReg, X_train, X_test, Y_train, Y_test)
print('Neural Network: ', r2, pseudor2)

# max_depth in [7,8] (nicht optimiert)
# n_estimators in [475, ...] (nicht optimiert)
RForreg = RandomForestRegressor(random_state = 0,
                                max_depth = 7,
                                n_estimators = 475)
RForreg.fit(X_train, Y_train)
r2, pseudor2 = evaluate(RForreg, X_train, X_test, Y_train, Y_test)
print('Random Forest: ', r2, pseudor2)

# max_depth in [4, 5] (nicht optimiert)
# learning_rate in [0.01, 0.02, ..., 0.1] (gut genug glaube ich)
GBoostreg = GradientBoostingRegressor(random_state = 0,
                                      learning_rate = 0.04,
                                      max_depth = 4)
GBoostreg.fit(X_train, Y_train)
r2, pseudor2 = evaluate(GBoostreg, X_train, X_test, Y_train, Y_test)
print('Gradient Boosting: ', r2, pseudor2)



#%% 

# =============================================================================
# Ensembling
# =============================================================================

ensemble = VotingRegressor([('linear', lr), 
                            ('ridge', ridgereg),
                            ('SVR', SVReg),
                            ('Neural Net', NNetReg),
                            ('R. Forest', RForreg),
                            ('G. Boost', GBoostreg)])

ensemble.fit(X_train, Y_train)
r2, pseudor2 = evaluate(ensemble, X_train, X_test, Y_train, Y_test)
print('Ensemble: ', r2, pseudor2)

#%%

# =============================================================================
# Feature Importances
# =============================================================================

coef1 = pd.Series(lr.coef_, bm_X.columns).sort_values()
coef1.plot(kind='bar', title='Linear Regression Coefficients')


coef2 = pd.Series(RForreg.feature_importances_, bm_X.columns).sort_values()
coef2.plot(kind='bar', title='Random Forest Feature Importances')
