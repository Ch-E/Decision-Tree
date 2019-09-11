# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 19:41:10 2019

@author: Charl
"""
#%% Import data
import pandas as pd
import numpy as np
from time import time

#%%
#**********************************Train & Test dataset**************************************
train = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/train_V2.csv')
train.head()

test = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/test_V2.csv')
test.head()

# check for null values in dataset - none found
train.isnull().sum().sum()
test.isnull().sum().sum()

train.winPlacePerc.fillna(1,inplace=True)
train.loc[train['winPlacePerc'].isnull()]

train["distance"] = train["rideDistance"]+train["walkDistance"]+train["swimDistance"]
train["skill"] = train["headshotKills"]+train["roadKills"]
train.drop(['rideDistance','walkDistance','swimDistance','headshotKills','roadKills'],inplace=True,axis=1)

print(train.shape)
train.head()

test["distance"] = test["rideDistance"]+test["walkDistance"]+test["swimDistance"]
test["skill"] = test["headshotKills"]+test["roadKills"]
test.drop(['rideDistance','walkDistance','swimDistance','headshotKills','roadKills'],inplace=True,axis=1)
print(test.shape)
test.head()

# dropped features based on feature importances
predictors = ["kills",
              "maxPlace",
              "numGroups",
              "distance",
              "boosts",
              "killStreaks",
              "weaponsAcquired",
              "DBNOs",
              "killPlace",
              ]

X = train[predictors]
X.head()

y = train['winPlacePerc']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#**********************************Train & Test dataset**************************************

#%% hyperparameter tuning
from sklearn.model_selection import GridSearchCV

def log(x):
    # can be used to write to log file
    print(x)

# Utility function to report best scores (from scikit-learn.org)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            log("Model with rank: {0}".format(i))
            log("Mean validation score: {0:.5f} (std: {1:.5f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            log("Parameters: {0}".format(results['params'][candidate]))
            log("")


#Function to determine the best fit (from scikit-learn.org)

def best_fit(clf, X_train, y_train):
    
    param_grid = {
                    'max_features':['sqrt','log2',None],
                    'max_depth': np.arange(1, 15),
                    'min_samples_split': range(2,16,2),
                    'min_samples_leaf': range(2,20,2),
                    'max_leaf_nodes': [5,10,None],
                 }

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10, n_jobs=8)

    import time as ttt
    now = time()
    log(ttt.ctime())
    
    grid_search.fit(X_train, y_train)
    
    report(grid_search.cv_results_, n_top=10)
    
    log(100*"-")
    log(ttt.ctime())
    log("Search (3-fold cross validation) took %.5f seconds for %d candidate parameter settings." 
        % (time() - now, len(grid_search.cv_results_['params'])))
    log('')
    log("The best parameters are %s with a score of %0.5f"
        % (grid_search.best_params_, grid_search.best_score_))
    
    return grid_search
#%%
from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor(random_state=101, max_features=None, 
                              max_depth=14, min_samples_split=2, min_samples_leaf=18, 
                              max_leaf_nodes=None)

# hyperparameter tuning method call (~3 hours run time)
#best_tree = best_fit(dtree, X_train, y_train)

dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

#%% feature importances

# list the feature importance value
importances = dtree.feature_importances_
print(importances)

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
indices

# Rearrange feature names so they match the sorted feature importances
feature_names = [X_train.columns[i] for i in indices]
feature_names

X_columns = X_train.columns
X_columns

# Print the feature ranking
print("Feature ranking (index):")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
# Print the feature ranking
print("Feature ranking (column name):")

for f in range(indices.shape[0]):
    print("%2d) %-*s %0.9f" % (f + 1, 10,
                            X_columns[indices[f]],
                            importances[indices[f]]))

#%% Metrics

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error:")
print(mean_absolute_error(y_test, predictions))


#%%
#**********************************Submission**************************************
test_id = test["Id"]
submit = pd.DataFrame({'Id': test_id, "winPlacePerc": y_test} , columns=['Id', 'winPlacePerc'])
print(submit.head())

submit.to_csv("submission.csv", index = False)
#**********************************Submission**************************************
#%%

# default parameters for DecisionTreeRegressor

#criterion=’mse’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
#min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, 
#min_impurity_decrease=0.0, min_impurity_split=None, presort=False


#The best parameters are {'max_leaf_nodes': None, 'min_samples_leaf': 18, 
#                         'min_samples_split': 2, 'max_features': None,
#                         'max_depth': 14} with a score of 0.89639




#from sklearn.model_selection import cross_validate
#scoring = ['accuracy', 'precision_macro', 'f1_macro']

#scores = cross_validate(dtree, X_train, y_train, scoring=scoring, return_train_score=False)






