# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 19:41:10 2019

@author: Charl
"""
#%% Import data

import pandas as pd
import numpy as np
from time import time

#%% Read dataset

train = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/train_V2.csv')

train.winPlacePerc.fillna(1,inplace=True)
train.loc[train['winPlacePerc'].isnull()]

# Create distance feature
train["distance"] = train["rideDistance"]+train["walkDistance"]+train["swimDistance"]
train.drop(['rideDistance','walkDistance','swimDistance'],inplace=True,axis=1)

# Create headshot_rate feature
train['headshot_rate'] = train['headshotKills'] / train['kills']
train['headshot_rate'] = train['headshot_rate'].fillna(0)

# Create playersJoined feature - used for normalisation
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')

#%% Data cleaning - removing outliers

# Row with NaN 'winPlacePerc' value - pointed out by averagemn (https://www.kaggle.com/donkeys)
train.drop(2744604, inplace=True)

# Players who got kills without moving
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['distance'] == 0))
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)

# Players who got more than 10 roadkills
train.drop(train[train['roadKills'] > 10].index, inplace=True)

# Players who got more than 30 kills
train[train['kills'] > 30].head(10)

# Players who made a minimum of 9 kills and have a headshot_rate of 100%
train[(train['headshot_rate'] == 1) & (train['kills'] > 8)].head(10)

# Players who made kills with a distance of more than 1 km
train.drop(train[train['longestKill'] >= 1000].index, inplace=True)

# Players who acquired more than 80 weapons
train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)

# Players how use more than 40 heals
train['heals'] = train['boosts']+train['heals']
train.drop(train[train['heals'] >= 40].index, inplace=True)

# Create normalised features
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)
train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['assistsNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['roadKillsNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['vehicleDestroysNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['killPointsNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['headshotKillsNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['revivesNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)

#%%

# Features that will be used for training
predictors = [
              "numGroups",
              "distance",
              "boosts",
              "killStreaks",
              "DBNOs",
              "killPlace",
              "killStreaks",
              "longestKill",
              "heals",
              "weaponsAcquired",
              "headshot_rate",
              "assistsNorm",
              "headshotKillsNorm",
              "damageDealtNorm",
              "killPointsNorm",
              "revivesNorm",
              "roadKillsNorm",
              "vehicleDestroysNorm",
              "killsNorm",
              "maxPlaceNorm",
              "matchDurationNorm",
              ]

X = train[predictors]
X.head()

y = train['winPlacePerc']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#%% Hyperparameter tuning

from sklearn.model_selection import GridSearchCV

def log(x):
    # can be used to write to a log file
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

# Function to determine the best fit (from scikit-learn.org)
def best_fit(clf, X_train, y_train):
    
    param_grid = {
                    'max_features':['sqrt','log2',None],
                    'max_depth': np.arange(1, 15),
                    'min_samples_split': range(2,16,2),
                    'min_samples_leaf': range(2,20,2),
                    'max_leaf_nodes': [5,10,None],
                 }

    # Run grid search
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

#%% Model

from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor(random_state=101, max_features=None, max_depth=14, min_samples_split=2,
                              min_samples_leaf=18, max_leaf_nodes=None)

# Hyperparameter tuning method call (~3 hours run time)
best_tree = best_fit(dtree, X_train, y_train)

# Output:
# The best parameters are {'max_leaf_nodes': None, 'min_samples_leaf': 18,
#                         'min_samples_split': 2, 'max_features': None,
#                         'max_depth': 14} with a score of 0.89639

dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

#%% Feature importances

# List the feature importance value
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

# Feature ranking (column name):
#  1) distance   0.695173563
#  2) killPlace  0.233122504
#  3) numGroups  0.017167211
#  4) boosts     0.010391879
#  5) assistsNorm 0.009764422
#  6) killsNorm  0.007610490
#  7) killStreaks 0.006952296
#  8) headshotKillsNorm 0.004577564
#  9) maxPlaceNorm 0.004277439
# 10) DBNOs      0.001806396
# 11) vehicleDestroysNorm 0.001694837
# 12) killStreaks 0.001486443
# 13) matchDurationNorm 0.001398479
# 14) roadKillsNorm 0.001211661
# 15) revivesNorm 0.001046521
# 16) heals      0.000673904
# 17) killPointsNorm 0.000578243
# 18) longestKill 0.000505060
# 19) weaponsAcquired 0.000403372
# 20) damageDealtNorm 0.000147940
# 21) headshot_rate 0.000009776

#%% Metrics

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

MAE = mean_absolute_error(y_test, predictions)
MSE = mean_squared_error(y_test, predictions)
R2 = r2_score(y_test, predictions)

print("Metrics:")
print("-------------------------------")
print("Mean Absolute Error: {}".format(MAE))
print("Mean Squared Error: {}".format(MSE))
print("R2 Score: {}".format(R2))

# Cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

cross_val_prediction = cross_val_predict(dtree, X_train, y_train, cv=5)

print("\n---------------------------------")
print("5-FOLD CROSS-VALIDATION")
print("---------------------------------")
print("Cross-validation score (R2): {}".format(cross_val_score(dtree, X_train, y_train, cv=5)))

#%% Submission

train_id = train["Id"]
submit = pd.DataFrame({'Id': train_id, "winPlacePerc": y_test} , columns=['Id', 'winPlacePerc'])
print("Submission head\n {}".format(submit.head()))

submit.to_csv("submission.csv", index = False)

#%%
