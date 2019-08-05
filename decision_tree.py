# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

#**********************************Train & Test dataset**************************************
train = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/train_V2.csv')
train.head()

test = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/test_V2.csv')
test.head()

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

predictors = [  "kills",
                "maxPlace",
                "numGroups",
                "distance",
                "boosts",
                "heals",
                "revives",
                "killStreaks",
                "weaponsAcquired",
                "winPoints",
                "skill",
                "assists",
                "damageDealt",
                "DBNOs",
                "killPlace",
                "killPoints",
                "vehicleDestroys",
                "longestKill"
               ]

X = train[predictors]
X.head()

y = train['winPlacePerc']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#**********************************Train & Test dataset**************************************

from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor()

dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)


#Metrics

#from sklearn.metrics import mean_absolute_error

#print(mean_absolute_error(y_test, predictions))

#from sklearn.metrics import classification_report, confusion_matrix

#print(confusion_matrix(y_test, predictions))
#print(classification_report(y_test, predictions))


#**********************************Submission**************************************
test_id = test["Id"]
submit = pd.DataFrame({'Id': test_id, "winPlacePerc": y_test} , columns=['Id', 'winPlacePerc'])
print(submit.head())

submit.to_csv("submission.csv", index = False)
#**********************************Submission**************************************









