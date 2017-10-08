#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 20:10:15 2017

@author: becca
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:28:14 2017

@author: becca
"""
#from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report
#from sklearn.svm import SVC

import pandas as pd
import os
import numpy as np

path = '/home/becca/Insight Project/data files'
os.chdir(path)

file = '/home/becca/Insight Project/data files/cats_dogs_munged.csv'
cats_dogs_munged = pd.read_csv(file)
cats_dogs_munged  = cats_dogs_munged.drop(labels=['Unnamed: 0'], axis=1)

y = cats_dogs_munged.status
cats_dogs_munged = cats_dogs_munged.drop(labels=['status'], axis=1)
x = np.array(cats_dogs_munged)
n_classes = len(y.unique())
n_samples = len(x)

##encode labels
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
list(le.classes_)

encoded_labels = le.transform(y)
print encoded_labels
reversed_labels = le.inverse_transform(encoded_labels)
print reversed_labels

y = le.transform(y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
param_grid = [{'n_estimators': [30, 35, 40, 45, 50, 60, 70, 80, 90, 100]}]
#,
#               'criterion': ["gini", "entropy"],
#               'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16, 17, "sqrt", "log2"],
#               "max_depth": [1,2,3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]}]
#,
#              "min_samples_split": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#              "min_samples_leaf": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#              "min_weight_fraction_leaf": [0.0,0.25,0.5, 1.0], "bootstrap": [True, False],
#              "oob_score": [True, False], "n_jobs": [-1, 1], "random_state": [1, 3, 5, None],
#              "warm_start": [True, False], 
#              "class_weight": [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]}

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
gridcv = GridSearchCV(clf, param_grid=param_grid, cv=sss, verbose=5)
gridcv.fit(x_train, y_train)

random_forest_tuning_paramters = pd.DataFrame(gridcv.cv_results_)
random_forest_tuning_paramters.to_csv('/home/becca/Insight Project/data files/random_forest parameters.csv', sep=',')