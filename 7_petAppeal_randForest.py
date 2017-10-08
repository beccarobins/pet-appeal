#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 18:09:27 2017

@author: becca
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.model_selection import cross_val_score, cross_val_predict

path = '/home/becca/Insight Project/data files'
os.chdir(path)

file = '/home/becca/Insight Project/data files/cats_dogs_munged.csv'
cats_dogs_munged = pd.read_csv(file)
cats_dogs_munged  = cats_dogs_munged.drop(labels=['Unnamed: 0'], axis=1)

y = cats_dogs_munged.status
cats_dogs_munged = cats_dogs_munged.drop(labels=['status'], axis=1)
x = cats_dogs_munged

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

Classes = ['Available','Adopted']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=1)

def saveVar(variable_to_save, file_name):
    with open(file_name+'.pickle',"wb") as f:
        pickle.dump(variable_to_save, f)

def plotROC(y_test, y_pred, model_str):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc=auc(fpr, tpr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(fpr, tpr, label='AUC=%0.2f'% roc_auc, color='#0bc7ff',linewidth=2.0)
    plt.ylabel('True Positive Rate', fontsize=(18), color='white')
    plt.xlabel('False Positive Rate', fontsize=(18), color='white')
    plt.tick_params(axis='both', which='major', labelsize=14, color='white')
    plt.tick_params(axis='both', which='minor', labelsize=14, color='white')
    plt.title('ROC Curve', fontsize=(18), color='white', fontweight='bold')
    leg = plt.legend(framealpha = 0, loc = 'lower right', fontsize=(14), frameon=False)
    for text in leg.get_texts():
        plt.setp(text, color = 'w')
    plt.plot([0,1],[0,1], color='#f1b82d', linestyle='--', linewidth=2.0)
    axes = plt.gca()
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_visible(False)
    ax.xaxis.label.set_color('white')
    ax.tick_params(axis='both', colors='white')
    plt.tight_layout()
    plt.show()
    fname = model_str+' ROC Curve.png'
    fig.savefig(fname, transparent=True)
    plt.close()
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig = plt.figure(1)
    fname = title +'.png'
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=(18), fontweight='bold')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=(14))
    plt.yticks(tick_marks, classes, fontsize=(14))

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    
    plt.ylabel('True label', fontsize=(18), fontweight='bold')
    plt.xlabel('Predicted label', fontsize=(18), fontweight='bold')
    plt.tight_layout()
    plt.show()
    fig.savefig(fname, transparent=True)
    plt.close()
    


model_rForest = RandomForestClassifier()
model_rForest.fit(x_train, y_train)
model_rForest.score(x_train, y_train)

#Predict Output
y_pred = model_rForest.predict(x_test)
y_score = model_rForest.score(x_test, y_test)
y_pred_prob = model_rForest.predict_proba(x_test)[:,1]

print(sklearn.metrics.f1_score(y_test, y_pred, average='binary'))
print(sklearn.metrics.accuracy_score(y_test, y_pred))

#plotROC(y_test, y_pred_prob, 'Random Forest - Default')

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

plot_confusion_matrix(cnf_matrix, classes=Classes, title='Random Forest - Default Model')
plot_confusion_matrix(cnf_matrix, classes=Classes, normalize=True, title='Random Forest - Default Model')

####Choose cross-validation iterator

#The ShuffleSplit iterator will generate a user defined number of 
#independent train / test dataset splits. Samples are first shuffled and 
#then split into a pair of train and test sets.
#It is possible to control the randomness for reproducibility of the results 
#by explicitly seeding the random_state pseudo random number generator.
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
val_scores = cross_val_score(model_rForest, x, y, cv=sss)
print val_scores
mean_val_score = np.mean(val_scores)
print mean_val_score
sd_val_score = np.std(val_scores)
print sd_val_score


#y_pred_prob_val = cross_val_predict(model_rForest, x_test, y_test, cv=sss, method='predict_proba')

#from sklearn.model_selection import StratifiedKFold
##Some classification problems can exhibit a large imbalance in the distribution 
##of the target classes: for instance there could be several times more negative
## samples than positive samples. In such cases it is recommended to use 
##stratified sampling as implemented in StratifiedKFold and StratifiedShuffleSplit
## to ensure that relative class frequencies is approximately preserved in each 
##train and validation fold.
#
###StratifiedKFold is a variation of k-fold which returns stratified folds: 
##each set contains approximately the same percentage of samples of 
##each target class as the complete set.
#
##RepeatedStratifiedKFold can be used to repeat Stratified 
##K-Fold n times with different randomization in each repetition.
#
##X = np.ones(10)
##y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
#skf = StratifiedKFold(n_splits=3)
#for train, test in skf.split(x, y):
#    print("%s %s" % (train, test))

#plotROC(y_test, y_pred_prob_val, 'Random Forest - validated')
#Classes = ['Adopted','Not Adopted']
#
#cnf_matrix = confusion_matrix(y_test, y_pred_val)
#np.set_printoptions(precision=2)
#
#plot_confusion_matrix(cnf_matrix, classes=Classes, title='Random Forest - Confusion matrix, without normalization')
#plot_confusion_matrix(cnf_matrix, classes=Classes, normalize=True, title='Random Forest - Normalized confusion matrix')

#model_rForest = RandomForestClassifier(max_depth=3, min_samples_leaf=2, min_samples_split=4, min_weight_fraction_leaf=0, max_features='sqrt', n_estimators=50, criterion = 'entropy')
#model_rForest.fit(x_train, y_train)
#model_rForest.score(x_train, y_train)
#
##Predict Output
#y_pred = model_rForest.predict(x_test)
#y_score = model_rForest.score(x_test, y_test)
#y_pred_prob = model_rForest.predict_proba(x_test)[:,1]
#
#sklearn.metrics.f1_score(y_test, y_pred, average='binary')
#sklearn.metrics.accuracy_score(y_test, y_pred)
model_rForest_tuned = RandomForestClassifier(criterion = 'entropy', n_estimators=90, max_features=4, max_depth=15)
model_rForest_tuned.fit(x_train, y_train)
model_rForest_tuned.score(x_train, y_train)

#Predict Output
y_pred = model_rForest_tuned.predict(x_test)
y_score = model_rForest_tuned.score(x_test, y_test)
y_pred_prob = model_rForest_tuned.predict_proba(x_test)[:,1]

print(sklearn.metrics.f1_score(y_test, y_pred, average='binary'))

print(sklearn.metrics.accuracy_score(y_test, y_pred))
print(sklearn.metrics.precision_score(y_test, y_pred))
print(sklearn.metrics.recall_score(y_test, y_pred))
print(sklearn.metrics.f1_score(y_test, y_pred))
print(sklearn.metrics.classification_report(y_test, y_pred))

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
val_scores = cross_val_score(model_rForest_tuned, x, y, cv=sss)
print val_scores
mean_val_score = np.mean(val_scores)
print mean_val_score
sd_val_score = np.std(val_scores)
print sd_val_score

plotROC(y_test, y_pred_prob, 'Random Forest - Tuned')

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

plot_confusion_matrix(cnf_matrix, classes=Classes, title='Random Forest Classification')
plot_confusion_matrix(cnf_matrix, classes=Classes, normalize=True, title='Random Forest Classification')

featureHeaders = list(cats_dogs_munged)

importances = model_rForest.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_rForest.estimators_], axis=0)

indices = np.argsort(importances)[::-1]

arr1 = indices
arr2 = np.array(featureHeaders) #featureHeaders is the name of my list of features
sorted_arr2 = arr2[arr1[::1]]#this bit of code sorts my list of features according to the order yielded by rf.feature_importances_

# Print the feature ranking

print("Feature ranking:")

for f in range(x_train.shape[1]):
  print("%d. %s (%f)" % (f + 1, sorted_arr2[f], importances[indices[f]]))


# Plot the feature importances of the forest
###WHITE FONTS
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title("Feature Importance", fontsize=(22), fontweight='bold', color='white')
plt.barh(range(x_train.shape[1]), importances[indices], color="#f1b82d")
ax.invert_yaxis()
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color('white')
ax.spines['right'].set_visible(False)
plt.yticks(range(x_train.shape[1]), sorted_arr2, ha='right')
plt.tick_params(axis='both', which='major', labelsize=14, color='white')
plt.tick_params(axis='both', which='minor', labelsize=14, color='white')
#plt.xlim([-1, x_train.shape[1]])
plt.gcf().set_size_inches(8,8)
##ax.invert_xaxis()
#ax.spines['top'].set_visible(False)
plt.tight_layout()
ax.tick_params(axis='both', colors='white')
plt.show()
model_str = 'Random Forest'
fname = model_str+' Feature Importance.png'
fig.savefig(fname, transparent=True)
plt.close()


##black fonts
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title("Feature Importance", fontsize=(22), fontweight='bold')
plt.barh(range(x_train.shape[1]), importances[indices], color="#f1b82d")
ax.invert_yaxis()
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color('white')
ax.spines['right'].set_visible(False)
plt.yticks(range(x_train.shape[1]), sorted_arr2, ha='right')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
#plt.xlim([-1, x_train.shape[1]])
plt.gcf().set_size_inches(8,8)
##ax.invert_xaxis()
#ax.spines['top'].set_visible(False)
plt.tight_layout()
ax.tick_params(axis='both')
plt.show()
model_str = 'Random Forest'
fname = model_str+' Feature Importance.png'

saveVar(model_rForest_tuned, 'Trained_Random_Forest')
