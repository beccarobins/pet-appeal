import pandas as pd
import numpy as np
import petAppeal
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

local_file_path = ''
petfinder_file = local_file_path + 'petfinder_data_clean.'

cats_dogs = pd.read_csv(petfinder_file)
cats_dogs = cats_dogs[(cats_dogs.animal == 'Cat') | (cats_dogs.animal == 'Dog')]

drop_cols = ['Unnamed: 0','address1', 'address2', 'email', 'pet_id', 'phone',
             'breed','lastUpdate', 'name', 'photos','description','zip',
             'city', 'state', 'shelter_id', 'fax', 'id']

cats_dogs = cats_dogs.drop(drop_cols, axis=1)

##Check for class imbalance; downsample if necessary
cats_dogs = petAppeal.balance_check(cats_dogs, 'status')

cats_dogs_encoded = petAppeal.encode_data(cats_dogs)

y = cats_dogs_encoded.status
drop_cols = ['status']
cats_dogs_encoded = cats_dogs_encoded.drop(drop_cols, axis=1)

x = np.array(cats_dogs_encoded)
Classes = y.unique()

##encode labels
le = preprocessing.LabelEncoder()
le.fit(y)

encoded_labels = le.transform(y)
reversed_labels = le.inverse_transform(encoded_labels)

y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

param_grid = [{'n_estimators': range(1,110,10),
               'criterion': ["gini", "entropy"],
               'max_features': range(1,18)+ ["sqrt", "log2"],
               'max_depth': range(1,55,5),
               'min_samples_split': range(10,110,10),
               'min_samples_leaf': range(10,110,10),
               'min_weight_fraction_leaf': [0.0, 0.25, 0.50, 0.75, 1.0],
               'bootstrap': [True, False],
               'oob_score': [True, False],
               'n_jobs': [-1, 1],
               'random_state': [1, 3, 5, None],
               'warm_start': [True, False], 
               'class_weight': [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    sss = StratifiedShuffleSplit(n_splits=3,
                                 test_size=0.2,
                                 random_state=0)
    clf = GridSearchCV(RandomForestClassifier(),
                       param_grid=param_grid,
                       cv=sss,
                       verbose=5,
                       scoring='%s_macro' % score)
    
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:", clf.best_params_)
    
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    y_true, y_pred = y_test, clf.predict(x_test)
    print("Detailed classification report:", classification_report(y_true, y_pred))
    

rForest_GridSearch_results = pd.DataFrame(clf.cv_results_)
rForest_GridSearch_results.to_csv(local_file_path+'rForest_GridSearch_results.csv')

#Use best estimator found in GridSearch for initial modeling
model_rForest = clf.best_estimator_
model_rForest.fit(x_train, y_train)
model_rForest.score(x_train, y_train)

y_pred = model_rForest.predict(x_test)
y_pred_prob = model_rForest.predict_proba(x_test)[:,1]

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

petAppeal.plot_confusion_matrix(cnf_matrix, classes=Classes, title='Random Forest - Default Model')
petAppeal.plot_confusion_matrix(cnf_matrix, classes=Classes, normalize=True, title='Random Forest - Default Model')

print 'Accuracy:', accuracy_score(y_test, y_pred)
print 'Precision:', precision_score(y_test, y_pred)
print 'Recall:', recall_score(y_test, y_pred)
print 'F1:', f1_score(y_test, y_pred)
print classification_report(y_test, y_pred)

importances = model_rForest.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_rForest.estimators_], axis=0)
featureHeaders = list(cats_dogs_encoded)

petAppeal.plot_feature_importance(x_train, importances, featureHeaders, 'black')

model_name = 'petfinder_trained_RF_classifier'
petAppeal.saveVar(model_rForest, model_name)
