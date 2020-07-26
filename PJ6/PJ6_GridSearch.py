#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:33:16 2020

@author: francois


PJ6 Openclassrooms : this script does Grid search validation and saves results
Results are saved in pickle file and loadable in notebook PJ6_modelisation

Usage :
    Launch a python3 console
    Run PJ6_GridSearch_prerequisites.py : exec(open('PJ6_GridSearch_prerequisites.py').read())
    Run this script (uncomment the validation run you want to pass)
        > exec(open('PJ6_GridSearch.py').read())
        
        
First run of GridSearch was done with :
    - 4 models  (KNN 5 or 10,  and doc2vec 10 or 200)
    - GRIDSEARCH_FILE_PREFIX = 'grid_search_results_'

"""

import sys, importlib


# The first time PJ6_GridSearch1.py is called by the interpretor : first "from...import" needs to be called
# But the second time, "from...import" must be called after reload
from functions import *
importlib.reload(sys.modules['functions'])
from functions import *

from sklearn.model_selection import StratifiedKFold

# Label encoding will be used for stratified split between all multi labels
from sklearn.preprocessing import LabelEncoder

#from sklearn.metrics import fbeta_score, make_scorer
#ftwo_scorer = make_scorer(fbeta_score, beta=2)

from sklearn.metrics import make_scorer

def get_new_labels(y):
    y_new = LabelEncoder().fit_transform([''.join(str(y.loc[l,:])) for l in y.index])
    return y_new

def precision_score_micro(y_true, y_pred):
    return(precision_score(y_true, y_pred, average='micro'))


# Pipeline and GridSearch
df_train = df_train_ori
df_test = df_test_ori


prediction_pipeline = Pipeline([
    ('doc2vec', Doc2Vec_Vectorizer(model_path=None, feature_totransform='all_text', n_dim=200)),
    #('features_selector', FeaturesSelector(features_toselect=['Tags'])),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5),)
    ])
    

param_grid = {
        'doc2vec__n_dim': [200, 10],
        'knn__n_neighbors': [5, 10], 
        }

print('\nGrid Search launch')

# USAGE : uncomment the part of code corresponding to the type of search you want to to
if (RECOMPUTE_GRIDSEARCH == True):
    model = prediction_pipeline
    
    kfolds = StratifiedKFold(5)
    
    scorer = make_scorer(precision_score_micro, greater_is_better=True)
    
    # Concatenate labels on 1 column only, to feed them to stratified split
    '''
    print('Calculate 1d labels for stratified split...')
    df_train_labels_1d = get_new_labels(df_train_labels)
    '''

    print('Launch grid search (good night)')
    # To do : add scoring function
    #grid_search = GridSearchCV(model, param_grid, verbose=10, error_score=np.nan, scoring=scorer, cv=kfolds.split(df_train, df_train_labels_1d), iid=False)
    grid_search = GridSearchCV(model, param_grid, verbose=10, error_score=np.nan, scoring=scorer, cv=5, iid=False)
    grid_search.fit(df_train, df_train_labels)

    print('Save grid search model to file')
    grid_search_res, df_grid_search_results_res = save_or_load_search_params(grid_search, 'gridsearch_PJ6')

