#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 00:25:25 2020

@author: francois


V2 : 
    With models from "21/12 matin"
    
    Best model identified at iteration 113 of hyperopt
    Visible in cross_validation_analyze_notebookV2.ipynb
    
    Model parameters got from convert_params_hyperopt_to_xgboost.py
    
V3 : 
    With models from "22/12 soir"

V4 : 
    With models from 21/12 (without gamma), with sample_weight hyper param as xgboost input

        
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

DATASET_INPUT_FILE = 'train.csv'

TRAIN_PERIMETER_SPLITS = True
TRAIN_PERIMETER_ALL = True
CALCULATE_MACD = False

MODEL_FILE = 'model.bin'

LOGFILE = 'janestreet_train_model.log'

import datetime

pd.set_option('display.max_rows', 500)

import sys
old_stdout = sys.stdout

log_file = open(LOGFILE,"a+")
sys.stdout = log_file

print(datetime.datetime.now())
print('Start of training script. Loading data...')

# Load data
    
df = pd.read_csv(DATASET_INPUT_FILE)
df['resp_positive'] = ((df['resp'])>0)*1  # Target to predict

print('Data loaded')
    
# Temporal features    
if (CALCULATE_MACD == True):
    #FEATURES_FOR_MACD = ['feature_'+str(i) for i in range(1,130)]
    FEATURES_FOR_MACD = ['feature_41', 'feature_42', 'feature_43', 'feature_44', 'feature_45']

    for feature in FEATURES_FOR_MACD:
        #df.loc[:, feature + '_macd'] = df[feature].ewm(span=12, adjust=False).mean().astype('float32') # Short term exponential moving average\
        #- df[feature].ewm(span=26, adjust=False).mean().astype('float32') # Long term exponential moving average
        
        macd_feature = df[feature].ewm(span=12, adjust=False).mean().astype('float32') # Short term exponential moving average\
        - df[feature].ewm(span=26, adjust=False).mean().astype('float32') # Long term exponential moving average

        # We calculate MACD (which is short term EWMA (span 12) minus long term EWMA (span 26)) minus signal (which is MACD EWMA span 9)
        df.loc[:, feature + '_macd_minus_signal'] = macd_feature - macd_feature.ewm(span=9, adjust=False).mean().astype('float32')
        df.loc[:, feature + '_macd'] = macd_feature
        
        #del macd_feature
        gc.collect()
    

#df = reduce_mem_usage(df)

# Remove first half of the data
#df.drop(index=df[df['date'] <= 249].index, inplace=True)

if (CALCULATE_MACD == True):
    FEATURES_MACD = [feat + '_macd_minus_signal' for feat in FEATURES_FOR_MACD] + [feat + '_macd' for feat in FEATURES_FOR_MACD]
    
    

# Split train test

# %% [code]
# TODO: make GitHub GIST
# TODO: add as dataset
# TODO: add logging with verbose

from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]
            
from matplotlib.colors import ListedColormap
    
# this is code slightly modified from the sklearn docs here:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    
    cmap_cv = plt.cm.coolwarm

    jet = plt.cm.get_cmap('jet', 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)   # inplace
    cmap_data = ListedColormap(jet(seq))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.Set3)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['target', 'day']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


# Features definition
#FEATURES_LIST_TOTRAIN = ['feature_'+str(i) for i in range(130)]
FEATURES_LIST_TOTRAIN = ['feature_'+str(i) for i in range(130)] + ['weight']
#FEATURES_LIST_TOTRAIN = [feat for feat in FEATURES_LIST_TOTRAIN if feat not in FEATURES_TODROP]

if (CALCULATE_MACD == True):
    #FEATURES_LIST_TOTRAIN.extend(['feature_'+str(i)+'_macd_minus_signal' for i in range(1,130)])  
    FEATURES_LIST_TOTRAIN.extend(FEATURES_MACD)  

# %% [code]
#FEATURES_LIST = ['feature_'+str(i) for i in range(130)] + ['feature_'+str(i)+'_macd' for i in range(1, 130)] + ['feature_'+str(i)+'_macd_minus_signal' for i in range(1,130)]+  ['weight']

# Utility calculation function

def utility_function(df_test, df_test_predictions):
    df_test.loc[:, 'utility_pj'] = df_test['weight'] * df_test['resp'] * df_test_predictions
    df_test_utility_pi = df_test.groupby('date').sum('utility_pj')['utility_pj']
    nb_unique_dates = df_test_utility_pi.shape[0]
    t = (df_test_utility_pi.sum() / np.sqrt(df_test_utility_pi.pow(2).sum())) * (np.sqrt(250 / np.abs(nb_unique_dates)))
    u = min(max(t, 0), 6) * df_test_utility_pi.sum()
    
    return(u)

# This function accounts for variable instance counts in each split by dividing utility_pi by number of instances (but this has been removed)
# It also does some copy of dataframe to prevent memory overwrite
def utility_function_normalized(df_test, df_test_predictions):
    df_test_copy = df_test.copy(deep=True)
    df_test_copy.loc[:, 'utility_pj'] = df_test_copy['weight'] * df_test_copy['resp'] * df_test_predictions
    #df_test_utility_pi = df_test_copy.groupby('date')['utility_pj'].sum() / df_test_copy.groupby('date')['utility_pj'].count()
    df_test_utility_pi = df_test_copy.groupby('date')['utility_pj'].sum()

    nb_unique_dates = df_test_utility_pi.shape[0]
    t = (df_test_utility_pi.sum() / np.sqrt(df_test_utility_pi.pow(2).sum())) * (np.sqrt(250 / np.abs(nb_unique_dates)))
    u = min(max(t, 0), 6) * df_test_utility_pi.sum()
    del df_test_copy
    
    return(u)

# Model wrapper

# %% [code]
# https://scikit-learn.org/stable/developers/develop.html#estimator-types
from sklearn.base import BaseEstimator, ClassifierMixin

class XGBClassifier_wrapper(BaseEstimator, ClassifierMixin):  
    ''' Params passed as dictionnary to __init__, for example :
        params_space = {
       'features': FEATURES_LIST_TOTRAIN, 
        'random_state': 42,
        'max_depth': 12,
        'n_estimators': 500,
        'learning_rate': 0.01,
        'subsample': 0.9,
        'colsample_bytree': 0.3,
        'tree_method': 'gpu_hist'
        }
    '''
    def __init__(self, params):
        self.fitted = False
        
        self.features = list(params['features'])
        self.random_state = params['random_state']
        self.max_depth = params['max_depth']
        self.n_estimators = params['n_estimators']
        self.learning_rate = params['learning_rate']
        self.subsample = params['subsample']
        self.colsample_bytree = params['colsample_bytree']
        self.gamma = params['gamma']
        self.tree_method = params['tree_method']  
        
        #print('Features assigned :')
        #print(self.features)

        self.model_internal = XGBClassifier(
            random_state= self.random_state,
            max_depth= self.max_depth,
            n_estimators= self.n_estimators,
            learning_rate= self.learning_rate,
            subsample= self.subsample,
            colsample_bytree= self.colsample_bytree,
            tree_method= self.tree_method,
            gamma = self.gamma,
            #objective= 'binary:logistic',
            #disable_default_eval_metric=True,
            )

    def fit(self, X, y=None):
        print('Model used for fitting:')
        print(self.model_internal)
        self.model_internal.fit(X[self.features], y, sample_weight=X['weight'])
        
        self.fitted = True
        return self

    def predict(self, X, y=None):
        if (self.fitted == True):
            print('predict called')
            return(self.model_internal.predict(X[self.features]))
        
        else:
            print('You must fit model first')
            return(None)

    def predict_proba(self, X, y=None):
        if (self.fitted == True):
            print('predict proba called')
            return(self.model_internal.predict_proba(X[self.features]))
        
        else:
            print('You must fit model first')
            return(None)
        

    #def set_params(self, **parameters):
    #    for parameter, value in parameters.items():
    #        setattr(self, parameter, value)

        
    def score(self, X, y=None):        
        print('Type of X:')
        print(type(X))
        
        print('Shape of X:')
        print(X.shape)
        
        print('Type of y:')
        print(type(y))
        
        print('model fitted ?')
        print(self.fitted) # Usually returns yes at this point when called by cross_val_score
        
        if y is None:
            print('y is None')
            y_preds = pd.Series(self.model_internal.predict(X.reset_index(drop=True)[self.features]))
            
        else: # cross_val_score goes there
            print('y is not None')
            y_preds = pd.Series(y)
        
        return(utility_function_normalized(X.reset_index(drop=True), y_preds)) 
    
    def accuracy_score(self, X, y=None):
        if y is None:
            print('y is None in accuracy_score method : pass predictions as y to avoid launching predict')
            y_preds = pd.Series(self.model_internal.predict(X.reset_index(drop=True)[self.features]))
            
        else: # cross_val_score goes there
            #print('y is not None')
            y_preds = pd.Series(y)
            
        return(accuracy_score(X['resp_positive'], y_preds))
                
#np.seterr('raise')

# Hyper parameters optimisation

# This variable is original params space in cross validation script. It is here for information purpose, and not used in this current script.
# It allows to know the match between internal hyperopt parameters and real parameters
# For example,  'max_depth' hyperopt parameter of 1 corresponds to a value of 8 (and param 2 to a value of 10) if max_depth is hp.choice('max_depth', [8, 10, 12, 15, 20])

hyperopt_params_space = {  
   'features': [['feature_'+str(i) for i in range(130)]], 
    'random_state': [42],
    'max_depth': [8, 9, 10],
    'n_estimators': [250, 500],
    'learning_rate': [0.01, 0.02, 0.1],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.2, 0.3, 0.6, 0.9],
    'gamma': [0.01, 0.1, 0.5, 1, 10],
    'tree_method': ['gpu_hist']
}

def hyperopt_train_test(params):
    print('New call of hyperopt_train_test')
    model_wrapped = XGBClassifier_wrapper(params)
    
    cv = PurgedGroupTimeSeriesSplit(
        n_splits=5,
        #n_splits=5,
        #max_train_group_size=150,
        max_train_group_size=180,
        group_gap=20,
        max_test_group_size=60
    )

    scores = []
    accuracy_scores = []

    i = 0
    for train_index, test_index in cv.split(df, (df['resp'] > 0)*1, df['date']):
        i += 1
        model_wrapped.fit(df.loc[train_index], (df.loc[train_index]['resp'] > 0).astype(np.byte))
        test_predictions = model_wrapped.predict(df.loc[test_index])
        scores.append(model_wrapped.score(df.loc[test_index], test_predictions))
        accuracy_scores.append(model_wrapped.accuracy_score(df.loc[test_index], test_predictions))  
        
        df_featimportance = pd.DataFrame(model_wrapped.model_internal.feature_importances_, index=df[model_wrapped.features].columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
        df_featimportance_cumulated = pd.concat([df_featimportance, pd.DataFrame({'% feat importance cumulé' : (df_featimportance['Importance'] / df_featimportance['Importance'].sum()).cumsum()})], axis=1)
        print(f'Feature importances for split {i}:')
        print(df_featimportance_cumulated)
    
    return({'utility_score': sum(scores), 'utility_scores': scores, 'utility_score_std': np.std(scores), 'accuracy_scores': accuracy_scores})

# Best models retrained in this script :

iteration_names = ['best_utility4_withoutgamma', 'best_utility5_withoutgamma']
iteration_params = [
        {
'features': ['feature_'+str(i) for i in range(130)],
'random_state': 42,
'max_depth': 10,
'n_estimators': 500,
'learning_rate': 0.02,
'subsample': 0.5,
'colsample_bytree': 0.9,
'gamma': None,
'tree_method': 'gpu_hist'
},
         
         {
'features': ['feature_'+str(i) for i in range(130)],
'random_state': 42,
'max_depth': 10,
'n_estimators': 500,
'learning_rate': 0.02,
'subsample': 0.5,
'colsample_bytree': 0.6,
'gamma': None,
'tree_method': 'gpu_hist'
},
        ]


n = 0 # Size of n is the len of iteration_params and iteration_names
for params in iteration_params:
    print(f'Starting training for iteration {n}')
    print(datetime.datetime.now())
    print('\n')
    
    if (TRAIN_PERIMETER_SPLITS == True):
        print('Train perimeter == splits')
        training_results = hyperopt_train_test(params)
        
        print('Training results:')
        print(training_results)
        
        # To do: add proba thresholds here
                
    if (TRAIN_PERIMETER_ALL == True):
        print('Train perimeter == ALL')
        model_wrapped = XGBClassifier_wrapper(params)
        
        print('Fit model on all data')
        model_wrapped.fit(df, (df['resp'] > 0).astype(np.byte))
        
        print('Save model')
        model_wrapped.model_internal.save_model(MODEL_FILE + iteration_names[n])
        
        df_featimportance = pd.DataFrame(model_wrapped.model_internal.feature_importances_, index=df[model_wrapped.features].columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
        df_featimportance_cumulated = pd.concat([df_featimportance, pd.DataFrame({'% feat importance cumulé' : (df_featimportance['Importance'] / df_featimportance['Importance'].sum()).cumsum()})], axis=1)
        print(f'Feature importances for ALL data:')
        print(df_featimportance_cumulated)    

    n += 1

sys.stdout = old_stdout
log_file.close()
