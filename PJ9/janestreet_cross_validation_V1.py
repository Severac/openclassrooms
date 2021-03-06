# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Thanks to https://www.kaggle.com/marketneutral/purged-time-series-cv-xgboost-optuna for cross validation strategy

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

#import janestreet
#env = janestreet.make_env() # initialize the environment

#!pip install datatable # Internet is not activated in this competition
#!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl
#import datatable as dt

import pickle
#MODEL_FILE = '/kaggle/working/model.pickle'
DATASET_INPUT_FILE = 'train.csv'
MODEL_FILE = 'model.bin'
MODEL_FILE_RETRAINED = 'model_retrained.bin'
HYPEROPT_RESULT_FILE = 'hyperopt_results.csv'
LOGFILE = 'janestreet_cross_validation.log'
HYPEROPT_TRIAL_RESULTS = 'trials_results.pkl'
NB_HYPEROPT_MAX_RUNS = 100

XGBOOST_MODEL_INTERFACE = 'sklearn' # or 'xgboost'
#XGBOOST_MODEL_INTERFACE = 'xgboost' # or 'xgboost'

LOAD_DF = False

CALCULATE_MACD = False
PREDICT_THRESHOLD = 0.5
TIMESERIES_SPLIT = True
DO_GRIDSEARCH = False
DO_HYPEROPT = True
SPLIT_TRAIN_TEST = False
RETRAIN_ON_ALL_DATA = False

import xgboost as xgb
import gc

from scipy import stats

import functools # For functools.partial

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import traceback
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import datetime

import sys
old_stdout = sys.stdout

log_file = open(LOGFILE,"a+")
sys.stdout = log_file

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#INPUT_DIR = '/kaggle/input/jane-street-market-prediction/'

pd.set_option('display.max_rows', 500)

# Load data
    
df = pd.read_csv(DATASET_INPUT_FILE)
df['resp_positive'] = ((df['resp'])>0)*1  # Target to predict
    
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

#if (CALCULATE_MACD == True):
    #FEATURES_TODROP = ['feature_'+str(i)+'_macd' for i in range(1, 130)] + ['feature_'+str(i)+'_macd_minus_signal' for i in range(1,130)]
#    FEATURES_TODROP = ['feature_'+str(i)+'_macd' for i in range(1, 130)]

# %% [code]
#if (CALCULATE_MACD == True):
#    #df.drop(FEATURES_TODROP, axis=1,inplace=True)
#    pass

# Drop of 0 weights
#df.drop(df[df['weight'] == 0].index, axis=0, inplace=True)
#df.reset_index(drop=True, inplace=True)


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
            #objective= 'binary:logistic',
            #disable_default_eval_metric=True,
            )
        
        
    def __init___old(self, features=FEATURES_LIST_TOTRAIN, random_state= 42,
                max_depth= 12,
                n_estimators= 500,
                learning_rate= 0.01,
                subsample= 0.9,
                colsample_bytree= 0.3,
                tree_method= 'gpu_hist'):

        self.fitted = False
        self.features = features
        self.random_state = random_state
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        
        self.model_internal = XGBClassifier(
            random_state= self.random_state,
            max_depth= self.max_depth,
            n_estimators= self.n_estimators,
            learning_rate= self.learning_rate,
            subsample= self.subsample,
            colsample_bytree= self.colsample_bytree,
            tree_method= self.tree_method,
            #objective= 'binary:logistic',
            #disable_default_eval_metric=True,
            )

    def fit(self, X, y=None):
        print('Model used for fitting:')
        print(self.model_internal)
        self.model_internal.fit(X[self.features], y)
        
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


params_space = {
   'features': hp.choice('features', [['feature_'+str(i) for i in range(130)], 
                                      ['feature_'+str(i) for i in range(130)] + ['weight']]), 
    'random_state': hp.choice('random_state', [42]),
    'max_depth': hp.choice('max_depth', [8, 10, 12, 15, 20]),
    'n_estimators': hp.choice('n_estimators', [50, 250, 500, 1000]),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
    'subsample': hp.choice('subsample', [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
    'tree_method': hp.choice('tree_method', ['gpu_hist'])
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

    for train_index, test_index in cv.split(df, (df['resp'] > 0)*1, df['date']):
        model_wrapped.fit(df.loc[train_index], (df.loc[train_index]['resp'] > 0).astype(np.byte))
        test_predictions = model_wrapped.predict(df.loc[test_index])
        scores.append(model_wrapped.score(df.loc[test_index], test_predictions))
        accuracy_scores.append(model_wrapped.accuracy_score(df.loc[test_index], test_predictions))    
    
    return({'utility_score': sum(scores), 'utility_scores': scores, 'utility_score_std': np.std(scores), 'accuracy_scores': accuracy_scores})

def f(params):
    print('New call of f')
    scores_obj = hyperopt_train_test(params)
    return {'loss': -scores_obj['utility_score'], 'utility_scores': scores_obj['utility_scores'], 'utility_score_std': scores_obj['utility_score_std'], 'accuracy': scores_obj['accuracy_scores'], 'status': STATUS_OK}


def run_a_trial():
    max_evals = nb_evals = 1

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(HYPEROPT_TRIAL_RESULTS, "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(f, params_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    
    print('Best model so far :')
    print(best)
    
    pickle.dump(trials, open(HYPEROPT_TRIAL_RESULTS, "wb"))

    trial_ids = []
    trial_losses = []
    trial_utility_scores = []
    trial_utility_std = []
    trial_accuracy = []
    trial_status = []
    trial_vals = []
    trial_book_time = []
    
    
    for trial in trials.trials:
        #print(trial)
        trial_ids.append(trial['tid'])
        trial_losses.append(trial['result']['loss'])
        trial_utility_scores.append(trial['result']['utility_scores'])
        trial_utility_std.append(trial['result']['utility_score_std'])
        trial_accuracy.append(trial['result']['accuracy'])
        trial_status.append(trial['result']['status'])
        trial_vals.append(trial['misc']['vals'])
        trial_book_time.append(str(trial['book_time']))
    
        
        #display(pd.DataFrame(trial).T)
        #print('\n\n')
    
    #pd.options.display.max_colwidth = 1000
    
    df_hyperopt_results = pd.DataFrame({'ids': trial_ids, 'status': trial_status, 'loss': trial_losses, 'utility_scores': trial_utility_scores, 'utility_score_std': trial_utility_std, 'accuracy': trial_accuracy, 'vals': trial_vals, 'book_time': trial_book_time})    
    df_hyperopt_results.to_csv(HYPEROPT_RESULT_FILE)

    print("\nOPTIMIZATION STEP COMPLETE.\n")

print("START OF OPTIMIZATION:")

nb = 0
while (True):    
    # Optimize a new model with the TPE Algorithm:
    print(f'Run {nb} since program start')
    print(datetime.datetime.now())
    try:
        run_a_trial()
        nb += 1
        
    except Exception as err:
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)



sys.stdout = old_stdout

log_file.close()
