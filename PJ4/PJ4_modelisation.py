#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ4 : transats dataset : modelisation notebook

# # Global variables and functions used in the notebook

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import zipfile
import urllib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import qgrid

import glob

from pandas.plotting import scatter_matrix

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV

# Final model evaluation measures, for customer :
# EVALUATION_PERCENT of the time, prediction errors will be below EVALUATION_THRESHOLD minutes
EVALUATION_PERCENT = 0.9  # 90% of the time
EVALUATION_THRESHOLD = 5

SAMPLED_DATA = True  # If True : data is sampled (NB_SAMPLES instances only) for faster testing purposes
NB_SAMPLES = 80000
#NB_SAMPLES = 800000
#NB_SAMPLES = 500000
LEARNING_CURVE_STEP_SIZE = int(NB_SAMPLES / 10) # Change that when you change NB_SAMPLES size

DATA_PATH = os.path.join("datasets", "transats")
DATA_PATH = os.path.join(DATA_PATH, "out")

DATA_PATH_FILE_INPUT = os.path.join(DATA_PATH, "transats_metadata_transformed.csv")


ALL_FEATURES = ['ORIGIN','CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','DEST','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME','ARR_DELAY','DEP_DELAY', 'TAXI_OUT', 'TAIL_NUM', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'NBFLIGHTS_FORDAY_FORAIRPORT']

'''
MODEL1_FEATURES = ['ORIGIN','CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','DEST','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME']
MODEL1_LABEL = 'ARR_DELAY'
'''

MODEL1_FEATURES = ['ORIGIN','CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','DEST','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'NBFLIGHTS_FORDAY_FORAIRPORT']
MODEL1_FEATURES_QUANTITATIVE = ['CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'NBFLIGHTS_FORDAY_FORAIRPORT']

MODEL1_GOUPBYMEAN_FEATURES = ['CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DEST','DISTANCE','CRS_ELAPSED_TIME', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'CRS_ARR_TIME', 'DAY_OF_WEEK', 'NBFLIGHTS_FORDAY_FORAIRPORT', 'ORIGIN', 'UNIQUE_CARRIER']
MODEL_GROUPBYMEAN_FEATURES_QUANTITATIVE = ['CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DISTANCE','CRS_ELAPSED_TIME', 'DEST', 'CRS_ARR_TIME', 'DAY_OF_WEEK', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'ORIGIN', 'UNIQUE_CARRIER' ]


MODEL1bis_FEATURES_QUANTITATIVE = ['CRS_DEP_TIME','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'NBFLIGHTS_FORDAY_FORAIRPORT']
MODEL1_LABEL = 'ARR_DELAY'

MODEL_1HOTALL_FEATURES = ['DISTANCE', 'CRS_ELAPSED_TIME', 'NBFLIGHTS_FORDAY_FORAIRPORT', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN', 'UNIQUE_CARRIER', 'CRS_DEP_TIME']
MODEL_1HOTALL_FEATURES_QUANTITATIVE = ['DISTANCE', 'CRS_ELAPSED_TIME', 'NBFLIGHTS_FORDAY_FORAIRPORT', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT']
# For later : maybe not include CRS_ELAPSED_TIME because close to DISTANCE


MODEL_GROUPBYMEAN2_FEATURES = ['DISTANCE', 'CRS_ELAPSED_TIME', 'NBFLIGHTS_FORDAY_FORAIRPORT', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN', 'UNIQUE_CARRIER', 'CRS_DEP_TIME']
MODEL_GROUPBYMEAN2_FEATURES_QUANTITATIVE = ['DISTANCE', 'CRS_ELAPSED_TIME', 'NBFLIGHTS_FORDAY_FORAIRPORT', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN', 'UNIQUE_CARRIER', 'CRS_DEP_TIME']


MODEL1_3FEATS = ['DISTANCE', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'DEP_DELAY']
MODEL1_3FEATS_QUANTITATIVE = ['DISTANCE', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'DEP_DELAY']

MODEL1_2FEATS = ['DISTANCE', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT']
MODEL1_2FEATS_QUANTITATIVE = ['DISTANCE', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT']



MODEL_cheat_FEATURES = ['ARR_DELAY','ORIGIN','CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','DEST','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'NBFLIGHTS_FORDAY_FORAIRPORT']
MODEL_cheat_FEATURES_QUANTITATIVE = ['ARR_DELAY','CRS_DEP_TIME','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'NBFLIGHTS_FORDAY_FORAIRPORT']

plt.rcParams["figure.figsize"] = [16,9] # Taille par défaut des figures de matplotlib

import seaborn as sns
sns.set()

#import common_functions

####### Paramètres pour sauver et restaurer les modèles :
import pickle
####### Paramètres à changer par l'utilisateur selon son besoin :

RECOMPUTE_GRIDSEARCH = False  # CAUTION : computation is several hours long
SAVE_GRID_RESULTS = False # If True : grid results object will be saved to pickle files that have GRIDSEARCH_FILE_PREFIX
LOAD_GRID_RESULTS = True # If True : grid results object will be loaded from pickle files that have GRIDSEARCH_FILE_PREFIX
                          # Grid search results are loaded with full samples (SAMPLED_DATA must be False)

'''
RECOMPUTE_GRIDSEARCH = True  # CAUTION : computation is several hours long
SAVE_GRID_RESULTS = True # If True : grid results object will be saved to pickle files that have GRIDSEARCH_FILE_PREFIX
LOAD_GRID_RESULTS = False # If True : grid results object will be loaded from pickle files that have GRIDSEARCH_FILE_PREFIX
'''
#GRIDSEARCH_CSV_FILE = 'grid_search_results.csv'

GRIDSEARCH_FILE_PREFIX = 'grid_search_results_'

EXECUTE_INTERMEDIATE_MODELS = False # If True: every intermediate model (which results are manually analyzed in the notebook) will be executed


# Necessary for predictors used in the notebook :
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import PolynomialFeatures


### For progress bar :
from tqdm import tqdm_notebook as tqdm

# Statsmodel : 
import statsmodels.formula.api as smf

import statsmodels.api as sm
from scipy import stats

SAVE_API_MODEL = True # If True : API model ill be saved
API_MODEL_PICKLE_FILE = 'API_model_PJ4.pickle'


# In[2]:


def qgrid_show(df):
    display(qgrid.show_grid(df, grid_options={'forceFitColumns': False, 'defaultColumnWidth': 170}))


# In[3]:


def load_data():
    # hhmm timed features formatted
    feats_hhmm = ['CRS_DEP_TIME',  'CRS_ARR_TIME']

    df = pd.read_csv(DATA_PATH_FILE_INPUT, sep=',', header=0, encoding='utf-8', low_memory=False, parse_dates=feats_hhmm)   
    
    # Drop outliers (low quantile data : extreme delays not enough represented)
    df.drop(index=df[(df['ARR_DELAY'] < df.ARR_DELAY.quantile(.01)) | (df['ARR_DELAY'] > df.ARR_DELAY.quantile(.99))].index, axis=0, inplace=True)
    
    return(df)


# In[4]:


def load_data_with_outliers():
    # hhmm timed features formatted
    feats_hhmm = ['CRS_DEP_TIME',  'CRS_ARR_TIME']

    df = pd.read_csv(DATA_PATH_FILE_INPUT, sep=',', header=0, encoding='utf-8', low_memory=False, parse_dates=feats_hhmm)   
        
    return(df)


# In[5]:


def custom_train_test_split_sample(df):
    from sklearn.model_selection import train_test_split
    
    if (SAMPLED_DATA == True):
        df_labels_discrete = pd.cut(df['ARR_DELAY'], bins=50)
        #df = df.sample(NB_SAMPLES).copy(deep=True)
        df, df2 = train_test_split(df, train_size=NB_SAMPLES, random_state=42, shuffle = True, stratify = df_labels_discrete)
        
    df_labels_discrete = pd.cut(df['ARR_DELAY'], bins=50)
    
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, shuffle = True, stratify = df_labels_discrete)
    #df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    
    df_train = df_train.copy()
    df_test = df_test.copy()

    '''
    # Old code: we sampled only training set. But that's a problem when you encounter values in test set (not sampled) that were not in training set
    if (SAMPLED_DATA == True):
        df_train = df_train.sample(NB_SAMPLES).copy(deep=True)
        df = df.loc[df_train.index]
    '''   
    
    return df, df_train, df_test


# In[6]:


def custom_train_test_split_sample_random(df):
    from sklearn.model_selection import train_test_split
    
    if (SAMPLED_DATA == True):
        df = df.sample(NB_SAMPLES).copy(deep=True)
        
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    
    df_train = df_train.copy()
    df_test = df_test.copy()

    '''
    # Old code: we sampled only training set. But that's a problem when you encounter values in test set (not sampled) that were not in training set
    if (SAMPLED_DATA == True):
        df_train = df_train.sample(NB_SAMPLES).copy(deep=True)
        df = df.loc[df_train.index]
    '''   
    
    return df, df_train, df_test


# In[7]:


def print_column_information(df, column_name):
    column_type = df.dtypes[column_name]
    print(f'Column {column_name}, type {column_type}\n')
    print('--------------------------')

    print(df[[column_name]].groupby(column_name).size().sort_values(ascending=False))
    print(df[column_name].unique())    
    print('\n')


# In[8]:


def display_percent_complete(df):
    not_na = 100 - (df.isnull().sum() * 100 / len(df))
    not_na_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_complete': not_na}).sort_values(by='percent_complete', ascending=False)
    display(not_na_df)


# In[9]:


def identify_features(df):
    all_features = ALL_FEATURES

    model1_features = MODEL1_FEATURES
    model1_label = MODEL1_LABEL
    
    quantitative_features = []
    qualitative_features = []
    features_todrop = []

    for feature_name in all_features:
        if (df[feature_name].dtype == 'object'):
            qualitative_features.append(feature_name)

        else:
            quantitative_features.append(feature_name)

    print(f'Quantitative features : {quantitative_features} \n')
    print(f'Qualitative features : {qualitative_features} \n')  
    
    return all_features, model1_features, model1_label, quantitative_features, qualitative_features


# In[10]:


def save_or_load_search_params(grid_search, save_file_suffix):
    if (SAVE_GRID_RESULTS == True):
        #df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
        #df_grid_search_results.to_csv(GRIDSEARCH_CSV_FILE)

        df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["mean_test_score"])],axis=1)
        df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["std_test_score"], columns=["std_test_score"])],axis=1)
        df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_fit_time"], columns=["mean_fit_time"])],axis=1)
        df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_score_time"], columns=["mean_score_time"])],axis=1)
        df_grid_search_results.to_csv(GRIDSEARCH_FILE_PREFIX + save_file_suffix + '.csv')

        with open(GRIDSEARCH_FILE_PREFIX + save_file_suffix + '.pickle', 'wb') as f:
            pickle.dump(grid_search, f, pickle.HIGHEST_PROTOCOL)
            
        return(grid_search, df_grid_search_results)

    if (LOAD_GRID_RESULTS == True):
        if ((SAVE_GRID_RESULTS == True) or (RECOMPUTE_GRIDSEARCH == True)):
            print('Error : if want to load grid results, you should not have saved them or recomputed them before, or you will loose all your training data')

        else:
            with open(GRIDSEARCH_FILE_PREFIX + save_file_suffix + '.pickle', 'rb') as f:
                grid_search = pickle.load(f)

            df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["mean_test_score"])],axis=1)
            df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["std_test_score"], columns=["std_test_score"])],axis=1)
            df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_fit_time"], columns=["mean_fit_time"])],axis=1)
            df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_score_time"], columns=["mean_score_time"])],axis=1)
            
            return(grid_search, df_grid_search_results)


# In[11]:


def evaluate_model(model, X_test, Y_test):
    Y_predict = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_predict)
    rmse = np.sqrt(mse)
    print(f'RMSE : {rmse}')
    


# In[12]:


def evaluate_model_MAE(model, X_test, Y_test):
    Y_predict = model.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_predict)
    print(f'MAE : {mae}')
    


# In[13]:


'''
This function returns the % of absolute errors of the model that are < threshold, percent of the time
'''

def evaluate_model_percent_threshold(model, X_test, Y_test, percent, threshold):
    Y_predict = model.predict(X_test)
    
    Y_AE = np.abs(Y_predict- Y_test)
    Y_AE_best = Y_AE[Y_AE <= Y_AE.quantile(percent)] # Take percent best error values (eliminate errors > Y_AE.quantile(percent))
    
    error_percent_threshold = (len(Y_AE_best[Y_AE_best < threshold]) / len(Y_AE_best)) * 100
    
    return (error_percent_threshold)
    


# In[14]:


'''
This function returns the maximum absolute error of the model, percent of the time
'''

def evaluate_model_percent_mean(model, X_test, Y_test, percent):
    Y_predict = model.predict(X_test)
    
    Y_AE = np.abs(Y_predict- Y_test)
    Y_AE_best = Y_AE[Y_AE <= Y_AE.quantile(percent)] # Take percent best error values (eliminate errors > Y_AE.quantile(percent))
    
    error_mean = Y_AE_best.mean()
    
    return (error_mean)

'''
This function returns the maximum absolute error of the model, 1 - percent of the time
'''

def evaluate_model_percent_worst_mean(model, X_test, Y_test, percent):
    Y_predict = model.predict(X_test)
    
    Y_AE = np.abs(Y_predict- Y_test)
    Y_AE_worst = Y_AE[Y_AE > Y_AE.quantile(percent)] # Take percent worst error values (eliminate errors > Y_AE.quantile(percent))
    
    error_mean_worst = Y_AE_worst.mean()
    
    return (error_mean_worst)
    


# In[15]:


def minibatch_generate_indexes(df_train_transformed, step_size):
    nb_instances = df_train_transformed.shape[0]
    final_index = nb_instances - 1

    for m in range(int(nb_instances/step_size)):
        left_index = m*step_size
        right_index = m*step_size + step_size - 1

        yield((left_index, right_index))

    # Last step :
    yield((left_index + step_size, final_index))


# In[16]:


def plot_learning_curves(model, X_train, X_test, y_train, y_test, step_size, evaluation_method='RMSE'):
    train_errors, val_errors = [], []
    
    minibatch_indexes = minibatch_generate_indexes(X_train, step_size)
    
    # Initiate progress bar
    #nb_instances = len(df_train_transformed)
    nb_instances = df_train_transformed.shape[0]
    nb_iter = int(nb_instances/step_size) + 1    
    progbar = tqdm(range(nb_iter))
    #cnt = 0
    print(f'Calculating learning curve for {nb_iter} iterations')
    
    for (left_index, right_index) in minibatch_indexes:
        model.fit(X_train[:right_index], y_train[:right_index])
        y_train_predict = model.predict(X_train[:right_index])
        y_test_predict = model.predict(X_test)
        
        if (evaluation_method == 'RMSE'):
            train_errors.append(mean_squared_error(y_train[:right_index], y_train_predict))
            val_errors.append(mean_squared_error(y_test, y_test_predict))
            
        elif (evaluation_method == 'MAE'):
            train_errors.append(mean_absolute_error(y_train[:right_index], y_train_predict))
            val_errors.append(mean_absolute_error(y_test, y_test_predict))            
        
        # Update progress bar
        progbar.update(1)
        #cnt += 1

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="test")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set iterations", fontsize=14) # not shown
    
    if (evaluation_method == 'RMSE'):
        plt.ylabel("RMSE", fontsize=14)              # not shown
        
    elif (evaluation_method == 'MAE'):
         plt.ylabel("MAE", fontsize=14)  


# In[17]:


#minibatches = minibatch_generate_indexes(df_train_transformed)


# In[18]:


def reset_data():
    df = load_data()
    all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)
    df, df_train, df_test = custom_train_test_split_sample(df)

    df_train_transformed = preparation_pipeline_meansort.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=None)
    df_train_transformed = prediction_pipeline_without_sparse.fit_transform(df_train_transformed)

    df_test_transformed = preparation_pipeline_meansort.transform(df_test)
    df_test_transformed = prediction_pipeline_without_sparse.transform(df_test_transformed)
    
    return df, df_train, df_test, df_train_transformed, df_test_transformed

def reset_data_old():
    df = load_data()
    all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)
    df, df_train, df_test = custom_train_test_split_sample(df)

    df_train_transformed = preparation_pipeline_meansort.fit_transform(df_train)
    df_train_transformed = prediction_pipeline_without_sparse.fit_transform(df_train_transformed)

    df_test_transformed = preparation_pipeline_meansort.transform(df_test)
    df_test_transformed = prediction_pipeline_without_sparse.transform(df_test_transformed)
    df_test_transformed.shape
    
    return df, df_train, df_test, df_train_transformed, df_test_transformed


# In[19]:


from IPython.display import display, Markdown
import sys

def display_freq_table(df, col_names):
    for col_name in col_names:    
        effectifs = df[col_name].value_counts(bins=50)

        modalites = effectifs.index # l'index de effectifs contient les modalités


        tab = pd.DataFrame(modalites, columns = [col_name]) # création du tableau à partir des modalités
        tab["Nombre"] = effectifs.values
        tab["Frequence"] = tab["Nombre"] / len(df) # len(data) renvoie la taille de l'échantillon
        tab = tab.sort_values(col_name) # tri des valeurs de la variable X (croissant)
        tab["Freq. cumul"] = tab["Frequence"].cumsum() # cumsum calcule la somme cumulée
        
        display(Markdown('#### ' + col_name))
        display(tab)


# In[20]:


class redirect_output(object):
    """context manager for reditrecting stdout/err to files"""
    
    """ 
    Useful to run long code, in order not to loose cell output if you close browser by mistake
    
    Usage in a cell :
    with redirect_output("my_output.txt"):
        Long code  (example : %run my_script.py )
        
    """

    def __init__(self, stdout='', stderr=''):
        self.stdout = stdout
        self.stderr = stderr

    def __enter__(self):
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr

        if self.stdout:
            sys.stdout = open(self.stdout, 'w')
        if self.stderr:
            if self.stderr == self.stdout:
                sys.stderr = sys.stdout
            else:
                sys.stderr = open(self.stderr, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr


# # First naive model

# In[21]:


df = load_data_with_outliers()
df, df_train, df_test = custom_train_test_split_sample_random(df)
all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)


# In[27]:


from sklearn import dummy

dum = dummy.DummyRegressor(strategy='mean')

# Entraînement
dum.fit(df_train, df_train[model1_label])

# Prédiction sur le jeu de test
y_pred_dum = dum.predict(df_test)

# Evaluate
print("RMSE : {:.2f}".format(np.sqrt(mean_squared_error(df_test[model1_label], y_pred_dum)) ))
print("MAE : {:.2f}".format(mean_absolute_error(df_test[model1_label], y_pred_dum) ))

error_mean = evaluate_model_percent_mean(dum, df_test, df_test[model1_label], 0.9)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')


# In[23]:


del df, df_train, df_test


# # Data load

# In[24]:


df = load_data()


# In[25]:


df.shape


# In[26]:


display_percent_complete(df)


# In[27]:


'''
for column_name in df.columns:
    print_column_information(df, column_name)
    
'''


# # Identification of features

# In[28]:


# Below are feature from dataset that we decided to keep: 
'''
all_features = ['ORIGIN','CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','DEST','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME','ARR_DELAY','DEP_DELAY', 'TAXI_OUT', 'TAIL_NUM']

model1_features = ['ORIGIN','CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','DEST','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME']
model1_label = 'ARR_DELAY'
'''

all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)


# # Split train set, test set

# In[29]:


df, df_train, df_test = custom_train_test_split_sample(df)

'''
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
df_train = df_train.copy()
df_test = df_test.copy()

if (SAMPLED_DATA == True):
    df_train = df_train.sample(NB_SAMPLES).copy(deep=True)
    df = df.loc[df_train.index]
'''


# In[30]:


df_train


# In[31]:


df_train[['ARR_DELAY', 'UNIQUE_CARRIER']].groupby('UNIQUE_CARRIER').mean().sort_values(by='ARR_DELAY', ascending=True)


# In[32]:


df_train[['ARR_DELAY', 'UNIQUE_CARRIER']].groupby('UNIQUE_CARRIER').mean().sort_values(by='ARR_DELAY', ascending=True).plot()


# # Features encoding

# In[21]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn import decomposition
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import statistics

from scipy import sparse

'''
Cette fonction fait un 1 hot encoding des features qui sont des catégories
Old function not used anymore
'''
    
def add_categorical_features_1hot(df, categorical_features_totransform):
    #df.drop(labels=categorical_features_totransform, axis=1, inplace=True)
    
    
    #df_encoded = pd.get_dummies(df, columns=categorical_features_totransform, sparse=True)
    
    for feature_totransform in categorical_features_totransform:
        print(f'Adding 1hot Feature : {feature_totransform}')
        
        print('First')
        df_transformed = df[feature_totransform].str.get_dummies().add_prefix(feature_totransform +'_')   
        
        #df_new = pd.get_dummies(df, columns=['ORIGIN'])
        
        
        
        
        #df.drop(labels=feature_totransform, axis=1, inplace=True)
        print('Second')
        del df[feature_totransform]
        
        print('Third')
        df = pd.concat([df, df_transformed], axis=1)
        
    return(df)


class HHMM_to_Minutes(BaseEstimator, TransformerMixin):
    def __init__(self, features_toconvert = ['CRS_DEP_TIME', 'CRS_ARR_TIME']):
        self.features_toconvert = features_toconvert
        return None
    
    def fit(self, df):      
        return self
    
    def transform(self, df):       
        for feature_toconvert in self.features_toconvert:
            print(f'Converting feature {feature_toconvert}\n')
            #print('1\n')
            df_concat = pd.concat([df[feature_toconvert].str.slice(start=0,stop=2, step=1),df[feature_toconvert].str.slice(start=2,stop=4, step=1)], axis=1).astype(int).copy(deep=True)
                    
            #print('2\n')
            df[feature_toconvert] = (df_concat.iloc[:, [0]] * 60 + df_concat.iloc[:, [1]])[feature_toconvert]
            del df_concat
            
            #print('3\n')
        
        return(df)

    
class HHMM_to_HH(BaseEstimator, TransformerMixin):
    def __init__(self, features_toconvert = ['CRS_DEP_TIME', 'CRS_ARR_TIME']):
        self.features_toconvert = features_toconvert
        return None
    
    def fit(self, df):      
        return self
    
    def transform(self, df):       
        for feature_toconvert in self.features_toconvert:
            print(f'Converting feature {feature_toconvert}\n')
            #print('1\n')

            df.loc[:, feature_toconvert] = df[feature_toconvert].str.slice(start=0,stop=2, step=1)
        
        return(df)
    
'''
class CategoricalFeatures1HotEncoder_old(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features_totransform=['ORIGIN', 'UNIQUE_CARRIER', 'DEST']):
        self.categorical_features_totransform = categorical_features_totransform
    
    def fit(self, df, labels=None):      
        return self
    
    def transform(self, df):
        # /!\ Array will not have the same shape if we fit an ensemble of samples that have less values than total dataset
        df_encoded = pd.get_dummies(df, columns=self.categorical_features_totransform, sparse=True)  # Sparse allows to gain memory. But then, standardscale must be with_mean=False
        #df_encoded = pd.get_dummies(df, columns=self.categorical_features_totransform, sparse=False)
        print('type of df : ' + str(type(df_encoded)))
        return(df_encoded)
'''

class CategoricalFeatures1HotEncoder(BaseEstimator, TransformerMixin):
    #def __init__(self, categorical_features_totransform=['ORIGIN', 'UNIQUE_CARRIER', 'DEST']):
    def __init__(self):
        #self.categorical_features_totransform = categorical_features_totransform
        self.fitted = False
        self.all_feature_values = {}
        #self.df_encoded = None
    
    #def fit(self, df, labels=None):      
    def fit(self, df, categorical_features_totransform=['ORIGIN', 'UNIQUE_CARRIER', 'DEST']):      
        print('Fit data')
        self.categorical_features_totransform = categorical_features_totransform
        print('!! categorical_features_totransform' + str(self.categorical_features_totransform))

        if (self.categorical_features_totransform != None):
            for feature_name in self.categorical_features_totransform:
                df[feature_name] = df[feature_name].astype(str) # Convert features to str in case they are not already     
                self.all_feature_values[feature_name] = feature_name + '_' + df[feature_name].unique()
        
        self.fitted = True
        
        return self
    
    def transform(self, df):
        if (self.fitted == False):
            self.fit(df)
        
        if (self.categorical_features_totransform != None):
            print('Transform data')
            
            print('1hot encode categorical features...')
            #df_encoded = pd.get_dummies(df, columns=self.categorical_features_totransform, sparse=True)  # Sparse allows to gain memory. But then, standardscale must be with_mean=False
            df_encoded = pd.get_dummies(df, columns=self.categorical_features_totransform, sparse=False)

            # Get category values that were in fitted data, but that are not in data to transform 
            for feature_name, feature_values in self.all_feature_values.items():
                diff_columns = list(set(feature_values) - set(df_encoded.columns.tolist()))
                print(f'Column values that were in fitted data but not in current data: {diff_columns}')

                if (len(diff_columns) > 0):
                    print('Adding those column with 0 values to the DataFrme...')
                    # Create columns with 0 for the above categories, in order to preserve same matrix shape between train et test set
                    zeros_dict = dict.fromkeys(diff_columns, 0)
                    df_encoded = df_encoded.assign(**zeros_dict)

            print('type of df : ' + str(type(df_encoded)))
            return(df_encoded)

        else:
            return(df)

class Aggregate_then_GroupByMean_then_Sort_numericalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features_totransform=['ORIGIN', 'UNIQUE_CARRIER', 'DEST']):
        self.categorical_features_totransform = categorical_features_totransform
        self.fitted = False
        self.all_feature_values = {}
    
    def fit(self, df, labels=None):      
        print('Fit data')
        
        self.feature_maps = {}
        
        for feature_name in self.categorical_features_totransform:
            print(f'Fitting feature {feature_name}')
            # List all feature values ordered by mean delay
            list_feature_mean_ordered = df[['ARR_DELAY', feature_name]].groupby(feature_name).mean().sort_values(by='ARR_DELAY', ascending=True).index.tolist()
            
            # Generate a dictionary of feature values as keys and index as values
            self.feature_maps[feature_name] = {}
            self.feature_maps[feature_name]['list_feature_mean_ordered_dict'] = {list_feature_mean_ordered[i] : i for i in range(len(list_feature_mean_ordered))  }
            
            #print('Dictionnary : ' + str(self.feature_maps[feature_name]['list_feature_mean_ordered_dict']))
            
            # BUG : we had to do that line of code in transform instead, not in fit (result is the same, no difference)
            #self.feature_maps[feature_name]['list_feature_mean_ordered_mapper'] = lambda k : self.feature_maps[feature_name]['list_feature_mean_ordered_dict'][k]
                  
        self.fitted = True
        
        return self
    
    def transform(self, df):
        if (self.fitted == False):
            print('Launching fit first (if you see this message : ensure that you have passed training set as input, not test set)')
            self.fit(df)
        
        print('Encode categorical features...')
        
        for feature_name in self.categorical_features_totransform:
            print(f'Encoding feature {feature_name} ...')
            #print('Dictionnary : ' + str(self.feature_maps[feature_name]['list_feature_mean_ordered_dict']))
            
            # Replace each feature value by its index (the lowest the index, the lowest the mean delay is for this feature)
            list_feature_mean_ordered_mapper = lambda k : self.feature_maps[feature_name]['list_feature_mean_ordered_dict'][k]
                            
            #df[feature_name] = df.loc[:, feature_name].apply(self.feature_maps[feature_name]['list_feature_mean_ordered_mapper'])  # BUG (we had to use line below instead)
            df[feature_name] = df.loc[:, feature_name].apply(list_feature_mean_ordered_mapper)

        return(df)
    
    
class FeaturesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features_toselect = None):  # If None : every column is kept, nothing is done
        self.features_toselect = features_toselect
    
    def fit(self, df, labels=None):      
        return self
    
    def transform(self, df):       
        if (self.features_toselect != None):
            filter_cols = [col for col in df if (col.startswith(tuple(self.features_toselect)))]
            
            filter_cols.sort()
            
            print("Features selected (in order): " + str(df[filter_cols].columns))
            
            df = df.loc[:, filter_cols]
            return(df)

        else:
            return(df)

'''
In order have less features globally: we Keep only features_tofilter that represent percent_tokeep% of total values
Features which values represent less than percent_tokeep% will be set "OTHERS" value instead of their real value
'''

class Filter_High_Percentile(BaseEstimator, TransformerMixin):
    def __init__(self, features_tofilter = ['ORIGIN', 'DEST'], percent_tokeep = 80):
        self.features_tofilter = features_tofilter
        self.percent_tokeep = percent_tokeep
        self.high_percentile = None
        self.low_percentile = None
    
    def fit(self, df, labels=None): 
        print('Fit high percentile filter...')
        for feature_tofilter in self.features_tofilter:
            # Get feature_tofilter values that represent 80% of data
            self.high_percentile = ((((df[[feature_tofilter]].groupby(feature_tofilter).size() / len(df)).sort_values(ascending=False)) * 100).cumsum() < self.percent_tokeep).where(lambda x : x == True).dropna().index.values.tolist()
            self.low_percentile = ((((df[[feature_tofilter]].groupby(feature_tofilter).size() / len(df)).sort_values(ascending=False)) * 100).cumsum() >= self.percent_tokeep).where(lambda x : x == True).dropna().index.values.tolist()

            total = len(df[feature_tofilter].unique())
            high_percentile_sum = len(self.high_percentile)
            low_percentile_sum = len(self.low_percentile)
            high_low_sum = high_percentile_sum + low_percentile_sum

            print(f'Total number of {feature_tofilter} values : {total}')
            print(f'Number of {feature_tofilter} high percentile (> {self.percent_tokeep}%) values : {high_percentile_sum}')
            print(f'Number of {feature_tofilter} low percentile values : {low_percentile_sum}')
            print(f'Sum of high percentile + low percentile values : {high_low_sum}')
        
        print('End of high percentile filter fit')
        return self
    
    def transform(self, df):       
        if (self.features_tofilter != None):
            print('Apply high percentile filter...')
            
            for feature_tofilter in self.features_tofilter:
                print(f'Apply filter on feature {feature_tofilter}')
                # To do for later : apply low_percentile specific to the feature, and not only last calculated low_percentile  (in our case it's the same percentile for ORIGIN and DEST so this is not a problem)
                df.loc[df[feature_tofilter].isin(self.low_percentile), feature_tofilter] = 'OTHERS'   
            
            return(df)    

        else:
            return(df)
        

    
class DenseToSparseConverter(BaseEstimator, TransformerMixin):
    def __init__(self):  # If None : every column is kept, nothing is done
        return None
    
    def fit(self, matrix, labels=None):      
        return self
    
    def transform(self, matrix):   
        return(sparse.csr_matrix(matrix))

    
'''
This class adds polynomial features in univariate way  (if feature X and n_degree 3 :  then it will add X², X³, and an intercept at the end)

Requires ndarray as input
'''    
class PolynomialFeaturesUnivariateAdder(BaseEstimator, TransformerMixin):
    def __init__(self, n_degrees=2):
        self.n_degrees = n_degrees
        self.fitted = False
    
    def fit(self, df, labels=None):
        self.fitted = True
        return self
    
    def transform(self, df):
        if (self.fitted == False):
            self.fit(df)

        nb_instances, n_features = df.shape
        df_poly = np.empty((nb_instances, 0)) # Create empty array of nb_instances line and 0 features yet (we'll concatenate polynomial features to it)

        progbar = tqdm(range(n_features))
        print('Adding polynomial features')

        for feature_index in range(n_features):    
            df_1feature = df[:,feature_index]  # Reshape 

            for n_degree in range(self.n_degrees):
                df_poly = np.c_[df_poly, np.power(df_1feature, n_degree + 1)]

            progbar.update(1)

        # Add bias (intercept)
        df_poly = np.c_[df_poly, np.ones((len(df_poly), 1))]  # add x0 = 1 feature        
        
        return(df_poly)

'''
This class adds polynomial features in univariate way  (if feature X and n_degree 3 :  then it will add X², X³, and an intercept at the end)

Requires DataFrame as input
'''        
    
class PolynomialFeaturesUnivariateAdder_DataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, n_degrees=2):
        self.n_degrees = n_degrees
        self.fitted = False
    
    def fit(self, df, features_toadd=None):  
        print('fit')
        self.features_toadd = features_toadd
        print('Features to add :')
        print(self.features_toadd)
        self.fitted = True
        return self
    
    def transform(self, df):
        print('transform')
        if (self.fitted == False):
            self.fit(df)

        nb_instances, n_features = df.shape
        #df_poly = np.empty((nb_instances, 0)) # Create empty array of nb_instances line and 0 features yet (we'll concatenate polynomial features to it)
        #df_poly = pd.DataFrame(index=df.index,columns=None)
        
        progbar = tqdm(range(len(self.features_toadd)))
        print('Adding polynomial features')

        for column_name in self.features_toadd:    
            #df_1feature = df.loc[:, column_name]

            for n_degree in range(1, self.n_degrees):
                #df = pd.concat([df, np.power(df.loc[:, column_name], n_degree + 1)], axis=1)
                df = pd.concat([df, np.power(df[[column_name]], n_degree + 1).rename(columns={column_name : column_name+'_DEG'+str(n_degree+1)})], axis=1)

            progbar.update(1)

        # Add bias (intercept)
        #df_poly = pd.concat([df_poly, np.ones((len(df_poly), 1))])  # add x0 = 1 feature        
        #del df
        return(df)    
    
    
class StandardScalerMultiple_old(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fitted = False
    
    def fit(self, df, columns=None):              
        self.columns = columns
        self.scaler = StandardScaler()
  
        if (self.columns == None):
            return(df)
        else:
            self.scaler.fit(df[self.columns].to_numpy())            
        
        return self
    
    def transform(self, df):
        if (self.fitted == False):
            self.fit(df)
        
        if (self.columns == None):
            return(df)
        
        else:
            df[self.columns] = self.scaler.transform(df[self.columns].to_numpy())

        return(df)
        
class StandardScalerMultiple(BaseEstimator, TransformerMixin):
    def __init__(self, features_toscale=None):
        self.fitted = False
        self.columns = features_toscale
    
    def fit(self, df):              
        print('Fit Std scale multiple')
        self.scaler = StandardScaler()
  
        if (self.columns == None):
            self.fitted = True
            return(df)
        else:
            self.scaler.fit(df[self.columns].to_numpy())            
            self.fitted = True
        
        return self
    
    def transform(self, df):
        print('Transform Std scale multiple')
        if (self.fitted == False):
            self.fit(df)
        
        if (self.columns == None):
            return(df)
        
        else:
            df.loc[:, self.columns] = self.scaler.transform(df.loc[:, self.columns].to_numpy())

        return(df)
        

class MinMaxScalerMultiple(BaseEstimator, TransformerMixin):
    def __init__(self, features_toscale=None):
        self.fitted = False
        self.columns = features_toscale
    
    def fit(self, df):              
        print('Fit Min max scaler multiple')
        self.scaler = MinMaxScaler()
  
        if (self.columns == None):
            self.fitted = True
            return(df)
        else:
            self.scaler.fit(df[self.columns].to_numpy())            
            self.fitted = True
        
        return self
    
    def transform(self, df):
        print('Transform Min max scaler multiple')
        if (self.fitted == False):
            self.fit(df)
        
        if (self.columns == None):
            return(df)
        
        else:
            df.loc[:, self.columns] = self.scaler.transform(df.loc[:, self.columns].to_numpy())

        return(df)        
        
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features_totransform = None):  # If None : every column is kept, nothing is done
        self.features_toselect = features_toselect
    
    def fit(self, df, labels=None):      
        return self
    
    def transform(self, df):       
        if (self.features_toselect != None):
            filter_cols = [col for col in df if (col.startswith(tuple(self.features_toselect)))]
            
            filter_cols.sort()
            
            print("Features selected (in order): " + str(df[filter_cols].columns))
            
            df = df.loc[:, filter_cols]
            return(df)

        else:
            return(df)
 
        
        
'''
conversion_pipeline = Pipeline([
    ('data_converter', HHMM_to_Minutes()),
    #('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()),
    #('standardscaler', preprocessing.StandardScaler()),
])
'''

preparation_pipeline = Pipeline([
    ('filter_highpercentile', Filter_High_Percentile()),
    ('data_converter', HHMM_to_Minutes()),
    ('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()),
    #('standardscaler', preprocessing.StandardScaler()),
])


'''
preparation_pipeline_meansort = Pipeline([
    #('filter_highpercentile', Filter_High_Percentile()),
    ('data_converter', HHMM_to_Minutes()),
    ('numericalEncoder', Aggregate_then_GroupByMean_then_Sort_numericalEncoder()),
    #('standardscaler', preprocessing.StandardScaler()),
])
'''

preparation_pipeline_meansort = Pipeline([
    #('filter_highpercentile', Filter_High_Percentile()),
    ('data_converter', HHMM_to_Minutes()),
    ('numericalEncoder', Aggregate_then_GroupByMean_then_Sort_numericalEncoder()),
    ('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()),
    #('standardscaler', preprocessing.StandardScaler()),
])


# If matrix is sparse, with_mean=False must be passed to StandardScaler
prediction_pipeline = Pipeline([
    ('features_selector', FeaturesSelector(features_toselect=MODEL1_FEATURES)),
    ('standardscaler', ColumnTransformer([
        ('standardscaler_specific', StandardScaler(), MODEL1_FEATURES_QUANTITATIVE)
    ], remainder='passthrough', sparse_threshold=1)),
    
    ('dense_to_sparse_converter', DenseToSparseConverter()),
    #('predictor', To_Complete(predictor_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
])
#copy=False passed to StandardScaler() allows to gain memory


prediction_pipeline_without_sparse = Pipeline([
    ('features_selector', FeaturesSelector(features_toselect=MODEL1_FEATURES)),
    ('standardscaler', ColumnTransformer([
        ('standardscaler_specific', StandardScaler(), MODEL1_FEATURES_QUANTITATIVE)
    #], remainder='passthrough', sparse_threshold=1)), # For sparse output. Seems not to work well.
    ], remainder='passthrough')),
    
    #('dense_to_sparse_converter', DenseToSparseConverter()),
    #('predictor', To_Complete(predictor_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
])

prediction_pipeline_groupbymean = Pipeline([
    ('features_selector', FeaturesSelector(features_toselect=MODEL1_GOUPBYMEAN_FEATURES)),
    ('standardscaler', ColumnTransformer([
        ('standardscaler_specific', StandardScaler(), MODEL_GROUPBYMEAN_FEATURES_QUANTITATIVE)
    ], remainder='passthrough')),
    #], remainder='passthrough', sparse_threshold=1)), # For sparse output. Seems not to work well.
    
    #('dense_to_sparse_converter', DenseToSparseConverter()),
    #('predictor', To_Complete(predictor_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
])



prediction_pipeline_1hotall_without_sparse = Pipeline([
    ('features_selector', FeaturesSelector(features_toselect=MODEL1_FEATURES)),
    ('standardscaler', ColumnTransformer([
        ('standardscaler_specific', StandardScaler(), MODEL1bis_FEATURES_QUANTITATIVE)
    ], remainder='passthrough', sparse_threshold=1)),
    
    #('dense_to_sparse_converter', DenseToSparseConverter()),
    #('predictor', To_Complete(predictor_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
])

prediction_pipeline_cheat_without_sparse = Pipeline([
    ('features_selector', FeaturesSelector(features_toselect=MODEL_cheat_FEATURES)),
    ('standardscaler', ColumnTransformer([
        ('standardscaler_specific', StandardScaler(), MODEL_cheat_FEATURES_QUANTITATIVE)
    ], remainder='passthrough', sparse_threshold=1)),
    
    #('dense_to_sparse_converter', DenseToSparseConverter()),
    #('predictor', To_Complete(predictor_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
])



preparation_pipeline_meansort_stdscale = Pipeline([
    ('data_converter', HHMM_to_Minutes()),
    ('numericalEncoder', Aggregate_then_GroupByMean_then_Sort_numericalEncoder()),
    #('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()),
    ('features_selector', FeaturesSelector(features_toselect=MODEL1_GOUPBYMEAN_FEATURES)),
    ('standardscaler', StandardScalerMultiple(features_toscale=MODEL1_GOUPBYMEAN_FEATURES)),
])


# Temporary modification :  HHMM_To_Minutes instead of HHMM_to_HH and CRS_DEP_TIME in quantitative MODEL_1HOTALL_FEATURES_QUANTITATIVE

# To defined features to 1hot encode, pass fit_transform parameter below:  
# categoricalfeatures_1hotencoder__categorical_features_totransform=['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN', 'UNIQUE_CARRIER', 'CRS_DEP_TIME']
preparation_pipeline_1hotall_minmax = Pipeline([
    ('filter_highpercentile', Filter_High_Percentile()),
    ('hour_extractor', HHMM_to_HH()),
    #('data_converter', HHMM_to_Minutes()),
    ('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()), 
    
    ('features_selector', FeaturesSelector(features_toselect=MODEL_1HOTALL_FEATURES)),
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=MODEL_1HOTALL_FEATURES_QUANTITATIVE)),
])


preparation_pipeline_meansort2_stdscale = Pipeline([
    ('filter_highpercentile', Filter_High_Percentile()),
    ('hour_extractor', HHMM_to_HH()),
    ('numericalEncoder', Aggregate_then_GroupByMean_then_Sort_numericalEncoder()),
    #('data_converter', HHMM_to_Minutes()),
    #('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()), 
    
    ('features_selector', FeaturesSelector(features_toselect=MODEL_GROUPBYMEAN2_FEATURES)),
    ('standardscaler', StandardScalerMultiple(features_toscale=MODEL_GROUPBYMEAN2_FEATURES)),
])


preparation_pipeline_2feats_stdscale = Pipeline([
    #('filter_highpercentile', Filter_High_Percentile()),
    #('hour_extractor', HHMM_to_HH()),
    #('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()), 
    
    ('features_selector', FeaturesSelector(features_toselect=MODEL1_2FEATS)),
    #('standardscaler', StandardScalerMultiple(features_toscale=MODEL1_2FEATS_QUANTITATIVE)),
])

preparation_pipeline_3feats_stdscale = Pipeline([
    #('filter_highpercentile', Filter_High_Percentile()),
    #('hour_extractor', HHMM_to_HH()),
    #('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()), 
    
    ('features_selector', FeaturesSelector(features_toselect=MODEL1_3FEATS)),
    #('standardscaler', StandardScalerMultiple(features_toscale=MODEL1_2FEATS_QUANTITATIVE)),
])



'''
# Old code that used scikit learn OneHotEncoder (which does not keep DataFrame type) instead of Pandas
preparation_pipeline2 = Pipeline([
    ('data_converter', HHMM_to_Minutes()),
    ('multiple_encoder', ColumnTransformer([
        ('categoricalfeatures_1hotencoder', OneHotEncoder(), ['ORIGIN', 'UNIQUE_CARRIER', 'DEST'])
    ], remainder='passthrough')),
    #('standardscaler', preprocessing.StandardScaler()),
])
'''

'''
ColumnTransformer([
        ('standardscaler_specific', StandardScaler(), ['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DISTANCE', 'CRS_ELAPSED_TIME', 'ARR_DELAY', 'DEP_DELAY', 'TAXI_OUT'])
    ], remainder='passthrough')
'''


# In[34]:


df


# In[35]:


df_train_transformed = preparation_pipeline.fit_transform(df_train)


# In[36]:


df_train_transformed


# In[37]:


df_train_transformed.shape


# In[38]:


df_train_transformed.info()


# In[39]:


#df_train_transformed = prediction_pipeline.fit_transform(df_train_transformed)  # Used if standard scale not commented out
df_train_transformed = prediction_pipeline_without_sparse.fit_transform(df_train_transformed)


# In[40]:


df_train_transformed.shape


# In[41]:


from scipy import sparse
sparse.issparse(df_train_transformed)


# In[42]:


#pd.DataFrame.sparse.from_spmatrix(df_train_transformed)


# In[43]:


pd.set_option('display.max_columns', 400)


# In[44]:


all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)


# # Test set encoding

# In[45]:


df_test_transformed = preparation_pipeline.transform(df_test)
#df_test_transformed = prediction_pipeline.transform(df_test_transformed)  # Used if standardscale not commented out
df_test_transformed = prediction_pipeline_without_sparse.transform(df_test_transformed)
DATA_LOADED = True
df_test_transformed.shape


# In[46]:


df_train_transformed


# In[47]:


df_test[model1_label]


# In[48]:


df_test.index


# In[49]:


df.loc[df_test.index, model1_label]


# # Linear regression

# In[50]:


df_train[model1_label].shape


# In[51]:


# Add bias :
df_train_transformed = np.c_[np.ones((len(df_train_transformed), 1)), df_train_transformed]  # add x0 = 1 to each instance
df_test_transformed = np.c_[np.ones((len(df_test_transformed), 1)), df_test_transformed]  # add x0 = 1 to each instance


# In[52]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    lin_reg = LinearRegression(normalize=False, fit_intercept=True)
    #lin_reg = TransformedTargetRegressor(regressor=lin_reg, transformer=StandardScaler())  # To scale y variable
    lin_reg.fit(df_train_transformed, df_train[model1_label])


# In[53]:


'''
lin_reg = linear_model.SGDRegressor(alpha=0,max_iter=200)
lin_reg.fit(df_train_transformed, df_train[model1_label])
'''


# In[54]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_test_predictions = lin_reg.predict(df_test_transformed)
    
    #lin_mse = mean_squared_error(df_test[model1_label], df_test_predictions)
    #lin_rmse = np.sqrt(lin_mse)
    #print(lin_rmse)
    
    print("Evaluation on test set :")
    evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])

    print('\n')

    print("Evaluation on training set :")
    evaluate_model(lin_reg, df_train_transformed, df_train[model1_label])

    error_mean = evaluate_model_percent_mean(lin_reg, df_test_transformed, df_test[model1_label], 0.8)
    
    print('\n')
    print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')


# => 42.17  (42.16679389006135)  
# => 26.998703285049196  with outliers removed  
# => 26.998703285632104 with TransformedTargetRegressor  
# => 26.99870280932372 without standardscale  
# => 27.00905767522797 with SGDRegressor ((alpha=0,max_iter=200) and standarscale
# 
# 
# With 80000 lines :   
# Evaluation on test set :   
# RMSE : 27.03778899779597   
# 
# 
# Evaluation on training set :   
# RMSE : 26.990456780220008   
# Mean prediction error 90.0% of the time :  10.01   

# In[55]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train_predictions = lin_reg.predict(df_train_transformed)
    df_test_predictions = lin_reg.predict(df_test_transformed)


# In[56]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plt.hist(df_test_predictions, bins=50)


# In[57]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plot_learning_curves(lin_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE)


# In[58]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train


# In[59]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train[[model1_label]]


# In[60]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    lin_reg.coef_


# array([  3.0601343 ,   0.54106613,   0.49255945,  -0.02052923,
#          1.6354574 ,  10.62073489, -12.18551101,   0.27450354,
#         -1.33980503,   0.10149764,   0.31685657,  -0.65541914,
#         -1.53379643,  -0.56033462,   2.17482374,   0.23276129,
#         -1.26869889,   3.05226253,   0.18607054,   5.51988916,
#          1.42903568,  -4.32234874,   0.30318026,  -0.94593642,
#         -1.40645942,   4.27814774,  -0.06934268,   2.69507292,
#          4.87336364,  -1.91588117,   1.21284379,  -1.94274921,
#          4.73238555,  -3.20774291,   0.47564597,  -1.6987073 ,
#          0.14379464,   4.40476573,  -1.74618153,  -2.76805405,
#          0.45234394,  -1.60170948,   1.1685909 ,  -2.46122663,
#         -1.75186943,  -0.42718454,   3.2389558 ,  -0.80286191,
#         -3.11232599,  -0.71881518,  -3.40806787,  -1.02946792,
#         -0.57180507,   0.58845347,  -5.32955616,   4.14911968,
#         -3.77501356,   1.94734073,   5.04213062,  -3.46294331,
#          4.36932467,   1.10828485,  -3.07715486,  -2.1540515 ,
#          0.59406537,  -0.21559221,  -1.4864192 ,  -1.92685958,
#         -1.13360134,  -2.79307414,  -3.16535316,  -2.14418489,
#         -0.3786179 ,  -1.35651547,  -0.65089003,   3.72717084,
#         -0.5316482 ,   6.38411318,   0.56280763,   2.32285819,
#         -0.71875407,  -3.7427498 ,  -0.72443612,   3.01913525,
#          0.0259135 ,   6.05048444,   6.88297842,  -2.05348981,
#          0.57619293,  -3.06150148,   2.91489312,  -3.69821352,
#          1.39265583,  -2.73996445,   0.48819288,   3.78111808,
#         -2.73090349,  -0.99799275,  -1.41459568,  -0.14998697,
#         -1.16199228,  -1.2855059 ,  -2.33902093,   2.26562809,
#          9.46369559,  -0.27437885,  -1.82963879,   0.47213692,
#         -2.5256052 ,  -2.053249  ,  -1.04523965])

# In[61]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_test[model1_label], df_test_predictions, color='coral', alpha=0.1)


# In[62]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train_transformed


# In[63]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plt.hist(df_test_predictions, bins=50)


# In[64]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plt.hist(df_test[model1_label], bins=50)


# In[65]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train_predictions = lin_reg.predict(df_train_transformed)


# In[66]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plt.hist(df_train_predictions, bins=50)


# In[67]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    from sklearn.model_selection import cross_validate

    #scores = cross_validate(lin_reg, df_train_transformed, df_train[model1_label], scoring='neg_root_mean_squared_error', cv=5)


# In[68]:


#scores['test_score'].mean()


# # ElasticNET regression

# In[69]:


from sklearn.model_selection import ShuffleSplit


shuffled_split_train = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)


# In[70]:


from sklearn.linear_model import ElasticNet


# In[71]:


from sklearn.model_selection import GridSearchCV

eNet = ElasticNet()

grid_search = GridSearchCV(eNet, param_grid = {"max_iter": [1, 5, 10],
                      "alpha": [10, 100],
                      "l1_ratio": np.arange(0.0, 1.0, 0.4)},cv=shuffled_split_train, scoring='neg_mean_squared_error', error_score=np.nan, verbose=2)


# In[72]:


'''
from sklearn.model_selection import GridSearchCV

eNet = ElasticNet()

grid_search = GridSearchCV(eNet, param_grid = {"max_iter": [1, 5, 10],
                      "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      "l1_ratio": np.arange(0.0, 1.0, 0.1)},cv=shuffled_split_train, scoring='neg_mean_squared_error', error_score=np.nan, verbose=2)
'''


# In[73]:


if (RECOMPUTE_GRIDSEARCH == True):
    grid_search.fit(df_train_transformed, df_train[model1_label])


# In[74]:


if ((SAVE_GRID_RESULTS == False) and (LOAD_GRID_RESULTS == True)):
    grid_search = None
    
grid_search, df_grid_search_results = save_or_load_search_params(grid_search, 'eNet_20200319')


# In[75]:


if ((EXECUTE_INTERMEDIATE_MODELS == True) and (LOAD_GRID_RESULTS == True)):
    df_grid_search_results.sort_values(by='mean_test_score', ascending=False)


# In[76]:


np.sqrt(1741.47)


# => 41.73092378560532

# In[77]:


if ((EXECUTE_INTERMEDIATE_MODELS == True) and (LOAD_GRID_RESULTS == True)):
    grid_search.best_estimator_


# In[78]:


if ((EXECUTE_INTERMEDIATE_MODELS == True) and (LOAD_GRID_RESULTS == True)):
    df_test_predictions = grid_search.best_estimator_.predict(df_test_transformed)
    mse = mean_squared_error(df_test[model1_label], df_test_predictions)
    rmse = np.sqrt(mse)
    print(rmse)


# In[79]:


if ((EXECUTE_INTERMEDIATE_MODELS == True) and (LOAD_GRID_RESULTS == True)):
    grid_search.best_estimator_.coef_


# array([ 3.81842241e-01,  5.38463257e-02,  4.20287224e-02,  5.22019019e-04,
#         3.01660130e-01, -5.81861174e-02, -6.72278997e-02, -1.27034969e-02,
#        -2.24325259e-04,  4.09514301e-04, -3.70631003e-03, -2.53436510e-03,
#        -7.75208542e-04,  9.33417957e-05,  4.17993863e-03, -9.42512871e-04,
#        -1.11806550e-03,  1.44798574e-02, -3.13856400e-03,  3.29966151e-03,
#         3.99529584e-03, -4.48915797e-03,  1.95911581e-03, -2.36326646e-03,
#        -4.08853652e-03,  1.66741438e-03,  5.08260117e-03,  1.27353349e-02,
#         9.28815390e-04, -1.23422335e-03,  3.00267997e-03, -1.85353662e-03,
#         6.23418838e-03, -1.61411894e-03, -2.91269348e-03, -6.75664468e-04,
#         1.69775346e-03,  1.63447325e-02, -2.33087459e-02, -2.59028581e-03,
#        -1.74542702e-03, -1.45041347e-04,  6.16849621e-05, -9.03449295e-04,
#        -5.89810820e-04, -5.62947068e-03,  9.74750809e-03,  2.35807415e-04,
#        -6.16562051e-03,  9.79361758e-04, -9.11503285e-04,  2.40351790e-04,
#        -8.00703606e-04,  2.80040048e-02, -1.62362534e-02,  2.32015532e-02,
#        -7.01852245e-02,  1.65919140e-02,  7.25363024e-03, -4.83051956e-03,
#         1.41858822e-02,  6.85437437e-03, -1.58676117e-02,  4.82723839e-03,
#         6.33969777e-03, -2.46758789e-02,  2.42193209e-04, -1.45870754e-04,
#         4.15867207e-03, -2.83023087e-03,  7.62388260e-05, -4.24610507e-03,
#         5.26083293e-04,  1.47810804e-03, -9.19516190e-03,  8.25254437e-03,
#        -5.18481866e-03,  1.03654765e-02,  2.87040417e-03, -1.10418835e-03,
#         9.52434223e-05, -2.21408556e-03, -6.79991381e-03,  6.17574698e-03,
#        -8.83819104e-04,  1.25073748e-02,  1.17980381e-02, -6.81557286e-04,
#         3.04168241e-03, -4.21957706e-03,  3.05120532e-03, -7.89844324e-04,
#        -2.93426593e-03, -8.38652464e-04,  1.05935638e-03,  6.74906118e-03,
#        -5.14042339e-03, -2.37742304e-03, -4.46900931e-04, -2.72629361e-03,
#         9.62469273e-04, -1.57391539e-03, -1.61121536e-05, -5.49546918e-03,
#         1.91647269e-02, -4.83097086e-05, -6.78384591e-03,  1.00960720e-03,
#        -2.12687675e-03, -2.30717357e-04,  1.28182126e-04])

# In[80]:


from sklearn import metrics 
sorted(metrics.SCORERS.keys())


# ## Naive approach

# ### Random value between min and max

# In[81]:


y_pred_random = np.random.randint(df['ARR_DELAY'].min(), df['ARR_DELAY'].max(), df_test['ARR_DELAY'].shape)
naive_mse = mean_squared_error(df_test[model1_label], y_pred_random)
naive_rmse = np.sqrt(naive_mse)
naive_rmse


# ### Always mean naive approach

# In[82]:


from sklearn import dummy

dum = dummy.DummyRegressor(strategy='mean')

# Entraînement
dum.fit(df_train, df_train[model1_label])

# Prédiction sur le jeu de test
y_pred_dum = dum.predict(df_test)

# Evaluate
print("RMSE : {:.2f}".format(np.sqrt(mean_squared_error(df_test[model1_label], y_pred_dum)) ))
print("MAE : {:.2f}".format(mean_absolute_error(df_test[model1_label], y_pred_dum) ))


# RMSE : 27.57  
# MAE : real value is around 21 (4.21 was because we were doing square root of MAE which was an default in the code)
# 
# RMSE of naive approach was 42 before removing outliers

# In[83]:


error_mean = evaluate_model_percent_mean(dum, df_test, df_test[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')


# In[84]:


error_90p_5min = evaluate_model_percent_threshold(dum, df_test, df_test[model1_label], EVALUATION_PERCENT, EVALUATION_THRESHOLD)
print(f'{error_90p_5min : .2f}% predictions have error below {EVALUATION_THRESHOLD} min, {EVALUATION_PERCENT*100}% of the time')


# In[85]:


plt.scatter(df_test[model1_label], y_pred_dum, color='coral')


# In[86]:


df_test[model1_label]


# In[87]:


y_pred_dum


# In[88]:


df['ARR_DELAY'].abs().mean()


# => With all samples and 70% most represented features, without StandardScale :  on test set : lin_rmse = 42.17  
# => With all samples and 80% most represented features, without StandardScale :  on test set : lin_rmse = 42.16  
# => With all samples and 80% most represented features, with StandardScale :  on test set : lin_rmse = 42.16

# # Random forest

# In[89]:


from sklearn.ensemble import RandomForestRegressor

if (EXECUTE_INTERMEDIATE_MODELS == True):
    random_reg = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=42)
    random_reg.fit(df_train_transformed, df_train[model1_label])


# In[90]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_test_predictions = random_reg.predict(df_test_transformed)
    mse = mean_squared_error(df_test[model1_label], df_test_predictions)
    rmse = np.sqrt(mse)
    print(rmse)


# => 42.373691516139964

# # SVM

# In[91]:


'''
from sklearn.svm import SVR

svm_reg = SVR(kernel="rbf", verbose=True)
svm_reg.fit(df_train_transformed, df_train[model1_label])
'''


# In[92]:


from sklearn.svm import LinearSVR

svm_reg = LinearSVR(random_state=42, tol=1e-5, verbose=True)

if (EXECUTE_INTERMEDIATE_MODELS == True):
    svm_reg.fit(df_train_transformed, df_train[model1_label])


# In[93]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    evaluate_model(svm_reg, df_test_transformed, df_test[model1_label])


# => RMSE : 43.45607643335432

# In[94]:


grid_search_SVR = GridSearchCV(svm_reg, param_grid = {"epsilon": [0, 0.5],
                              "C": [1, 5, 10, 100, 1000],
                              "loss": ['epsilon_insensitive', 'squared_epsilon_insensitive'],},cv=shuffled_split_train, scoring='neg_mean_squared_error', error_score=np.nan, verbose=2)


# In[95]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    if (RECOMPUTE_GRIDSEARCH == True):
        grid_search_SVR.fit(df_train_transformed, df_train[model1_label])


# => Warning at execution : /home/francois/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   "the number of iterations.", ConvergenceWarning)  

# In[96]:


if ((SAVE_GRID_RESULTS == False) and (LOAD_GRID_RESULTS == True) and (EXECUTE_INTERMEDIATE_MODELS == True)):
    grid_search = None

if (EXECUTE_INTERMEDIATE_MODELS == True):
    grid_search_SVR, df_grid_search_results = save_or_load_search_params(grid_search_SVR, 'LinearSVR_20200319')


# In[97]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_grid_search_results.sort_values(by='mean_test_score', ascending=False)


# In[98]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    np.sqrt(1709.197402)


# In[99]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    grid_search_SVR.best_estimator_


# In[100]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    evaluate_model(grid_search_SVR.best_estimator_, df_test_transformed, df_test[model1_label])


# => Best estimator :  inearSVR(C=1, dual=True, epsilon=0, fit_intercept=True, intercept_scaling=1.0,
#           loss='squared_epsilon_insensitive', max_iter=1000, random_state=0,
#           tol=1e-05, verbose=True)  
# 
# => RMSE : 42.16

# # Polynomial features + linear regression

# In[101]:


df_train_transformed


# In[102]:


poly = ColumnTransformer([
                                ('poly', PolynomialFeatures(degree=2), [0, 1, 2, 3, 4, 5, 6])     
                                ], remainder='passthrough', sparse_threshold=1)

#poly.fit(df_train_transformed, df_train[model1_label])
#poly.fit(df_train_transformed)


# In[103]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train_transformed = poly.fit_transform(df_train_transformed)
    df_test_transformed = poly.transform(df_test_transformed)


# In[104]:


df_train_transformed.shape


# In[105]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    lin_reg = LinearRegression()
    lin_reg.fit(df_train_transformed, df_train[model1_label])


# In[106]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])


# => 42.11719088178065

# # Polynomial features + random forest

# In[107]:


from sklearn.ensemble import RandomForestRegressor

if (EXECUTE_INTERMEDIATE_MODELS == True):
    random_reg = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=42)
    random_reg.fit(df_train_transformed, df_train[model1_label])


# In[108]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_test_predictions = random_reg.predict(df_test_transformed)
    evaluate_model(random_reg, df_test_transformed, df_test[model1_label])


# In[109]:


#evaluate_model(polynomial_reg, df_test_transformed, df_test[model1_label])


# # New try with group by + mean + sort encoding of categorical features
# With preparation_pipeline_meansort instead of preparation_pipeline

# In[58]:


if (DATA_LOADED == True):
    del df
    del df_train
    del df_test
    del df_train_transformed
    del df_test_transformed


# In[59]:


df = load_data()


# In[60]:


all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)


# In[61]:


#df, df_train, df_test = custom_train_test_split_sample_random(df)
df, df_train, df_test = custom_train_test_split_sample(df)


# In[62]:


df_test


# In[63]:


df.loc[4681469]


# In[25]:


#df_train_transformed = preparation_pipeline_meansort_standardscale.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=None)


# In[64]:


df_train_transformed = preparation_pipeline_meansort_stdscale.fit_transform(df_train)
df_test_transformed = preparation_pipeline_meansort_stdscale.transform(df_test)
DATA_LOADED = True
df_test_transformed.shape


# In[65]:


df_test


# In[66]:


df_test_transformed


# In[116]:


df_train


# In[117]:


df_train_transformed


# In[118]:


df_train_transformed.shape[1]


# In[119]:


len(MODEL1_GOUPBYMEAN_FEATURES)


# In[120]:


df_train_transformed.shape[1]


# In[121]:


df_train[df_train['CRS_DEP_TIME'] < 200]


# In[122]:


df['CRS_DEP_TIME']


# In[123]:


df_train['CRS_DEP_TIME']


# In[124]:


plt.hist(df_train['CRS_DEP_TIME'], bins=50)


# In[125]:


for feat_name in df_train_transformed.columns:
    fig = plt.figure()
    fig.suptitle(feat_name)
    plt.hist(df_train_transformed[feat_name], bins=50)
    plt.plot()


# In[126]:


abs(df_train['ARR_DELAY'].min())


# In[127]:


(df_train['ARR_DELAY'] + abs(df_train['ARR_DELAY'].min())).hist(bins=50)


# In[128]:


df_train['ARR_DELAY'].hist(bins=50)


# In[129]:


df_train['ARR_DELAY'].hist(bins=50, log=True)


# In[130]:


df_test['ARR_DELAY'].hist(bins=50)


# In[131]:


df_train_labels = df_train[model1_label]
df_test_labels = df_test[model1_label]


# In[132]:


df_train_labels_positive = df_train[model1_label] + abs(df_train[model1_label].min()) + 1


# In[133]:


pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)


# In[134]:


df_train_labels_positive_log = pt.fit_transform(df_train_labels_positive.to_numpy().reshape(-1, 1))


# In[135]:


#df_train_labels_positive_log_inverse = pt.inverse_transform(df_train_labels_positive_log) -1 - abs(df_train['ARR_DELAY'].min())


# ## With scaling of labels

# In[136]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(df_train_transformed, df_train_labels_positive_log)

df_train_predictions_positive_log = lin_reg.predict(df_train_transformed)
df_train_predictions_positive = pt.inverse_transform(df_train_predictions_positive_log)
df_train_predictions = df_train_predictions_positive -1 - abs(df_train['ARR_DELAY'].min())

df_test_predictions_positive_log = lin_reg.predict(df_test_transformed)
df_test_predictions_positive = pt.inverse_transform(df_test_predictions_positive_log)
df_test_predictions = df_test_predictions_positive -1 - abs(df_test['ARR_DELAY'].min())

mse = mean_squared_error(df_train_labels, df_train_predictions)
rmse = np.sqrt(mse)
print(f'RMSE on training set : {rmse}')

mse = mean_squared_error(df_test_labels, df_test_predictions)
rmse = np.sqrt(mse)
print(f'RMSE on test set : {rmse}')


# => Log scaling of labels does not seem to make a difference. Result is even worse  (28.4 instead of 27)

# In[137]:


plt.hist(df_train_predictions_positive_log, bins=50)


# In[138]:


plt.hist(df_train_predictions, bins=50)


# ## Without scaling of labels

# In[139]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(df_train_transformed, df_train[model1_label])

df_test_predictions = lin_reg.predict(df_test_transformed)

mse = mean_squared_error(df_train_labels, df_train_predictions)
rmse = np.sqrt(mse)
print(f'RMSE on training set : {rmse}')

mse = mean_squared_error(df_test_labels, df_test_predictions)
rmse = np.sqrt(mse)
print(f'RMSE on test set : {rmse}')


# => Evaluation on test set :  
# RMSE : 27.079383490783385  
# 
#   
# Evaluation on training set :  
# RMSE : 27.07763523727725  
# 
# 
# 
# En ayant enlevé  :  'NBFLIGHTS_FORDAY_FORAIRPORT',  
#        'ORIGIN', 'UNIQUE_CARRIER' :  
# 
# Evaluation on test set :  
# RMSE : 27.19385016531133  
# 
# Remettre juste 'NBFLIGHTS_FORDAY_FORAIRPORT' n'y change rien  
# En remettant ORIGIN => passage à 27.14  
# En remettant UNIQUE_CARRIER => passage à 27.08

# In[140]:


error_percent_threshold = evaluate_model_percent_threshold(lin_reg, df_test_transformed, df_test[model1_label], EVALUATION_PERCENT, 20)
print(f'{error_percent_threshold : .2f}% predictions have error below {EVALUATION_THRESHOLD} min, {EVALUATION_PERCENT*100}% of the time')


# In[141]:


error_mean = evaluate_model_percent_mean(lin_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')


# In[142]:


plt.hist(df_train_predictions, bins=50)


# => RMSE on training set : 41.35267146874754 (close to RMSE on test set => under fitting)

# In[143]:


plt.hist(df_test_predictions, bins=50)


# In[144]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values on test set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_test[model1_label], df_test_predictions, color='coral', alpha=0.1)


# In[145]:


df_train_predictions = lin_reg.predict(df_train_transformed)


# In[146]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):

g = sns.jointplot(x=df_train[model1_label], y=df_train_predictions, kind='hex', color='blue', height=10)
#sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

g.set_axis_labels("Actual", "Predicted")
plt.subplots_adjust(top=0.9)
plt.suptitle('Linear regression : comparison actual values / predict values on training set', fontsize = 16)

plt.savefig('linreg_actual_vs_predicted_training_set.png', dpi=400)
    


# In[147]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / predicted values on test set')
plt.xlabel("Predicted")
plt.ylabel("Actual - Predicted")
plt.scatter(df_test[model1_label] - df_test_predictions, df_test_predictions, color='blue', alpha=0.1)


# In[148]:


df_train_predictions = lin_reg.predict(df_train_transformed)

#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison actual values / predict values on training set')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.1)


# In[149]:


lin_reg.coef_


# In[150]:


# Feature importances :
(abs(lin_reg.coef_) / (abs(lin_reg.coef_).sum()))


# Features < 0.01 importance :   
# 'NBFLIGHTS_FORDAY_FORAIRPORT',
#        'ORIGIN', 'UNIQUE_CARRIER'
#        
# => NBFLIGHTS_FORDAY_FORAIRPORT removed from MODEL1_GOUPBYMEAN_FEATURES  
# => ORIGIN and UNIQUE_CARRIER kept   

# In[151]:


df_train_transformed.shape


# In[152]:


df_train_transformed


# In[153]:


plot_learning_curves(lin_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE)


# ## Random forest

# In[154]:


get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import RandomForestRegressor\n\n#if (EXECUTE_INTERMEDIATE_MODELS == True):\nrandom_reg = RandomForestRegressor(n_estimators=100, max_depth=100, n_jobs=-1, random_state=42)\nrandom_reg.fit(df_train_transformed, df_train[model1_label])')


# In[155]:


# Model obtained via GridSearch :
#%%time
from sklearn.ensemble import RandomForestRegressor

#if (EXECUTE_INTERMEDIATE_MODELS == True):    
random_reg = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=1000,
                      max_features=4, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
                      oob_score=False, random_state=42, verbose=0,
                      warm_start=False)

random_reg.fit(df_train_transformed, df_train[model1_label])


# In[156]:


print("Evaluation on test set :")
evaluate_model(random_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(random_reg, df_train_transformed, df_train[model1_label])


# With parameters before GridSearch optimisation :
#     With shuffle split by ARR_DELAY strategy and 80000 samples :  
#     RMSE : 27.779707360544677  
# 
#     Evaluation on training set:  
#     RMSE : 10.27032737489414  

# In[157]:


error_mean = evaluate_model_percent_mean(random_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')


# => ~ 10 min with 80000 lines  
# => 9.75 min with 800000 lines

# In[158]:


error_mean_train = evaluate_model_percent_mean(random_reg, df_train_transformed, df_train[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean_train : .2f}')


# In[159]:


df_test_predictions = random_reg.predict(df_test_transformed)


# In[160]:


df_train_predictions = random_reg.predict(df_train_transformed)


# In[161]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / predicted values on test set')
plt.xlabel("Predicted")
plt.ylabel("Actual - Predicted")
plt.scatter(df_test_predictions, df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[162]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / actual values on test set')
plt.xlabel("Actual label")
plt.ylabel("Actual label - Predicted label")
plt.scatter(df_test[model1_label], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[163]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / predicted values on training set')
plt.xlabel("Actual label")
plt.ylabel("Actual label - Predicted label")
plt.scatter(df_train[model1_label], df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[164]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / instance numbers on training set')
plt.xlabel("Instance number")
plt.ylabel("Actual label - Predicted label")
plt.scatter(range(df_train.shape[0]), df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[165]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Predicted - actual / CRS_ELAPSED_TIME values on test set')
plt.xlabel("CRS_ELAPSED_TIME")
plt.ylabel("Actual - Predicted")
plt.scatter(df_test['CRS_ELAPSED_TIME'], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[166]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):

g = sns.jointplot(x=df_train[model1_label], y=df_train_predictions, kind='hex', color='blue', height=10)
#sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

g.set_axis_labels("Actual", "Predicted")
plt.subplots_adjust(top=0.9)
plt.suptitle('Random forest : comparison actual values / predict values on training set', fontsize = 16)

plt.savefig('randomreg_actual_vs_predicted_training_set.png', dpi=400)
    


# In[167]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):

g = sns.jointplot(x=df_test[model1_label], y=df_test_predictions, kind='hex', color='blue', height=10)
#sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

g.set_axis_labels("Actual", "Predicted")
plt.subplots_adjust(top=0.9)
plt.suptitle('Random forest overfit : comparison actual values / predict values on test set', fontsize = 16)

plt.savefig('randomreg_overfit_actual_vs_predicted_test_set.png', dpi=400)


# In[168]:


df_train_predictions = random_reg.predict(df_train_transformed)

#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Random forest : Comparison actual values / predict values on training set')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.1)


# In[169]:


df_test_predictions = random_reg.predict(df_test_transformed)

#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Random forest : Comparison actual values / predict values on test set')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.scatter(df_test[model1_label], df_test_predictions, color='coral', alpha=0.1)


# In[170]:


df_train_transformed.columns


# In[171]:


pd.set_option('display.max_rows', 200)


# In[172]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : random_reg.feature_importances_})


# In[173]:


pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].sort_values(by='Feature importance', ascending=False).cumsum()], axis=1)


# In[174]:


random_reg.feature_importances_


# In[175]:


random_reg.feature_importances_.cumsum()


# In[ ]:





# => feature importance : 

# In[176]:


random_reg.estimators_[0]


# In[177]:


'''
from sklearn.tree import export_graphviz
export_graphviz(random_reg.estimators_[0], out_file="tree.dot", rounded=True, filled=True)
'''


# In[178]:


LEARNING_CURVE_STEP_SIZE


# In[179]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    #df_test_predictions = random_reg.predict(df_test_transformed)
    #evaluate_model(random_reg, df_test_transformed, df_test[model1_label])
    plot_learning_curves(random_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], int(LEARNING_CURVE_STEP_SIZE))


# ## Random forest: Grid Search of parameters

# In[180]:


df_train_transformed.columns


# In[59]:


get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import RandomForestRegressor\n\n#with redirect_output("gridsearch_output_randomforest_mse_20200416.txt"):\nif (RECOMPUTE_GRIDSEARCH == True):\n    random_reg = RandomForestRegressor(n_jobs=-1, random_state=42)\n\n    param_grid = {\n            \'n_estimators\':  [10, 100, 200, 500, 1000],\n            \'max_depth\': [10, 100, 200, 500, 1000],\n            \'max_features\': [2, 4, 8, 12],\n            \'max_leaf_nodes\': [2, 10, 100, None],\n            #\'criterion\': [\'mse\', \'mae\'],\n            \'criterion\': [\'mse\'],\n            \'n_jobs\': [-1],\n            \'random_state\': [42],\n        }\n\n    grid_search = GridSearchCV(random_reg, param_grid, cv=5, verbose=2, error_score=np.nan, scoring=\'neg_mean_squared_error\')\n    grid_search.fit(df_train_transformed, df_train[model1_label])')


# [CV] criterion=mse, max_depth=1000, max_features=4, max_leaf_nodes=None, n_estimators=500, n_jobs=-1, random_state=42 
# 
# /home/francois/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_validation.py:547: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
# joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker. The exit codes of the workers are {SIGKILL(-9)}
# 
#   FitFailedWarning)
# 
# [CV]  criterion=mse, max_depth=1000, max_features=4, max_leaf_nodes=None, n_estimators=500, n_jobs=-1, random_state=42, total=  18.8s
# [CV] criterion=mse, max_depth=1000, max_features=4, max_leaf_nodes=None, n_estimators=500, n_jobs=-1, random_state=42 
# 
# exception calling callback for <Future at 0x7f4b6817e6d0 state=finished raised TerminatedWorkerError>
# Traceback (most recent call last):
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/_base.py", line 625, in _invoke_callbacks
#     callback(self)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 309, in __call__
#     self.parallel.dispatch_next()
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 731, in dispatch_next
#     if not self.dispatch_one_batch(self._original_iterator):
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 759, in dispatch_one_batch
#     self._dispatch(tasks)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 716, in _dispatch
#     job = self._backend.apply_async(batch, callback=cb)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/_parallel_backends.py", line 510, in apply_async
#     future = self._workers.submit(SafeFunction(func))
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/reusable_executor.py", line 151, in submit
#     fn, *args, **kwargs)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/process_executor.py", line 1022, in submit
#     raise self._flags.broken
# joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker. The exit codes of the workers are {SIGKILL(-9)}
# /home/francois/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_validation.py:547: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
# joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker. The exit codes of the workers are {SIGKILL(-9)}
# 
#   FitFailedWarning)
# 
# [CV]  criterion=mse, max_depth=1000, max_features=4, max_leaf_nodes=None, n_estimators=500, n_jobs=-1, random_state=42, total=  19.9s
# [CV] criterion=mse, max_depth=1000, max_features=4, max_leaf_nodes=None, n_estimators=1000, n_jobs=-1, random_state=42 
# 
# exception calling callback for <Future at 0x7f4b682b7b10 state=finished raised TerminatedWorkerError>
# Traceback (most recent call last):
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/_base.py", line 625, in _invoke_callbacks
#     callback(self)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 309, in __call__
#     self.parallel.dispatch_next()
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 731, in dispatch_next
#     if not self.dispatch_one_batch(self._original_iterator):
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 759, in dispatch_one_batch
#     self._dispatch(tasks)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 716, in _dispatch
#     job = self._backend.apply_async(batch, callback=cb)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/_parallel_backends.py", line 510, in apply_async
#     future = self._workers.submit(SafeFunction(func))
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/reusable_executor.py", line 151, in submit
#     fn, *args, **kwargs)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/process_executor.py", line 1022, in submit
#     raise self._flags.broken
# joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker. The exit codes of the workers are {SIGKILL(-9)}
# /home/francois/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_validation.py:547: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
# joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker. The exit codes of the workers are {SIGKILL(-9)}
# 
#   FitFailedWarning)
# 
# [CV]  criterion=mse, max_depth=1000, max_features=4, max_leaf_nodes=None, n_estimators=1000, n_jobs=-1, random_state=42, total=  45.4s
# [CV] criterion=mse, max_depth=1000, max_features=4, max_leaf_nodes=None, n_estimators=1000, n_jobs=-1, random_state=42 
# 
# exception calling callback for <Future at 0x7f4b6343a490 state=finished raised TerminatedWorkerError>
# Traceback (most recent call last):
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/_base.py", line 625, in _invoke_callbacks
#     callback(self)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 309, in __call__
#     self.parallel.dispatch_next()
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 731, in dispatch_next
#     if not self.dispatch_one_batch(self._original_iterator):
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 759, in dispatch_one_batch
#     self._dispatch(tasks)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 716, in _dispatch
#     job = self._backend.apply_async(batch, callback=cb)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/_parallel_backends.py", line 510, in apply_async
#     future = self._workers.submit(SafeFunction(func))
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/reusable_executor.py", line 151, in submit
#     fn, *args, **kwargs)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/process_executor.py", line 1022, in submit
#     raise self._flags.broken
# joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker. The exit codes of the workers are {SIGKILL(-9)}
# /home/francois/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_validation.py:547: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
# joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker. The exit codes of the workers are {SIGKILL(-9)}
# 
#   FitFailedWarning)
# 
# [CV]  criterion=mse, max_depth=1000, max_features=4, max_leaf_nodes=None, n_estimators=1000, n_jobs=-1, random_state=42, total=   8.0s
# [CV] criterion=mse, max_depth=1000, max_features=4, max_leaf_nodes=None, n_estimators=1000, n_jobs=-1, random_state=42 
# 
# exception calling callback for <Future at 0x7f4b67f74dd0 state=finished raised TerminatedWorkerError>
# Traceback (most recent call last):
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/_base.py", line 625, in _invoke_callbacks
#     callback(self)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 309, in __call__
#     self.parallel.dispatch_next()
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 731, in dispatch_next
#     if not self.dispatch_one_batch(self._original_iterator):
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 759, in dispatch_one_batch
#     self._dispatch(tasks)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/parallel.py", line 716, in _dispatch
#     job = self._backend.apply_async(batch, callback=cb)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/_parallel_backends.py", line 510, in apply_async
#     future = self._workers.submit(SafeFunction(func))
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/reusable_executor.py", line 151, in submit
#     fn, *args, **kwargs)
#   File "/home/francois/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/process_executor.py", line 1022, in submit
#     raise self._flags.broken
# joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker. The exit codes of the workers are {SIGKILL(-9)}
# /home/francois/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_validation.py:547: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
# joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker. The exit codes of the workers are {SIGKILL(-9)}
# 
#   FitFailedWarning)
# 
# 
# [CV] criterion=mse, max_depth=1000, max_features=12, max_leaf_nodes=None, n_estimators=1000, n_jobs=-1, random_state=42 
# 
# /home/francois/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_validation.py:547: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
# joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker. The exit codes of the workers are {SIGKILL(-9)}
# 
#   FitFailedWarning)
# [Parallel(n_jobs=1)]: Done 2000 out of 2000 | elapsed: 404.9min finished
# 

# In[27]:


if ((SAVE_GRID_RESULTS == False) and (LOAD_GRID_RESULTS == True)):
    grid_search = None
    
grid_search, df_grid_search_results = save_or_load_search_params(grid_search, 'randomforest_meansort_80000samples_20200414')


# In[28]:


grid_search.best_estimator_


# In[29]:


pd.set_option('display.max_rows', 1000)
df_grid_search_results.sort_values(by='mean_test_score', ascending=False)


# In[185]:


print("Evaluation on test set :")
evaluate_model(grid_search.best_estimator_, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(grid_search.best_estimator_, df_train_transformed, df_train[model1_label])


# ## Random forest: additionnal Grid Search of parameters
# We had a memory error on those runs : we'll need to launch them again  
# [CV] criterion=mse, max_depth=1000, max_features=4, max_leaf_nodes=None, n_estimators=500, n_jobs=-1, random_state=42     
# [CV] criterion=mse, max_depth=1000, max_features=4, max_leaf_nodes=None, n_estimators=1000, n_jobs=-1, random_state=42   
# 

# In[186]:


get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import RandomForestRegressor\n\n#with redirect_output("gridsearch_output_randomforest_mse_20200416.txt"):\nif (RECOMPUTE_GRIDSEARCH == True):\n    random_reg = RandomForestRegressor(n_jobs=-1, random_state=42)\n\n    param_grid = {\n            \'n_estimators\':  [500],\n            \'max_depth\': [1000],\n            \'max_features\': [4],\n            \'max_leaf_nodes\': [None],\n            #\'criterion\': [\'mse\', \'mae\'],\n            \'criterion\': [\'mse\'],\n            \'n_jobs\': [-1],\n            \'random_state\': [42],\n        }\n\n    grid_search2 = GridSearchCV(random_reg, param_grid, cv=5, verbose=2, error_score=np.nan, scoring=\'neg_mean_squared_error\')\n    grid_search2.fit(df_train_transformed, df_train[model1_label])')


# In[187]:


if ((SAVE_GRID_RESULTS == False) and (LOAD_GRID_RESULTS == True)):
    grid_search2 = None
    
grid_search2, df_grid_search_results2 = save_or_load_search_params(grid_search2, 'randomforest_meansort_80000samples_run2_20200414')


# In[188]:


grid_search2.best_estimator_


# In[189]:


pd.set_option('display.max_rows', 1000)
df_grid_search_results2.sort_values(by='mean_test_score', ascending=False)


# ## Evaluation of Grid Search best estimator
# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,  
#                       max_features=4, max_leaf_nodes=None,  
#                       min_impurity_decrease=0.0, min_impurity_split=None,  
#                       min_samples_leaf=1, min_samples_split=2,  
#                       min_weight_fraction_leaf=0.0, n_estimators=1000,  
#                       n_jobs=-1, oob_score=False, random_state=42, verbose=0,  
#                       warm_start=False)  

# In[30]:


random_reg = grid_search.best_estimator_


# In[31]:


print("Evaluation on test set :")
evaluate_model(random_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(random_reg, df_train_transformed, df_train[model1_label])

error_mean = evaluate_model_percent_mean(random_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')

error_mean_worst = evaluate_model_percent_worst_mean(random_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {100 - (EVALUATION_PERCENT)*100}% of the time : {error_mean_worst : .2f}')


# Model before Grid Search :
# => ~ 10 min with 80000 lines  
# => 9.75 min with 800000 lines
# 
# Grid search best estimator :  
# => Mean prediction error 90.0% of the time :  9.92   with 80000 lines
# 

# In[32]:


df_test_predictions = random_reg.predict(df_test_transformed)


# In[33]:


df_train_predictions = random_reg.predict(df_train_transformed)


# In[66]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / predicted values on test set')
plt.xlabel("Predicted")
plt.ylabel("Actual - Predicted")
plt.scatter(df_test_predictions, df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[67]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / actual values on test set')
plt.xlabel("Actual label")
plt.ylabel("Actual label - Predicted label")
plt.scatter(df_test[model1_label], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[68]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / predicted values on training set')
plt.xlabel("Actual label")
plt.ylabel("Actual label - Predicted label")
plt.scatter(df_train[model1_label], df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[72]:


df_train_residuals = df_train[model1_label] - df_train_predictions
max_residual = df_train_residuals.abs().max()
sample_weights = (max_residual - df_train_residuals.abs()) / max_residual


# In[73]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / predicted values on training set')
plt.xlabel("Actual label")
plt.ylabel("Actual label - Predicted label")
plt.scatter(df_train[model1_label],sample_weights, color='blue', alpha=0.1)


# In[74]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):

g = sns.jointplot(x=df_train[model1_label], y=sample_weights, kind='hex', color='blue', height=10)
#sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

g.set_axis_labels("Actual label", "Weighted instance value")
plt.subplots_adjust(top=0.9)
plt.suptitle('Random forest : weighted values based on residuals', fontsize = 16)

plt.savefig('linreg_residuals_weighted_training_set.png', dpi=400)


# In[75]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / instance numbers on training set')
plt.xlabel("Instance number")
plt.ylabel("Actual label - Predicted label")
plt.scatter(range(df_train.shape[0]), df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[76]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Predicted - actual / CRS_ELAPSED_TIME values on test set')
plt.xlabel("CRS_ELAPSED_TIME")
plt.ylabel("Actual - Predicted")
plt.scatter(df_test['CRS_ELAPSED_TIME'], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[77]:


df_train_predictions = random_reg.predict(df_train_transformed)

#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Random forest : Comparison actual values / predict values on training set')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.1)


# In[75]:


df_test_predictions = random_reg.predict(df_test_transformed)

#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Random forest : Comparison actual values / predict values on test set')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.scatter(df_test[model1_label], df_test_predictions, color='coral', alpha=0.1)


# In[76]:


df_train_transformed.columns


# In[77]:


pd.set_option('display.max_rows', 200)


# In[78]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : random_reg.feature_importances_})


# In[79]:


df_feature_importances_global = pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].rename(columns={'Feature importance' : 'Cumulated feature importance'}).sort_values(by='Cumulated feature importance', ascending=False).cumsum()], axis=1)


# In[80]:


#df.style.set_properties(**{'text-align': 'center'})

df_feature_importances_global.style.format({
    'Feature importance': '{:,.2%}'.format,
    'Cumulated feature importance': '{:,.2%}'.format,
})


# In[81]:


random_reg.feature_importances_


# In[82]:


random_reg.feature_importances_.cumsum()


# In[ ]:





# => feature importance : 

# In[83]:


random_reg.estimators_[0]


# In[84]:


'''
from sklearn.tree import export_graphviz
export_graphviz(random_reg.estimators_[0], out_file="tree.dot", rounded=True, filled=True)
'''


# In[85]:


LEARNING_CURVE_STEP_SIZE


# In[86]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
#df_test_predictions = random_reg.predict(df_test_transformed)
#evaluate_model(random_reg, df_test_transformed, df_test[model1_label])
plot_learning_curves(random_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], int(LEARNING_CURVE_STEP_SIZE))


# ## Boosting

# In[215]:


from sklearn.ensemble import GradientBoostingRegressor

boost_reg = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=10, subsample=0.1, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=10, init=None, random_state=42, max_features=4, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False)

boost_reg.fit(df_train_transformed, df_train[model1_label])


# In[216]:


print("Evaluation on test set :")
evaluate_model(boost_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(boost_reg, df_train_transformed, df_train[model1_label])


# In[217]:


error_mean = evaluate_model_percent_mean(boost_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')


# With boosting, all data :  
# Mean prediction error 90.0% of the time :  9.88

# In[218]:


df_test_predictions = boost_reg.predict(df_test_transformed)


# In[219]:


df_train_predictions = boost_reg.predict(df_train_transformed)


# In[220]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / predicted values on test set')
plt.xlabel("Predicted")
plt.ylabel("Actual - Predicted")
plt.scatter(df_test_predictions, df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[221]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / actual values on test set')
plt.xlabel("Actual label")
plt.ylabel("Actual label - Predicted label")
plt.scatter(df_test[model1_label], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[222]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / predicted values on training set')
plt.xlabel("Actual label")
plt.ylabel("Actual label - Predicted label")
plt.scatter(df_train[model1_label], df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[223]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / instance numbers on training set')
plt.xlabel("Instance number")
plt.ylabel("Actual label - Predicted label")
plt.scatter(range(df_train.shape[0]), df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[224]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Predicted - actual / CRS_ELAPSED_TIME values on test set')
plt.xlabel("CRS_ELAPSED_TIME")
plt.ylabel("Actual - Predicted")
plt.scatter(df_test['CRS_ELAPSED_TIME'], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[225]:


df_train_predictions = boost_reg.predict(df_train_transformed)

#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Random forest : Comparison actual values / predict values on training set')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.1)


# In[226]:


df_test_predictions = boost_reg.predict(df_test_transformed)

#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Random forest : Comparison actual values / predict values on test set')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.scatter(df_test[model1_label], df_test_predictions, color='coral', alpha=0.1)


# In[227]:


df_train_transformed.columns


# In[228]:


pd.set_option('display.max_rows', 200)


# In[229]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : boost_reg.feature_importances_})


# In[230]:


pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].sort_values(by='Feature importance', ascending=False).cumsum()], axis=1)


# ## Final model : random forest with weighted optimisation (suppress training instances)

# In[78]:


WEIGHT_THRESHOLD = 0.8
df_train_transformed = df_train_transformed[sample_weights > WEIGHT_THRESHOLD]


# In[79]:


df_train = df_train[sample_weights > WEIGHT_THRESHOLD]


# In[80]:


sample_weights_haircut = sample_weights[sample_weights > WEIGHT_THRESHOLD]


# In[81]:


get_ipython().run_cell_magic('time', '', "from sklearn.ensemble import RandomForestRegressor\n\n#if (EXECUTE_INTERMEDIATE_MODELS == True):\nrandom_reg = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,\n                  max_features=4, max_leaf_nodes=None,\n                  min_impurity_decrease=0.0, min_impurity_split=None,\n                  min_samples_leaf=1, min_samples_split=2,\n                  min_weight_fraction_leaf=0.0, n_estimators=1000,\n                  n_jobs=-1, oob_score=False, random_state=42, verbose=0,\n                  warm_start=False)\nrandom_reg.fit(df_train_transformed, df_train[model1_label], sample_weights_haircut)")


# In[82]:


print("Evaluation on test set :")
evaluate_model(random_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(random_reg, df_train_transformed, df_train[model1_label])


# With weight threshold of 0.7 :  
# Evaluation on test set :  
# RMSE : 27.505904323162397  
# 
# 
# Evaluation on training set :  
# RMSE : 15.105634267719225  
# 
# With weight threshold of 0.9 :  
# Evaluation on test set :  
# RMSE : 27.34542651549861  
# 
# 
# Evaluation on training set :  
# RMSE : 8.161417508179818  
# 
# With weight threshold of 0.8 :  
# Evaluation on test set :  
# RMSE : 27.691875273334873  
# 
# 
# Evaluation on training set :  
# RMSE : 12.758479277575331  

# In[83]:


error_mean = evaluate_model_percent_mean(random_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')


# => ~ 10 min with 80000 lines  
# => 9.75 min with 800000 lines
# 
# With sample weight optimisation : 
# Mean prediction error 90.0% of the time :  9.02
# 
# With weight threshold of 0.7 :  
# Mean prediction error 90.0% of the time :  8.13    
# 
# With weight threshold of 0.9 :  
# Mean prediction error 90.0% of the time :  8.35  
# 
# With weight threshold of 0.8 :  
# Mean prediction error 90.0% of the time :  7.90

# In[84]:


error_mean_worst = evaluate_model_percent_worst_mean(random_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {100 - (EVALUATION_PERCENT)*100}% of the time : {error_mean_worst : .2f}')


# In[85]:


df_test_predictions = random_reg.predict(df_test_transformed)


# In[86]:


df_train_predictions = random_reg.predict(df_train_transformed)


# In[87]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / predicted values on test set')
plt.xlabel("Predicted")
plt.ylabel("Actual - Predicted")
plt.scatter(df_test_predictions, df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[88]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / actual values on test set')
plt.xlabel("Actual label")
plt.ylabel("Actual label - Predicted label")
plt.scatter(df_test[model1_label], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[89]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / predicted values on training set')
plt.xlabel("Actual label")
plt.ylabel("Actual label - Predicted label")
plt.scatter(df_train[model1_label], df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[90]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Actual - predicted / instance numbers on training set')
plt.xlabel("Instance number")
plt.ylabel("Actual label - Predicted label")
plt.scatter(range(df_train.shape[0]), df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[91]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Comparison Predicted - actual / CRS_ELAPSED_TIME values on test set')
plt.xlabel("CRS_ELAPSED_TIME")
plt.ylabel("Actual - Predicted")
plt.scatter(df_test['CRS_ELAPSED_TIME'], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[92]:


df_train_predictions = random_reg.predict(df_train_transformed)

#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Random forest : Comparison actual values / predict values on training set')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.1)


# In[93]:


df_test_predictions = random_reg.predict(df_test_transformed)

#if (EXECUTE_INTERMEDIATE_MODELS == True):
fig = plt.figure()
fig.suptitle('Random forest : Comparison actual values / predict values on test set')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.scatter(df_test[model1_label], df_test_predictions, color='coral', alpha=0.1)


# In[94]:


df_train_transformed.columns


# In[95]:


pd.set_option('display.max_rows', 50)


# In[96]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : random_reg.feature_importances_})


# In[97]:


pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].sort_values(by='Feature importance', ascending=False).cumsum()], axis=1)


# Tests

# # Save API model files for user interface

# In[98]:


if (SAVE_API_MODEL == True):    
    API_model = {}
    API_model['dataprep'] = preparation_pipeline_meansort_stdscale
    API_model['prediction'] = random_reg
    
    with open(API_MODEL_PICKLE_FILE, 'wb') as f:
        pickle.dump(API_model, f, pickle.HIGHEST_PROTOCOL)        


# ## Printing values to test the API

# In[99]:


realdf_test_predictions = pd.DataFrame(df_test_predictions, columns=['Predicted ARR delay'])


# In[100]:


realdf_test_predictions_with_label = pd.concat([df_test.reset_index(), realdf_test_predictions.reset_index()], axis=1)


# In[106]:


realdf_test_predictions_with_label[realdf_test_predictions_with_label['Predicted ARR delay'] > 17]


# In[102]:


df.dtypes


# In[103]:


df_train.dtypes


# In[104]:


df_test_predictions.max()


# In[105]:


df.loc[1876726, :]


# In[68]:


df_test_transformed.loc[1876726, :]


# In[180]:


df.loc[2494819, :]


# ## Polynomial regression degree 3

# In[251]:


poly = PolynomialFeatures(degree=3)
poly.fit(df_train_transformed)
df_train_transformed = poly.transform(df_train_transformed)
df_test_transformed = poly.transform(df_test_transformed)


# In[252]:


poly.n_output_features_


# In[253]:


df_train_transformed.shape


# In[254]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
lin_reg = LinearRegression()
lin_reg.fit(df_train_transformed, df_train[model1_label])


# In[255]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):
evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])


# => RMSE on test set with outliers : RMSE : 42.12678182212536  
# => RMSE on test set 80000 instances degree 2 :26.962742760959262  
# => RMSE on test set 80000 instances degree 3: 26.973373855188367

# In[256]:


evaluate_model(lin_reg, df_train_transformed, df_train[model1_label])


# => RMSE on training set with outliers : 41.26055791264713  
# => RMSE on training set 80000 instances degree 2 : 26.962742760959262
# => RMSE on trainings et with 80000 instances degree 3 :  RMSE : 26.64926940350148

# In[257]:


print('\n')

error_mean = evaluate_model_percent_mean(lin_reg, df_train_transformed, df_train[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')


# In[258]:


print('\n')

error_mean_test = evaluate_model_percent_mean(lin_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean_test: .2f}')


# In[259]:


plot_learning_curves(lin_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE)


# In[260]:


# Feature importances :
(abs(lin_reg.coef_) / (abs(lin_reg.coef_).sum()))


# In[261]:


df_train_transformed[:,0].shape


# In[262]:


df_train_predictions =  lin_reg.predict(df_train_transformed)
df_test_predictions =  lin_reg.predict(df_test_transformed)


# In[263]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):

g = sns.jointplot(x=df_train[model1_label], y=df_train_predictions, kind='hex', color='blue', height=10)
#sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

g.set_axis_labels("Actual", "Predicted")
plt.subplots_adjust(top=0.9)
plt.suptitle('Linear regression : comparison actual values / predict values on training set', fontsize = 16)

plt.savefig('linreg_actual_vs_predicted_training_set.png', dpi=400)


# In[264]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):

g = sns.jointplot(x=df_test[model1_label], y=df_test_predictions, kind='hex', color='blue', height=10)
#sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

g.set_axis_labels("Actual", "Predicted")
plt.subplots_adjust(top=0.9)
plt.suptitle('Linear regression : comparison actual values / predict values on test set', fontsize = 16)

plt.savefig('linreg_actual_vs_predicted_test_set.png', dpi=400)


# ## Polynomial regression univariate, and higher degree

# ### Degree 8

# In[ ]:


if (DATA_LOADED == True):
    del df
    del df_train
    del df_test
    del df_train_transformed
    del df_test_transformed


# In[24]:


df = load_data()


# In[25]:


all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)


# In[26]:


df, df_train, df_test = custom_train_test_split_sample(df)


# In[27]:


df_train_transformed = preparation_pipeline_meansort.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=None)
df_train_transformed = prediction_pipeline_without_sparse.fit_transform(df_train_transformed)

df_test_transformed = preparation_pipeline_meansort.transform(df_test)
df_test_transformed = prediction_pipeline_without_sparse.transform(df_test_transformed)
DATA_LOADED = True
df_test_transformed.shape


# In[28]:


nb_instances = df_train_transformed.shape[0]


# In[29]:


poly = PolynomialFeaturesUnivariateAdder(n_degrees = 8)


# In[30]:


df_train_transformed = poly.fit_transform(df_train_transformed)
df_test_transformed = poly.fit_transform(df_test_transformed)


# In[31]:


lin_reg = LinearRegression()

lin_reg.fit(df_train_transformed, df_train[model1_label])

df_test_predictions = lin_reg.predict(df_test_transformed)

print("Evaluation on test set :")
evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(lin_reg, df_train_transformed, df_train[model1_label])


# In[32]:


plot_learning_curves(lin_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE)


# ### Degree 4

# In[33]:


del df
del df_train
del df_test
del df_train_transformed
del df_test_transformed


# In[34]:


df, df_train, df_test, df_train_transformed, df_test_transformed = reset_data()


# In[35]:


df_train_transformed


# In[36]:


poly = PolynomialFeaturesUnivariateAdder(n_degrees = 4)
df_train_transformed = poly.fit_transform(df_train_transformed)
df_test_transformed = poly.transform(df_test_transformed)


# In[37]:


lin_reg = LinearRegression()

lin_reg.fit(df_train_transformed, df_train[model1_label])

df_test_predictions = lin_reg.predict(df_test_transformed)
evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])
plot_learning_curves(lin_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE)


# In[38]:


#lin_reg.summary


# # New try with 1 hot encode of : 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK'

# In[39]:


if (DATA_LOADED == True):
    del df
    del df_train
    del df_test
    del df_train_transformed
    del df_test_transformed


# In[40]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df = load_data()


# In[41]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)


# In[42]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df, df_train, df_test = custom_train_test_split_sample(df)


# In[43]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train_transformed = preparation_pipeline_meansort.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK'])
    df_train_transformed = prediction_pipeline_1hotall_without_sparse.fit_transform(df_train_transformed)

    df_test_transformed = preparation_pipeline_meansort.transform(df_test)
    df_test_transformed = prediction_pipeline_1hotall_without_sparse.transform(df_test_transformed)
    DATA_LOADED = True
    df_test_transformed.shape


# In[44]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()

    lin_reg.fit(df_train_transformed, df_train[model1_label])

    df_test_predictions = lin_reg.predict(df_test_transformed)
    evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])


# => RMSE : 41.98  
# => RMSE without outliers : 26.88

# In[45]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    evaluate_model(lin_reg, df_train_transformed, df_train[model1_label])


# => RMSE on training set : 41.12  
# => RMSE training set without outliers : 26.89

# In[46]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    lin_reg.coef_


# In[47]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    # Feature importances :
    (abs(lin_reg.coef_) / (abs(lin_reg.coef_).sum()))


# In[48]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train_transformed.shape


# In[49]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train_transformed


# In[50]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plot_learning_curves(lin_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE)


# In[51]:


'''
To gain memory
y_train = df_train[model1_label]
y_test = df_test[model1_label]
del df_train
del df_test
'''


# In[52]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train_transformed.shape


# In[53]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    np.asarray(df_train_transformed)


# In[54]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    #X2 = sm.add_constant(df_train_transformed)
    est = sm.OLS(df_train[model1_label], np.asarray(df_train_transformed.astype(float)))
    est2 = est.fit()
    print(est2.summary())


# In[55]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_test[model1_label], df_test_predictions, color='coral')


# In[56]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plt.hist(df_test_predictions, bins=50)


# In[57]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plt.hist(df_test[model1_label], bins=50)


# ### Degree 2

# In[58]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    nb_instances = df_train_transformed.shape[0]


# In[59]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    poly = PolynomialFeaturesUnivariateAdder(n_degrees = 2)


# In[60]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train_transformed = poly.fit_transform(df_train_transformed)
    df_test_transformed = poly.fit_transform(df_test_transformed)


# In[61]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    lin_reg = LinearRegression()

    lin_reg.fit(df_train_transformed, df_train[model1_label])

    df_test_predictions = lin_reg.predict(df_test_transformed)
    evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])
    plot_learning_curves(lin_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE)


# # Random forest

# In[62]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    from sklearn.ensemble import RandomForestRegressor

    if (EXECUTE_INTERMEDIATE_MODELS == True):
        random_reg = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42)
        random_reg.fit(df_train_transformed, df_train[model1_label])


# In[63]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    random_reg.feature_importances_


# => feature importance : 25% for CRS_ARR_TIME and 14% for UNIQUE_CARRIER  in previous model  (not this one)

# In[64]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    random_reg.estimators_[0]


# In[65]:


'''
from sklearn.tree import export_graphviz
export_graphviz(random_reg.estimators_[0], out_file="tree.dot", rounded=True, filled=True)
'''


# In[66]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_test_predictions = random_reg.predict(df_test_transformed)
    evaluate_model(random_reg, df_test_transformed, df_test[model1_label])
    plot_learning_curves(random_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE)


# In[67]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plot_learning_curves(lin_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE, evaluation_method='MAE')


# # Cheat model : give access to the model to the ARR_DELAY variable ! it should now learn

# In[68]:


if ((DATA_LOADED == True) and (EXECUTE_INTERMEDIATE_MODELS == True)):
    del df
    del df_train
    del df_test
    del df_train_transformed
    del df_test_transformed


# In[69]:


df = load_data()


# In[70]:


all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)


# In[71]:


df, df_train, df_test = custom_train_test_split_sample(df)


# In[72]:


df_train_transformed = preparation_pipeline_meansort.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK'])
df_train_transformed = prediction_pipeline_cheat_without_sparse.fit_transform(df_train_transformed)

df_test_transformed = preparation_pipeline_meansort.transform(df_test)
df_test_transformed = prediction_pipeline_cheat_without_sparse.transform(df_test_transformed)
DATA_LOADED = True
df_test_transformed.shape


# In[73]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(df_train_transformed, df_train[model1_label])

df_test_predictions = lin_reg.predict(df_test_transformed)
evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])


# 

# In[74]:


evaluate_model(lin_reg, df_train_transformed, df_train[model1_label])


# => RMSE on training set : 41.12  
# => RMSE training set without outliers : 26.89

# In[75]:


lin_reg.coef_


# In[76]:


# Feature importances :
(abs(lin_reg.coef_) / (abs(lin_reg.coef_).sum()))


# In[77]:


df_train_transformed.shape


# In[78]:


df_train_transformed


# In[79]:


plot_learning_curves(lin_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE)


# In[80]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / predicted values on test set')
    plt.xlabel("Actual label")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(df_test[model1_label], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[81]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values on test set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_test[model1_label], df_test_predictions, color='coral', alpha=0.1)


# # Random forest without polynomial feature

# In[82]:


if ((DATA_LOADED == True) and (EXECUTE_INTERMEDIATE_MODELS == True)):
    del df
    del df_train
    del df_test
    del df_train_transformed
    del df_test_transformed
    del df_train_predictions
    del df_test_predictions


# In[83]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df = load_data()


# In[84]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)


# In[85]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df, df_train, df_test = custom_train_test_split_sample(df)


# In[86]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train_transformed = preparation_pipeline_meansort.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK'])
    df_train_transformed = prediction_pipeline_1hotall_without_sparse.fit_transform(df_train_transformed)

    df_test_transformed = preparation_pipeline_meansort.transform(df_test)
    df_test_transformed = prediction_pipeline_1hotall_without_sparse.transform(df_test_transformed)
    DATA_LOADED = True
    df_test_transformed.shape


# In[87]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    get_ipython().run_line_magic('time', '')
    from sklearn.ensemble import RandomForestRegressor

    if (EXECUTE_INTERMEDIATE_MODELS == True):
        random_reg = RandomForestRegressor(n_estimators=100, max_depth=100, n_jobs=-1, random_state=42)
        random_reg.fit(df_train_transformed, df_train[model1_label])


# In[88]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    print("Evaluation on test set :")
    evaluate_model(random_reg, df_test_transformed, df_test[model1_label])

    print('\n')

    print("Evaluation on training set :")
    evaluate_model(random_reg, df_train_transformed, df_train[model1_label])


# => n_estimators=10, max_depth=10 : RMSE = 26.489032357237143  
# => n_estimators=100, max_depth=10 : RMSE = 26.452279766206914  
# => n_estimators=100, max_depth=100 : RMSE train = 9.623992685309045, RMSE test = 25.688478031845328

# In[89]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_test_predictions = random_reg.predict(df_test_transformed)


# In[90]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / predicted values on test set')
    plt.xlabel("Predicted")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test[model1_label] - df_test_predictions, df_test_predictions, color='blue', alpha=0.1)


# In[91]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    random_reg.feature_importances_


# In[92]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    random_reg.feature_importances_.cumsum()


# => feature importance : 25% for CRS_ARR_TIME and 14% for UNIQUE_CARRIER  in previous model  (not this one)

# In[93]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    random_reg.estimators_[0]


# In[94]:


'''
from sklearn.tree import export_graphviz
export_graphviz(random_reg.estimators_[0], out_file="tree.dot", rounded=True, filled=True)
'''


# In[95]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    LEARNING_CURVE_STEP_SIZE


# In[96]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_test_predictions = random_reg.predict(df_test_transformed)
    evaluate_model(random_reg, df_test_transformed, df_test[model1_label])
    plot_learning_curves(random_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE*5)


# In[97]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_test[model1_label], df_test_predictions, color='coral', alpha=0.1)


# In[98]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    df_train_predictions = random_reg.predict(df_train_transformed)

if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values on training set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.1)


# In[99]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plt.hist(df_test_predictions, bins=50)


# In[100]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plt.hist(df_test[model1_label], bins=50)


# # New try with 1 hot encode of : 'ORIGIN', 'CARRIER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_DEP_TIME' (scheduled dep hour)

# In[101]:


if ((DATA_LOADED == True) and (EXECUTE_INTERMEDIATE_MODELS == True)):
    del df
    del df_train
    del df_test
    del df_train_transformed
    del df_test_transformed


# In[102]:


df = load_data()


# In[103]:


df


# In[104]:


all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)


# In[105]:


#df, df_train, df_test = custom_train_test_split_sample_random(df)
df, df_train, df_test = custom_train_test_split_sample(df)


# In[106]:


#df_train_transformed = preparation_pipeline_meansort_standardscale.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=None)


# In[107]:


df_train_transformed = preparation_pipeline_1hotall_minmax.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN', 'UNIQUE_CARRIER', 'CRS_DEP_TIME'])
#df_train_transformed = preparation_pipeline_1hotall_minmax.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN', 'UNIQUE_CARRIER'])
df_test_transformed = preparation_pipeline_1hotall_minmax.transform(df_test)
DATA_LOADED = True
df_test_transformed.shape


# In[108]:


for feat_name in df_train_transformed.columns:
    if (feat_name in MODEL_1HOTALL_FEATURES_QUANTITATIVE):
        fig = plt.figure()
        fig.suptitle(feat_name)
        plt.hist(df_train_transformed[feat_name], bins=50)
        plt.plot()


# ## Linear regression

# In[109]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(df_train_transformed, df_train[model1_label])

df_test_predictions = lin_reg.predict(df_test_transformed)

print("Evaluation on test set :")
evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(lin_reg, df_train_transformed, df_train[model1_label])


# => With 80000 samples and train_test_split simple random (without stratify on ARR_DELAY ):  
# Evaluation on test set :  
# RMSE : 27.143078661756135  
# 
# 
# Evaluation on training set :  
# RMSE : 27.053691910444368  
# 
# 
# Same with stratify split on ARR_DELAY :
# Evaluation on test set :  
# RMSE : 26.972141699472907  
# 
# 
# Evaluation on training set :  
# RMSE : 26.873648666466085  

# In[110]:


plt.hist(df_test_predictions, bins=50)


# In[111]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values on test set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_test[model1_label], df_test_predictions, color='coral', alpha=0.1)


# In[112]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / predicted values on test set')
    plt.xlabel("Predicted")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test_predictions, df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[113]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / actual values on test set')
    plt.xlabel("Actual")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test[model1_label], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[114]:


df_train_predictions = lin_reg.predict(df_train_transformed)

if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values on training set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.1)


# In[115]:


df_test_transformed.columns


# In[116]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Predicted - actual / CRS_ELAPSED_TIME values on test set')
    plt.xlabel("CRS_ELAPSED_TIME")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test['CRS_ELAPSED_TIME'], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[117]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    for column_name in df_test_transformed.columns:
        fig = plt.figure()
        fig.suptitle('Comparison Predicted - actual / values of one feature on test set')
        plt.xlabel(column_name)
        plt.ylabel("Actual - Predicted")
        plt.scatter(df_test_transformed[column_name], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[118]:


lin_reg.coef_


# In[119]:


coef_feature_importances = (abs(lin_reg.coef_) / (abs(lin_reg.coef_).sum()))


# In[120]:


coef_feature_importances.sum()


# In[121]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : coef_feature_importances})


# In[122]:


pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].sort_values(by='Feature importance', ascending=False).cumsum()], axis=1)


# ## Random forest

# In[123]:


get_ipython().run_cell_magic('time', '', "from sklearn.ensemble import RandomForestRegressor\n\n#if (EXECUTE_INTERMEDIATE_MODELS == True):\n    #Old Random Fores before having done cross validation (done in group + mean sort encoding part)\n    #random_reg = RandomForestRegressor(n_estimators=100, max_depth=100, n_jobs=-1, random_state=42)\n    \nrandom_reg = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,\n                  max_features=4, max_leaf_nodes=None,\n                  min_impurity_decrease=0.0, min_impurity_split=None,\n                  min_samples_leaf=1, min_samples_split=2,\n                  min_weight_fraction_leaf=0.0, n_estimators=1000,\n                  n_jobs=-1, oob_score=False, random_state=42, verbose=0,\n                  warm_start=False)\n\nrandom_reg.fit(df_train_transformed, df_train[model1_label])")


# In[124]:


print("Evaluation on test set :")
evaluate_model(random_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(random_reg, df_train_transformed, df_train[model1_label])

error_mean = evaluate_model_percent_mean(random_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')


# With old random forest (without cross validation)
# 
#     With random split strategy :
# 
#     Evaluation on test set :  
#     RMSE : 27.067602891326597  
# 
#     Evaluation on training set :  
#     RMSE : 10.126836763135943  
# 
#     With shuffle split by ARR_DELAY strategy :  
#     Evaluation on test set :  
#     RMSE : 27.16789238410523  
# 
# 
#     Evaluation on training set :  
#     RMSE : 10.072030081721588  
#     
# With new random forest (after cross validation):  
#     With shuffle split by ARR_DELAY strategy :    
#     
#     Evaluation on test set :
#     RMSE : 27.2916161530929
# 
# 
#     Evaluation on training set :
#     RMSE : 27.00703718949282
#     
#     Mean prediction error 90.0% of the time :  10.22

# In[125]:


df_test_predictions = random_reg.predict(df_test_transformed)


# In[126]:


df_train_predictions = random_reg.predict(df_train_transformed)


# In[127]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / predicted values on test set')
    plt.xlabel("Predicted")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test_predictions, df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[128]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / actual values on test set')
    plt.xlabel("Actual label")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(df_test[model1_label], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[129]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / predicted values on training set')
    plt.xlabel("Actual label")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(df_train[model1_label], df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[130]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / instance numbers on training set')
    plt.xlabel("Instance number")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(range(df_train.shape[0]), df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[131]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Predicted - actual / CRS_ELAPSED_TIME values on test set')
    plt.xlabel("CRS_ELAPSED_TIME")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test['CRS_ELAPSED_TIME'], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[132]:


df_train_predictions = random_reg.predict(df_train_transformed)

if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Random forest : Comparison actual values / predict values on training set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.1)


# In[133]:


df_train_transformed.columns


# In[134]:


pd.set_option('display.max_rows', 200)


# In[135]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : random_reg.feature_importances_})


# In[136]:


pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].sort_values(by='Feature importance', ascending=False).cumsum()], axis=1)


# In[137]:


random_reg.feature_importances_


# In[138]:


random_reg.feature_importances_.cumsum()


# => feature importance : 

# In[139]:


random_reg.estimators_[0]


# In[140]:


'''
from sklearn.tree import export_graphviz
export_graphviz(random_reg.estimators_[0], out_file="tree.dot", rounded=True, filled=True)
'''


# In[141]:


LEARNING_CURVE_STEP_SIZE


# In[142]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    #df_test_predictions = random_reg.predict(df_test_transformed)
    #evaluate_model(random_reg, df_test_transformed, df_test[model1_label])
    plot_learning_curves(random_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], int(LEARNING_CURVE_STEP_SIZE))


# ### Parameter search and cross validation

# In[143]:


'''
%%time
from sklearn.ensemble import RandomForestRegressor

if (RECOMPUTE_GRIDSEARCH == True):
    random_reg = RandomForestRegressor(n_jobs=-1, random_state=42)

    param_grid = {
            'n_estimators':  [10, 100, 200, 500, 1000],
            'max_depth': [10, 100, 200, 500, 1000],
            'max_features': [2, 5, 50, 137],
            'max_leaf_nodes': [2, 10, 100, None],
            #'criterion': ['mse', 'mae'],
            'criterion': ['mse'],
            'n_jobs': [-1],
            'random_state': [42],
        }

    grid_search = GridSearchCV(random_reg, param_grid, cv=5, verbose=2, error_score=np.nan, scoring='neg_mean_squared_error')
    grid_search.fit(df_train_transformed, df_train[model1_label])
    
'''


# ## Linear regression with degree 8 polynomial

# In[144]:


poly = PolynomialFeaturesUnivariateAdder_DataFrame(n_degrees = 8)


# In[145]:


MODEL_1HOTALL_FEATURES_QUANTITATIVE


# In[146]:


df_train_transformed


# In[147]:


df_test_transformed


# In[148]:


df_train_transformed = poly.fit_transform(df_train_transformed, features_toadd=MODEL_1HOTALL_FEATURES_QUANTITATIVE)


# In[149]:


df_test_transformed = poly.transform(df_test_transformed)


# In[150]:


df_test_transformed


# In[151]:


df_train_transformed


# In[152]:


for col_name in df_test_transformed.columns:
    print(col_name)


# In[153]:


lin_reg = LinearRegression()

lin_reg.fit(df_train_transformed, df_train[model1_label])

df_test_predictions = lin_reg.predict(df_test_transformed)

print("Evaluation on test set :")
evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(lin_reg, df_train_transformed, df_train[model1_label])


# In[154]:


coef_feature_importances = (abs(lin_reg.coef_) / (abs(lin_reg.coef_).sum()))


# In[155]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : coef_feature_importances})


# In[156]:


pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].sort_values(by='Feature importance', ascending=False).cumsum()], axis=1)


# # New try with same features as above, but with mean sort / group by mean encoding

# In[157]:


if (DATA_LOADED == True):
    del df
    del df_train
    del df_test
    del df_train_transformed
    del df_test_transformed


# In[158]:


import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


# In[159]:


df = load_data()


# In[160]:


df


# In[161]:


all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)


# In[162]:


#df, df_train, df_test = custom_train_test_split_sample_random(df)
df, df_train, df_test = custom_train_test_split_sample(df)


# In[163]:


#df_train_transformed = preparation_pipeline_meansort_standardscale.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=None)


# In[164]:


df_train_transformed = preparation_pipeline_meansort2_stdscale.fit_transform(df_train)
#df_train_transformed = preparation_pipeline_1hotall_minmax.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN', 'UNIQUE_CARRIER'])
df_test_transformed = preparation_pipeline_meansort2_stdscale.transform(df_test)
DATA_LOADED = True
df_test_transformed.shape


# In[165]:


for feat_name in df_train_transformed.columns:
    if (feat_name in MODEL_GROUPBYMEAN2_FEATURES_QUANTITATIVE):
        fig = plt.figure()
        fig.suptitle(feat_name)
        plt.hist(df_train_transformed[feat_name], bins=50)
        plt.plot()


# ## Linear regression

# In[166]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(df_train_transformed, df_train[model1_label])

df_test_predictions = lin_reg.predict(df_test_transformed)

print("Evaluation on test set :")
evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(lin_reg, df_train_transformed, df_train[model1_label])

print('\n')

error_mean = evaluate_model_percent_mean(lin_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')


# => With new random forest (after cross validation), 80000 samples and train_test_split stratified on ARR_DELAY :
# Evaluation on test set :  
# RMSE : 27.181240312137405  
# 
# Evaluation on training set :  
# RMSE : 27.131737354138156  

# In[167]:


plt.hist(df_test_predictions, bins=50)


# In[168]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values on test set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_test[model1_label], df_test_predictions, color='coral', alpha=0.1)


# In[169]:


if (EXECUTE_INTERMEDIATE_MODELS == True):

    g = sns.jointplot(x=df_test[model1_label], y=df_test_predictions, kind='hex', color='blue', height=10)
    #sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

    g.set_axis_labels("Actual", "Predicted")
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Linear regression : comparison actual values / predict values on test set', fontsize = 16)
    
    plt.savefig('linreg_actual_vs_predicted.png', dpi=400)
    


# In[170]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / predicted values on test set')
    plt.xlabel("Predicted")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test_predictions, df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[171]:


if (EXECUTE_INTERMEDIATE_MODELS == True):

    g = sns.jointplot(x=df_test[model1_label], y=df_test[model1_label] - df_test_predictions, color='blue', kind='hex', height=10)
    #sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

    g.set_axis_labels("Actual", "Actual - Predicted")
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Linear regression : comparison residuals vs actual values on test set', fontsize = 16)
    
    plt.savefig('linreg_residuals_vs_actual.png', dpi=400)
    


# In[172]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / actual values on test set')
    plt.xlabel("Actual")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test[model1_label], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[173]:


df_train_predictions = lin_reg.predict(df_train_transformed)

if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values on training set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.1)


# In[174]:


df_test_transformed.columns


# In[175]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Predicted - actual / CRS_ELAPSED_TIME values on test set')
    plt.xlabel("CRS_ELAPSED_TIME")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test['CRS_ELAPSED_TIME'], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[176]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    for column_name in df_test_transformed.columns:
        fig = plt.figure()
        fig.suptitle('Comparison Predicted - actual / values of one feature on test set')
        plt.xlabel(column_name)
        plt.ylabel("Actual - Predicted")
        plt.scatter(df_test_transformed[column_name], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[177]:


lin_reg.coef_


# In[178]:


coef_feature_importances = (abs(lin_reg.coef_) / (abs(lin_reg.coef_).sum()))


# In[179]:


coef_feature_importances.sum()


# In[180]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : coef_feature_importances})


# In[181]:


pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].sort_values(by='Feature importance', ascending=False).cumsum()], axis=1)


# ## Random forest

# In[182]:


get_ipython().run_cell_magic('time', '', "from sklearn.ensemble import RandomForestRegressor\n\n#if (EXECUTE_INTERMEDIATE_MODELS == True):\n    #Old Random Fores before having done cross validation (done in group + mean sort encoding part)\n    #random_reg = RandomForestRegressor(n_estimators=100, max_depth=100, n_jobs=-1, random_state=42)\n    \nrandom_reg = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,\n                  max_features=4, max_leaf_nodes=None,\n                  min_impurity_decrease=0.0, min_impurity_split=None,\n                  min_samples_leaf=1, min_samples_split=2,\n                  min_weight_fraction_leaf=0.0, n_estimators=1000,\n                  n_jobs=-1, oob_score=False, random_state=42, verbose=0,\n                  warm_start=False)\n\nrandom_reg.fit(df_train_transformed, df_train[model1_label])")


# In[183]:


print("Evaluation on test set :")
evaluate_model(random_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(random_reg, df_train_transformed, df_train[model1_label])

print('\n')

error_mean = evaluate_model_percent_mean(random_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')


# With new random forest (after cross validation):  
#     With shuffle split by ARR_DELAY strategy :    
#     
# Evaluation on test set :  
# RMSE : 26.727603337896312  
# 
# 
# Evaluation on training set :  
# RMSE : 25.28427960714828  
# 
# Mean prediction error 90.0% of the time :  9.89
# 

# In[ ]:





# In[184]:


df_test_predictions = random_reg.predict(df_test_transformed)


# In[185]:


df_train_predictions = random_reg.predict(df_train_transformed)


# In[186]:


if (EXECUTE_INTERMEDIATE_MODELS == True):

    g = sns.jointplot(x=df_test[model1_label], y=df_test_predictions, kind='hex', color='blue', height=10)
    #sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)
    
    g.set_axis_labels("Actual", "Predicted")
    
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Random forest : comparison actual values / predict values on test set', fontsize = 16)
    
    plt.savefig('random_reg_actual_vs_predicted.png')
    


# In[187]:


if (EXECUTE_INTERMEDIATE_MODELS == True):

    g = sns.jointplot(x=df_test_predictions, y=df_test[model1_label] - df_test_predictions, kind='hex', color='blue', height=10)
    #sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)
    
    g.set_axis_labels("Predicted", "Actual - Predicted")
    
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Random forest : Comparison Actual - predicted / predicted values on test set', fontsize = 16)
    
    plt.savefig('random_reg_residuals_vs_predicted.png')
    


# In[188]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / predicted values on test set')
    plt.xlabel("Predicted")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test_predictions, df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[189]:


if (EXECUTE_INTERMEDIATE_MODELS == True):

    g = sns.jointplot(x=df_test[model1_label], y=df_test[model1_label] - df_test_predictions, color='blue', kind='hex', height=10)
    #sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

    g.set_axis_labels("Actual", "Actual - Predicted")
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Random forest : comparison residuals vs actual values on test set', fontsize = 16)
    
    plt.savefig('randomreg_residuals_vs_actual.png', dpi=400)


# In[190]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / actual values on test set')
    plt.xlabel("Actual label")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(df_test[model1_label], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[191]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / predicted values on training set')
    plt.xlabel("Actual label")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(df_train[model1_label], df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[192]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / instance numbers on training set')
    plt.xlabel("Instance number")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(range(df_train.shape[0]), df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[193]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Predicted - actual / CRS_ELAPSED_TIME values on test set')
    plt.xlabel("CRS_ELAPSED_TIME")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test['CRS_ELAPSED_TIME'], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[194]:


df_train_predictions = random_reg.predict(df_train_transformed)

if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Random forest : Comparison actual values / predict values on training set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.1)


# In[195]:


df_train_transformed.columns


# In[196]:


pd.set_option('display.max_rows', 200)


# In[197]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : random_reg.feature_importances_})


# In[198]:


pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].sort_values(by='Feature importance', ascending=False).cumsum()], axis=1)


# In[199]:


random_reg.feature_importances_


# In[200]:


random_reg.feature_importances_.cumsum()


# In[ ]:





# => feature importance : 

# In[201]:


random_reg.estimators_[0]


# In[202]:


'''
from sklearn.tree import export_graphviz
export_graphviz(random_reg.estimators_[0], out_file="tree.dot", rounded=True, filled=True)
'''


# In[203]:


LEARNING_CURVE_STEP_SIZE


# In[204]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    #df_test_predictions = random_reg.predict(df_test_transformed)
    #evaluate_model(random_reg, df_test_transformed, df_test[model1_label])
    plot_learning_curves(random_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], int(LEARNING_CURVE_STEP_SIZE))


# # New try with 3 quantitative features including DEP_DELAY

# In[181]:


if (DATA_LOADED == True):
    del df
    del df_train
    del df_test
    del df_train_transformed
    del df_test_transformed


# In[22]:


df = load_data()


# In[23]:


df


# In[24]:


all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)


# In[25]:


df, df_train, df_test = custom_train_test_split_sample_random(df)


# In[26]:


#df_train_transformed = preparation_pipeline_meansort_standardscale.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=None)


# In[27]:


#df_train_transformed = preparation_pipeline_3feats_stdscale.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=None)
df_train_transformed = preparation_pipeline_3feats_stdscale.fit_transform(df_train)
df_test_transformed = preparation_pipeline_3feats_stdscale.transform(df_test)
DATA_LOADED = True
df_test_transformed.shape


# In[28]:


for feat_name in df_train_transformed.columns:
    if (feat_name in MODEL1_3FEATS_QUANTITATIVE):
        fig = plt.figure()
        fig.suptitle(feat_name)
        plt.hist(df_train_transformed[feat_name], bins=50)
        plt.plot()


# ## Linear regression

# In[29]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(df_train_transformed, df_train[model1_label])

df_test_predictions = lin_reg.predict(df_test_transformed)

print("Evaluation on test set :")
evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(lin_reg, df_train_transformed, df_train[model1_label])


# In[30]:


error_mean = evaluate_model_percent_mean(lin_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')

error_mean_worst = evaluate_model_percent_worst_mean(lin_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {100 - (EVALUATION_PERCENT)*100}% of the time : {error_mean_worst : .2f}')


# In[215]:


plt.hist(df_test_predictions, bins=50)


# In[216]:


plt.hist(df_test[MODEL1_LABEL], bins=50)


# In[217]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values on test set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_test[model1_label], df_test_predictions, color='coral', alpha=0.1)


# In[218]:


if (EXECUTE_INTERMEDIATE_MODELS == True):

    g = sns.jointplot(x=df_test[model1_label], y=df_test_predictions, kind='hex', color='blue', height=10)
    #sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

    g.set_axis_labels("Actual", "Predicted")
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Model 2 : Comparison actual values / predict values on test set', fontsize = 16)
    
    plt.savefig('linreg_model2_actual_vs_predicted_training_set.png', dpi=400)
    


# In[219]:


if (EXECUTE_INTERMEDIATE_MODELS == True):

    g = sns.jointplot(x=df_test[model1_label], y=df_test[model1_label] - df_test_predictions, color='blue', kind='hex', height=10)
    #sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

    g.set_axis_labels("Actual", "Actual - Predicted")
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Linear regression : comparison residuals vs actual values on test set', fontsize = 16)
    
    plt.savefig('linreg_model2_residuals_vs_actual.png', dpi=400)
    


# In[220]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / predicted values on test set')
    plt.xlabel("Predicted")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test_predictions, df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[221]:


df_train_predictions = lin_reg.predict(df_train_transformed)

if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values on training set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.01)


# In[222]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    for column_name in df_test_transformed.columns:
        fig = plt.figure()
        fig.suptitle('Comparison Predicted - actual / values of one feature on test set')
        plt.xlabel(column_name)
        plt.ylabel("Actual - Predicted")
        plt.scatter(df_test_transformed[column_name], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[223]:


lin_reg.coef_


# In[224]:


coef_feature_importances = (abs(lin_reg.coef_) / (abs(lin_reg.coef_).sum()))


# In[225]:


coef_feature_importances.sum()


# In[226]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : coef_feature_importances})


# In[227]:


pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].sort_values(by='Feature importance', ascending=False).cumsum()], axis=1)


# In[31]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):

g = sns.jointplot(x=df_test[model1_label], y=df_test_predictions, kind='hex', color='blue', height=10)
#sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

g.set_axis_labels("Actual", "Predicted")
plt.subplots_adjust(top=0.9)
plt.suptitle('Model with departure delay : comparison actual values / predict values on test set', fontsize = 16)

plt.savefig('linreg_model2_actual_vs_predicted_test_set.png', dpi=400)


# ## Random forest

# In[32]:


get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import RandomForestRegressor\n\n#if (EXECUTE_INTERMEDIATE_MODELS == True):\nrandom_reg = RandomForestRegressor(n_estimators=200, max_depth=500, n_jobs=-1, random_state=42)\nrandom_reg.fit(df_train_transformed, df_train[model1_label])')


# In[33]:


print("Evaluation on test set :")
evaluate_model(random_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(random_reg, df_train_transformed, df_train[model1_label])


# In[48]:


error_mean = evaluate_model_percent_mean(random_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {EVALUATION_PERCENT*100}% of the time : {error_mean : .2f}')

error_mean_worst = evaluate_model_percent_worst_mean(random_reg, df_test_transformed, df_test[model1_label], 0.8)
print(f'Mean prediction error {100 - (EVALUATION_PERCENT)*100}% of the time : {error_mean_worst : .2f}')


# In[34]:


df_test_predictions = random_reg.predict(df_test_transformed)


# In[35]:


df_train_predictions = random_reg.predict(df_train_transformed)


# In[36]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / predicted values on test set')
    plt.xlabel("Predicted")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test_predictions, df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[37]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / actual values on test set')
    plt.xlabel("Actual label")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(df_test[model1_label], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[38]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / actual values on training set')
    plt.xlabel("Actual label")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(df_train[model1_label], df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[39]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / instance numbers on training set')
    plt.xlabel("Instance number")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(range(df_train.shape[0]), df_train[model1_label] - df_train_predictions, color='blue', alpha=0.01)


# In[40]:


df_train_predictions = random_reg.predict(df_train_transformed)

if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Random forest : Comparison actual values / predict values on training set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.1)


# In[41]:


df_train_transformed.columns


# In[42]:


pd.set_option('display.max_rows', 200)


# In[43]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : random_reg.feature_importances_})


# In[44]:


pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].sort_values(by='Feature importance', ascending=False).cumsum()], axis=1)


# In[49]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : random_reg.feature_importances_})

df_feature_importances_global = pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].rename(columns={'Feature importance' : 'Cumulated feature importance'}).sort_values(by='Cumulated feature importance', ascending=False).cumsum()], axis=1)

#df.style.set_properties(**{'text-align': 'center'})

df_feature_importances_global.style.format({
    'Feature importance': '{:,.2%}'.format,
    'Cumulated feature importance': '{:,.2%}'.format,
})


# In[45]:


random_reg.feature_importances_


# In[46]:


random_reg.feature_importances_.cumsum()


# In[47]:


#if (EXECUTE_INTERMEDIATE_MODELS == True):

g = sns.jointplot(x=df_test[model1_label], y=df_test_predictions, kind='hex', color='blue', height=10)
#sns.jointplot(x=df_test[model1_label], y=df_test_predictions, alpha=0.01)

g.set_axis_labels("Actual", "Predicted")
plt.subplots_adjust(top=0.9)
plt.suptitle('Model 2 with departure delay, random forest : comparison actual values / predict values on test set', fontsize = 16)

plt.savefig('randomreg_model2_actual_vs_predicted_test_set.png', dpi=400)


# In[243]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plot_learning_curves(random_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE)


# In[ ]:





# # New try with 2 quantitative features, not including DEP_DELAY

# In[244]:


if (DATA_LOADED == True):
    del df
    del df_train
    del df_test
    del df_train_transformed
    del df_test_transformed


# In[245]:


df = load_data()


# In[246]:


df


# In[247]:


all_features, model1_features, model1_label, quantitative_features, qualitative_features = identify_features(df)


# In[248]:


df, df_train, df_test = custom_train_test_split_sample_random(df)


# In[249]:


#df_train_transformed = preparation_pipeline_meansort_standardscale.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=None)


# In[250]:


#df_train_transformed = preparation_pipeline_2feats_stdscale.fit_transform(df_train, categoricalfeatures_1hotencoder__categorical_features_totransform=None)
df_train_transformed = preparation_pipeline_2feats_stdscale.fit_transform(df_train)
df_test_transformed = preparation_pipeline_2feats_stdscale.transform(df_test)
DATA_LOADED = True
df_test_transformed.shape


# In[251]:


for feat_name in df_train_transformed.columns:
    if (feat_name in MODEL1_2FEATS_QUANTITATIVE):
        fig = plt.figure()
        fig.suptitle(feat_name)
        plt.hist(df_train_transformed[feat_name], bins=50)
        plt.plot()


# ## Linear regression

# In[252]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(df_train_transformed, df_train[model1_label])

df_test_predictions = lin_reg.predict(df_test_transformed)

print("Evaluation on test set :")
evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(lin_reg, df_train_transformed, df_train[model1_label])


# In[253]:


plt.hist(df_test_predictions, bins=50)


# In[254]:


plt.hist(df_test[MODEL1_LABEL], bins=50)


# In[255]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values on test set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_test[model1_label], df_test_predictions, color='coral', alpha=0.1)


# In[256]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / predicted values on test set')
    plt.xlabel("Predicted")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test_predictions, df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[257]:


df_train_predictions = lin_reg.predict(df_train_transformed)

if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison actual values / predict values on training set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.01)


# In[258]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    for column_name in df_test_transformed.columns:
        fig = plt.figure()
        fig.suptitle('Comparison Predicted - actual / values of one feature on test set')
        plt.xlabel(column_name)
        plt.ylabel("Actual - Predicted")
        plt.scatter(df_test_transformed[column_name], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[259]:


lin_reg.coef_


# In[260]:


coef_feature_importances = (abs(lin_reg.coef_) / (abs(lin_reg.coef_).sum()))


# In[261]:


coef_feature_importances.sum()


# In[262]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : coef_feature_importances})


# In[263]:


pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].sort_values(by='Feature importance', ascending=False).cumsum()], axis=1)


# ## Random forest

# In[264]:


get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import RandomForestRegressor\n\n#if (EXECUTE_INTERMEDIATE_MODELS == True):\nrandom_reg = RandomForestRegressor(n_estimators=200, max_depth=500, n_jobs=-1, random_state=42)\nrandom_reg.fit(df_train_transformed, df_train[model1_label])')


# In[265]:


print("Evaluation on test set :")
evaluate_model(random_reg, df_test_transformed, df_test[model1_label])

print('\n')

print("Evaluation on training set :")
evaluate_model(random_reg, df_train_transformed, df_train[model1_label])


# In[266]:


df_test_predictions = random_reg.predict(df_test_transformed)


# In[267]:


df_train_predictions = random_reg.predict(df_train_transformed)


# In[268]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / predicted values on test set')
    plt.xlabel("Predicted")
    plt.ylabel("Actual - Predicted")
    plt.scatter(df_test_predictions, df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[269]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / actual values on test set')
    plt.xlabel("Actual label")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(df_test[model1_label], df_test[model1_label] - df_test_predictions, color='blue', alpha=0.1)


# In[270]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / actual values on training set')
    plt.xlabel("Actual label")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(df_train[model1_label], df_train[model1_label] - df_train_predictions, color='blue', alpha=0.1)


# In[271]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Comparison Actual - predicted / instance numbers on training set')
    plt.xlabel("Instance number")
    plt.ylabel("Actual label - Predicted label")
    plt.scatter(range(df_train.shape[0]), df_train[model1_label] - df_train_predictions, color='blue', alpha=0.01)


# In[272]:


df_train_predictions = random_reg.predict(df_train_transformed)

if (EXECUTE_INTERMEDIATE_MODELS == True):
    fig = plt.figure()
    fig.suptitle('Random forest : Comparison actual values / predict values on training set')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.scatter(df_train[model1_label], df_train_predictions, color='coral', alpha=0.1)


# In[273]:


df_train_transformed.columns


# In[274]:


pd.set_option('display.max_rows', 200)


# In[275]:


df_feature_importances = pd.DataFrame(data = {'Feature name' : df_train_transformed.columns, 'Feature importance' : random_reg.feature_importances_})


# In[276]:


pd.concat([df_feature_importances.sort_values(by='Feature importance', ascending=False),            df_feature_importances[['Feature importance']].sort_values(by='Feature importance', ascending=False).cumsum()], axis=1)


# In[277]:


random_reg.feature_importances_


# In[278]:


random_reg.feature_importances_.cumsum()


# In[279]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    plot_learning_curves(random_reg, df_train_transformed, df_test_transformed, df_train[model1_label], df_test[model1_label], LEARNING_CURVE_STEP_SIZE)


# # Annex : unused code

# In[280]:


'''from sklearn import linear_model

regressor = linear_model.SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
regressor.fit(df_transformed, df_train[model1_label])
'''


# In[281]:


'''
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(df_train_transformed, df_train[model1_label])
'''


# In[282]:


'''
df_test_predictions = svm_reg.predict(df_test_transformed)
svm_mse = mean_squared_error(df_test[model1_label], df_test_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse
'''


# In[283]:


'''
from sklearn.model_selection import StratifiedShuffleSplit

stratified_split_train = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
'''


# poly = PolynomialFeatures(degree=3)
# poly.fit(df_train_transformed)
# df_train_transformed = poly.transform(df_train_transformed)
# df_test_transformed = poly.transform(df_test_transformed)

# df_train_transformed.shape

# if (EXECUTE_INTERMEDIATE_MODELS == True):
#     lin_reg = LinearRegression()
#     lin_reg.fit(df_train_transformed, df_train[model1_label])

# if (EXECUTE_INTERMEDIATE_MODELS == True):
#     evaluate_model(lin_reg, df_test_transformed, df_test[model1_label])

# evaluate_model_MAE(lin_reg, df_test_transformed, df_test[model1_label])

# 

# # This code is now in a transformer function :
# n_degrees = 3
# n_features = df_train_transformed.shape[1]
# 
# nb_instances = df_train_transformed.shape[0]
# df_poly = np.empty((nb_instances, 0)) # Create empty array of nb_instances line and 0 features yet (we'll concatenate polynomial features to it)
# 
# progbar = tqdm(range(n_features))
# print('Adding polynomial features')
# 
# for feature_index in range(n_features):    
#     df_1feature = df_train_transformed[:,feature_index]  # Reshape 
#     
#     for n_degree in range(n_degrees):
#         df_poly = np.c_[df_poly, np.power(df_1feature, n_degree + 1)]
#     
#     progbar.update(1)
#     
# # Add bias (intercept)
# df_poly = np.c_[df_poly, np.ones((len(df_poly), 1))]  # add x0 = 1 feature

# X_train, X_test, income_train, income_test = tts( other_colums, income_column,
#                          shuffle = True, stratify = Income_column)`

# df.shape

# df_labels_discrete = pd.cut(df['ARR_DELAY'], bins=50)

# df_labels_discrete.head(50)

# df[['ARR_DELAY']]

# display_freq_table(df, ['ARR_DELAY'])

# df['ARR_DELAY'].quantile([0,1])

# df.ARR_DELAY.quantile(.01)

# df[df['ARR_DELAY'] > df.ARR_DELAY.quantile(.99)]

# df.loc[(df['ARR_DELAY'] < df.ARR_DELAY.quantile(.01)) | (df['ARR_DELAY'] > df.ARR_DELAY.quantile(.99)) , :]

# ((df['ARR_DELAY'] < df.ARR_DELAY.quantile(.01)) | (df['ARR_DELAY'] > df.ARR_DELAY.quantile(.99))).index

# ((df['ARR_DELAY'] < df.ARR_DELAY.quantile(.01)) | (df['ARR_DELAY'] > df.ARR_DELAY.quantile(.99)))

# df.shape

# df_labels_discrete.shape

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(df, df[model1_label], test_size=0.1, random_state=42, shuffle = True, stratify = df_labels_discrete)

# from sklearn.model_selection import train_test_split
# X_train, X_test = train_test_split(df, test_size=0.1, random_state=42, shuffle = True, stratify = df_labels_discrete)

# X_test

# split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
# for train_index, test_index in split.split(df, df_labels_discrete):
#     strat_train_set = df.loc[train_index]
#     strat_test_set = df.loc[test_index]

# df_labels_discrete.value_counts()

# df[['DEST']]

# df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

# poly = ColumnTransformer([
#                                 ('poly', PolynomialFeatures(degree=2), [0, 1, 2, 3, 4, 5, 6])     
#                                 ], remainder='passthrough', sparse_threshold=1)
# 
# poly.fit(df_train_transformed, df_train[model1_label])

# '''
# #Too slow
# 
# from sklearn.ensemble import RandomForestRegressor
# 
# if (EXECUTE_INTERMEDIATE_MODELS == True):
#     random_reg = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
#     random_reg.fit(df_train_transformed, df_train[model1_label])
#     
# '''

# '''
# if (EXECUTE_INTERMEDIATE_MODELS == True):
#     df_test_predictions = random_reg.predict(df_test_transformed)
#     evaluate_model(random_reg, df_test_transformed, df_test[model1_label])
# '''

# # Add bias :
# # Bias has been removed: its linear regression coeficient was 0
# '''
# df_train_transformed = np.c_[np.ones((len(df_train_transformed), 1)), df_train_transformed]  # add x0 = 1 to each instance
# df_test_transformed = np.c_[np.ones((len(df_test_transformed), 1)), df_test_transformed]  # add x0 = 1 to each instance
# '''

# In[ ]:





# '''
# # Commented out because memory error
# polynomial_reg = Pipeline([('poly', PolynomialFeatures(degree=3)),
#                           ('linear', LinearRegression(fit_intercept=False))])
# 
# polynomial_reg.fit(df_train_transformed, df_train[model1_label])
# '''

# %%time
# from sklearn.ensemble import RandomForestRegressor
# 
# if (EXECUTE_INTERMEDIATE_MODELS == True):
#     random_reg = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
#                       max_features=4, max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, n_estimators=1000,
#                       n_jobs=-1, oob_score=False, random_state=42, verbose=0,
#                       warm_start=False)
#     
#     for i in range(10):
#         print(f'Training {i}')
#         df_train_residuals = df_train[model1_label] - df_train_predictions
#         max_residual = df_train_residuals.abs().max()
#         sample_weights = (max_residual - df_train_residuals.abs()) / max_residual
#         
#         df_train_residuals_mean = df_train_residuals.mean()
#         print(f'Mean of residuals : {df_train_residuals_mean}')
#         
#         random_reg.fit(df_train_transformed, df_train[model1_label], sample_weights)
# 
#         df_train_predictions = random_reg.predict(df_train_transformed)
# 
# 
# 

# for feat_indice in range(df_train_transformed.shape[1]):
#     fig = plt.figure()
#     plt.hist(df_train_transformed.iloc[:, feat_indice], bins=50)
