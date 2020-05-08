#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ5 : Online Retail dataset :  modelisation notebook 

# In[118]:


get_ipython().run_line_magic('matplotlib', 'inline')

#%load_ext autoreload  # Autoreload has a bug : when you modify function in source code and run again, python kernel hangs :(
#%autoreload 2

import datetime as dt

import sys, importlib

from functions import *
importlib.reload(sys.modules['functions'])

import pandas as pd

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

import datetime as dt

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

DATA_PATH = os.path.join("datasets", "onlineretail")
DATA_PATH = os.path.join(DATA_PATH, "out")

DATA_PATH_FILE_INPUT = os.path.join(DATA_PATH, "OnlineRetail_transformed.csv")


ALL_FEATURES = []

MODEL_FEATURES=['InvoiceNo', 'InvoiceDate', 'CustomerID', 'TotalPrice', 'DescriptionNormalized', 'InvoiceMonth']

plt.rcParams["figure.figsize"] = [16,9] # Taille par défaut des figures de matplotlib

import seaborn as sns
sns.set()

#import common_functions

####### Paramètres pour sauver et restaurer les modèles :
import pickle
####### Paramètres à changer par l'utilisateur selon son besoin :

RECOMPUTE_GRIDSEARCH = True  # CAUTION : computation is several hours long
SAVE_GRID_RESULTS = False # If True : grid results object will be saved to pickle files that have GRIDSEARCH_FILE_PREFIX
LOAD_GRID_RESULTS = False # If True : grid results object will be loaded from pickle files that have GRIDSEARCH_FILE_PREFIX
                          # Grid search results are loaded with full samples (SAMPLED_DATA must be False)

'''
RECOMPUTE_GRIDSEARCH = True  # CAUTION : computation is several hours long
SAVE_GRID_RESULTS = True # If True : grid results object will be saved to pickle files that have GRIDSEARCH_FILE_PREFIX
LOAD_GRID_RESULTS = False # If True : grid results object will be loaded from pickle files that have GRIDSEARCH_FILE_PREFIX
'''
#GRIDSEARCH_CSV_FILE = 'grid_search_results.csv'

GRIDSEARCH_FILE_PREFIX = 'grid_search_results_'

EXECUTE_INTERMEDIATE_MODELS = True # If True: every intermediate model (which results are manually analyzed in the notebook) will be executed


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
API_MODEL_PICKLE_FILE = 'API_model_PJ5.pickle'


# # Load data

# In[119]:


df = load_data(DATA_PATH_FILE_INPUT)


# In[120]:


df.info()


# In[121]:


df, df_train, df_test = custom_train_test_split_sample(df, 'TotalPrice')


# In[122]:


df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)


# In[123]:


df_train.info()


# In[124]:


df_train_ori = df_train.copy(deep=True)
df_test_ori = df_test.copy(deep=True)


# # Display some data

# In[146]:


df_nocancel = df[df['InvoiceNo'].str.startswith('C') == False]
df_nocancel.reset_index(inplace=True)

df_gbproduct = df_nocancel[['StockCode', 'TotalPrice']].groupby('StockCode').sum()['TotalPrice']


# In[126]:


df_nocancel.head(2)


# In[127]:


df_nocancel.info()


# In[128]:


invoice_dates = pd.to_datetime(df_nocancel["InvoiceDate"], format="%Y-%m-%d ")


# In[129]:


invoice_dates = pd.to_datetime(df_nocancel["InvoiceDate"])


# In[130]:


np.maximum((pd.to_datetime('2011-12-09 12:50:00') - invoice_dates) / (np.timedelta64(1, "M")), 1)[123456]


# In[131]:


invoice_dates[2000:2010]


# In[132]:


df_nocancel.loc[2000:2010,'InvoiceDate']


# In[133]:


df_nocancel.loc[100000:100010,'InvoiceMonth']


# In[139]:


df[df['InvoiceNo'].str.startswith('C') == True]['CustomerID'].unique()


# # Top value products (must be saved with the model, and passed to it)

# In[155]:


top_value_products = df_gbproduct.sort_values(ascending=False).head(200).index


# In[156]:


top_value_products


# # Preparation pipeline

# In[25]:


importlib.reload(sys.modules['functions'])


# In[26]:


df_train = df_train_ori
df_test = df_test_ori


# In[27]:


preparation_pipeline = Pipeline([
    ('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel())
    
    # Ajouter le log scale du TotalPrice et le MinMaxScale à la fin
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
    
    #('hour_extractor', HHMM_to_HH()),
    #('data_converter', HHMM_to_Minutes()),
    #('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()), 
    
    
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=MODEL_1HOTALL_FEATURES_QUANTITATIVE)),
])


# In[28]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[30]:


df_train.head(6)

