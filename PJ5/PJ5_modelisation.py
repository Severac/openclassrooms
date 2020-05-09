#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ5 : Online Retail dataset :  modelisation notebook 

# In[422]:


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

MODEL_FEATURES=['InvoiceNo', 'InvoiceDate', 'CustomerID', 'TotalPrice', 'DescriptionNormalized', 'InvoiceMonth', 'StockCode']

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

# In[423]:


df = load_data(DATA_PATH_FILE_INPUT)


# In[424]:


df.info()


# In[425]:


df, df_train, df_test = custom_train_test_split_sample(df, 'TotalPrice')


# In[426]:


df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)


# In[427]:


df_train.info()


# In[428]:


df_train_ori = df_train.copy(deep=True)
df_test_ori = df_test.copy(deep=True)


# # Top value products (must be saved with the model, and passed to it)

# In[429]:


TOP_VALUE_THRESHOLD = 10
top_value_products = df_gbproduct.sort_values(ascending=False).head(TOP_VALUE_THRESHOLD).index  # Get top value products


# In[430]:


df[df['StockCode'].str.isalpha()]['StockCode'].unique()


# In[436]:


df[df['StockCode'] == 'CRUK']['TotalPrice'].sum()


# In[437]:


df[df['StockCode'] == 'D']


# In[417]:


df_gbproduct[df_gbproduct.index.isin(df[df['StockCode'].str.isalpha()]['StockCode'].unique())]


# In[418]:


df[df['StockCode'].isin(['POST', 'D', 'M', 'PADS', 'DOT', 'CRUK'])].shape


# In[419]:


df.shape


# In[420]:


df.drop(index=df[df['StockCode'].isin(['POST', 'D', 'M', 'PADS', 'DOT', 'CRUK'])].index, axis=0, inplace=True)


# In[421]:


df


# In[372]:


df[df['StockCode'] == 'DOT']


# In[348]:


top_value_products


# In[ ]:





# In[358]:


df_nocancel[df_nocancel['StockCode'] == 'POST']['TotalPrice'].sum()


# In[360]:


df_gbproduct.sort_values(ascending=False).head(10)


# In[350]:


pd.set_option('display.max_rows', 200)
df[df['StockCode'] == 'POST'].sample(200)


# # Preparation pipeline

# In[334]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[335]:


df_train = df_train_ori
df_test = df_test_ori


# In[336]:


preparation_pipeline = Pipeline([
    ('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products))
    
    # Ajouter le log scale du TotalPrice et le MinMaxScale à la fin
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
    
    #('hour_extractor', HHMM_to_HH()),
    #('data_converter', HHMM_to_Minutes()),
    #('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()), 
    
    
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=MODEL_1HOTALL_FEATURES_QUANTITATIVE)),
])


# In[337]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[338]:


df_train


# In[339]:


df_train[df_train['BoughtTopValueProduct'] == 1]


# In[344]:


for s in df[df['CustomerID'] == '12350']['StockCode'].unique():
    if s in top_value_products:
        print(f'{s} : yes')


# In[234]:


df_train.info()


# # Annex

# ## Display some data

# In[407]:


df_nocancel = df[df['InvoiceNo'].str.startswith('C') == False]
df_nocancel.reset_index(inplace=True)

df_gbproduct = df_nocancel[['StockCode', 'TotalPrice']].groupby('StockCode').sum()['TotalPrice']


# In[408]:


df_nocancel.head(2)


# In[409]:


df_nocancel.info()


# In[410]:


invoice_dates = pd.to_datetime(df_nocancel["InvoiceDate"], format="%Y-%m-%d ")


# In[411]:


invoice_dates = pd.to_datetime(df_nocancel["InvoiceDate"])


# In[412]:


np.maximum((pd.to_datetime('2011-12-09 12:50:00') - invoice_dates) / (np.timedelta64(1, "M")), 1)[123456]


# In[ ]:





# In[413]:


df_gbcustom_firstorder = df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()


# In[414]:


df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate']


# In[45]:


(   pd.to_datetime('2011-12-09 12:50:00')   - pd.to_datetime(df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
)\
  / (np.timedelta64(1, "M"))


# In[52]:


# Number of months between first order date and last date of the dataset
series_gbclient_nbmonths = np.maximum((
   (
   pd.to_datetime('2011-12-09 12:50:00')\
   - pd.to_datetime(df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
   )\
    / (np.timedelta64(1, "M"))
), 1)


# In[ ]:


df_nocancel[['CustomerID', ]]


# In[75]:


df_gbcustom_firstorder


# In[48]:


df_nocancel[df_nocancel['CustomerID'] == '18281'].sort_values(by='InvoiceDate', ascending=True)


# In[14]:


invoice_dates[2000:2010]


# In[15]:


df_nocancel.loc[2000:2010,'InvoiceDate']


# In[ ]:





# In[16]:


df_nocancel.loc[100000:100010,'InvoiceMonth']


# In[17]:


df[df['InvoiceNo'].str.startswith('C') == True]['CustomerID'].unique()


# In[361]:


# Product codes that contain chars instead of numbers
df[df['StockCode'].str.isalpha()]['StockCode'].unique()


# # For debug / test (clean code is in functions.py)

# In[189]:


df_train = df_train_ori
df_test = df_test_ori


# In[190]:


df_train.head(6)


# In[191]:


df_train


# In[192]:


df_train_nocancel = df_train[df_train['InvoiceNo'].str.startswith('C') == False]
df_train_nocancel.reset_index(inplace=True)


# In[193]:


feat_list = ['CustomerID', 'TotalPrice']
feat_list_bow = [col for col in df_train_nocancel if col.startswith('DescriptionNormalized_')]
feat_list.extend(feat_list_bow)


# In[194]:


feat_list


# In[195]:


df_train_gbcust_nocancel = df_train_nocancel[feat_list].groupby('CustomerID').sum()


# In[196]:


df_train_gbcust_nocancel[feat_list_bow] = df_train_gbcust_nocancel[feat_list_bow].clip(upper=1)


# In[197]:


df_train_gbcust_nocancel


# In[198]:


# Number of months between first order date and last date of the dataset
series_train_gbclient_nbmonths = np.maximum((
   (
   pd.to_datetime('2011-12-09 12:50:00')\
   - pd.to_datetime(df_train_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
   )\
    / (np.timedelta64(1, "M"))
), 1)


# In[199]:


series_train_gbclient_nbmonths


# In[200]:


df_train_gbcust_nocancel['TotalPrice'] 


# In[201]:


df_train_gbcust_nocancel['TotalPrice'] = df_train_gbcust_nocancel['TotalPrice'] / series_train_gbclient_nbmonths


# In[202]:


df_train_gbcust_nocancel


# In[203]:


df_train


# In[204]:


custid_cancelled = df_train[df_train['InvoiceNo'].str.startswith('C') == True]['CustomerID'].unique()


# In[ ]:





# In[ ]:




