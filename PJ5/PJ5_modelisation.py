#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ5 : Online Retail dataset :  modelisation notebook 

# In[1]:


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

#MODEL_FEATURES=['InvoiceNo', 'InvoiceDate', 'CustomerID', 'TotalPrice', 'DescriptionNormalized', 'InvoiceMonth', 'StockCode']
MODEL_CLIENT_FEATURES = ['TotalPricePerMonth', 'DescriptionNormalized', 'HasEverCancelled', 'BoughtTopValueProduct' ]

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

# In[2]:


df = load_data(DATA_PATH_FILE_INPUT)


# In[3]:


df.info()


# In[4]:


df, df_train, df_test = custom_train_test_split_sample(df, 'TotalPrice')


# In[5]:


df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)


# In[6]:


df_train.info()


# In[7]:


df_train_ori = df_train.copy(deep=True)
df_test_ori = df_test.copy(deep=True)


# # Top value products (must be saved with the model, and passed to it)

# In[8]:


df_nocancel = df_train[df_train['InvoiceNo'].str.startswith('C') == False]
df_nocancel.reset_index(inplace=True)

df_gbproduct = df_nocancel[['StockCode', 'TotalPrice']].groupby('StockCode').sum()['TotalPrice']


# In[9]:


TOP_VALUE_PRODUCT_THRESHOLD = 20
top_value_products = df_gbproduct.sort_values(ascending=False).head(TOP_VALUE_PRODUCT_THRESHOLD).index  # Get top value products


# In[10]:


top_value_products


# # Preparation pipeline

# In[61]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[62]:


df_train = df_train_ori
df_test = df_test_ori


# In[63]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='PCA', n_dim=200)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[64]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[65]:


df_test = preparation_pipeline.transform(df_test)


# In[66]:


df_train


# In[67]:


df_train.info()


# # Explained variance of bag of words features

# In[68]:


from display_factorial import *
importlib.reload(sys.modules['display_factorial'])


# In[69]:


display_scree_plot(preparation_pipeline['dimensionality_reductor'].reductor)


# # 2D visualization

# In[74]:


pca = PCA(n_components=2)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[75]:


X_transformed[:,1]


# In[76]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1])
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[77]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, test set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1])
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# # Annex

# ## Display some data

# In[24]:


df_nocancel = df[df['InvoiceNo'].str.startswith('C') == False]
df_nocancel.reset_index(inplace=True)

df_gbproduct = df_nocancel[['StockCode', 'TotalPrice']].groupby('StockCode').sum()['TotalPrice']


# In[25]:


df_nocancel.head(2)


# In[26]:


df_nocancel.info()


# In[27]:


invoice_dates = pd.to_datetime(df_nocancel["InvoiceDate"], format="%Y-%m-%d ")


# In[28]:


invoice_dates = pd.to_datetime(df_nocancel["InvoiceDate"])


# In[29]:


np.maximum((pd.to_datetime('2011-12-09 12:50:00') - invoice_dates) / (np.timedelta64(1, "M")), 1)[123456]


# In[ ]:





# In[30]:


df_gbcustom_firstorder = df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()


# In[31]:


df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate']


# In[32]:


(   pd.to_datetime('2011-12-09 12:50:00')   - pd.to_datetime(df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
)\
  / (np.timedelta64(1, "M"))


# In[33]:


# Number of months between first order date and last date of the dataset
series_gbclient_nbmonths = np.maximum((
   (
   pd.to_datetime('2011-12-09 12:50:00')\
   - pd.to_datetime(df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
   )\
    / (np.timedelta64(1, "M"))
), 1)


# In[34]:


df_nocancel[['CustomerID', ]]


# In[35]:


df_gbcustom_firstorder


# In[36]:


df_nocancel[df_nocancel['CustomerID'] == '18281'].sort_values(by='InvoiceDate', ascending=True)


# In[37]:


invoice_dates[2000:2010]


# In[38]:


df_nocancel.loc[2000:2010,'InvoiceDate']


# In[ ]:





# In[39]:


df_nocancel.loc[100000:100010,'InvoiceMonth']


# In[40]:


df[df['InvoiceNo'].str.startswith('C') == True]['CustomerID'].unique()


# In[41]:


# Product codes that contain chars instead of numbers
df[df['StockCode'].str.isalpha()]['StockCode'].unique()


# # For debug / test (clean code is in functions.py)

# In[42]:


df_train = df_train_ori
df_test = df_test_ori


# In[43]:


df_train.head(6)


# In[44]:


df_train


# In[45]:


df_train_nocancel = df_train[df_train['InvoiceNo'].str.startswith('C') == False]
df_train_nocancel.reset_index(inplace=True)


# In[46]:


feat_list = ['CustomerID', 'TotalPrice']
feat_list_bow = [col for col in df_train_nocancel if col.startswith('DescriptionNormalized_')]
feat_list.extend(feat_list_bow)


# In[47]:


feat_list


# In[48]:


df_train_gbcust_nocancel = df_train_nocancel[feat_list].groupby('CustomerID').sum()


# In[49]:


df_train_gbcust_nocancel[feat_list_bow] = df_train_gbcust_nocancel[feat_list_bow].clip(upper=1)


# In[50]:


df_train_gbcust_nocancel


# In[51]:


# Number of months between first order date and last date of the dataset
series_train_gbclient_nbmonths = np.maximum((
   (
   pd.to_datetime('2011-12-09 12:50:00')\
   - pd.to_datetime(df_train_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
   )\
    / (np.timedelta64(1, "M"))
), 1)


# In[52]:


series_train_gbclient_nbmonths


# In[53]:


df_train_gbcust_nocancel['TotalPrice'] 


# In[54]:


df_train_gbcust_nocancel['TotalPrice'] = df_train_gbcust_nocancel['TotalPrice'] / series_train_gbclient_nbmonths


# In[55]:


df_train_gbcust_nocancel


# In[56]:


df_train


# In[57]:


custid_cancelled = df_train[df_train['InvoiceNo'].str.startswith('C') == True]['CustomerID'].unique()


# In[ ]:





# In[ ]:




