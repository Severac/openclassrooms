#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ5 : Online Retail dataset :  modelisation notebook 

# In[207]:


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

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import entropy

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


# # Preparation pipeline : model with bow features + TotalPricePerMonth + BoughtTopValueProduct + HasEverCancelled

# In[11]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[12]:


df_train = df_train_ori
df_test = df_test_ori


# In[13]:


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


# In[14]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[15]:


df_test = preparation_pipeline.transform(df_test)


# In[16]:


df_train


# In[17]:


df_train.info()


# # Explained variance of bag of words features

# In[18]:


from display_factorial import *
importlib.reload(sys.modules['display_factorial'])


# In[19]:


display_scree_plot(preparation_pipeline['dimensionality_reductor'].reductor)


# # 2D visualization

# In[20]:


pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[21]:


X_transformed[:,1]


# In[22]:


print('Binarisation of color categories')
bins = [-np.inf,df_train['TotalPricePerMonth'].quantile(0.25),        df_train['TotalPricePerMonth'].quantile(0.50),        df_train['TotalPricePerMonth'].quantile(0.75),        df_train['TotalPricePerMonth'].quantile(1)]

labels = [0, 1, 2, 3]

df_score_cat_train = pd.cut(df_train['TotalPricePerMonth'], bins=bins, labels=labels)


bins = [-np.inf,df_test['TotalPricePerMonth'].quantile(0.25),        df_test['TotalPricePerMonth'].quantile(0.50),        df_test['TotalPricePerMonth'].quantile(0.75),        df_test['TotalPricePerMonth'].quantile(1)]

labels = [0, 1, 2, 3]

df_score_cat_test = pd.cut(df_test['TotalPricePerMonth'], bins=bins, labels=labels)


# In[23]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[24]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = df_train['TotalPricePerMonth'],
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients.html') 


# In[25]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, test set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# # Model with only bag of word features

# In[26]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[27]:


df_train = df_train_ori
df_test = df_test_ori


# In[28]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='PCA', n_dim=200)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[29]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[30]:


df_test = preparation_pipeline.transform(df_test)


# In[31]:


df_train


# In[32]:


pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[33]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[34]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = df_score_cat_train,
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_onlybow.html') 


# # Model with bow features + TotalPricePerMonth

# In[35]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[36]:


df_train = df_train_ori
df_test = df_test_ori


# In[37]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='PCA', n_dim=200)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[38]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[39]:


df_test = preparation_pipeline.transform(df_test)


# In[40]:


df_train


# In[41]:


pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[ ]:





# In[42]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW + TotalPricePerMonth')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[43]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = df_score_cat_train,
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_onlybow.html') 


# # Model with bow features + TotalPricePerMonth + HasEverCancelled

# In[44]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[45]:


df_train = df_train_ori
df_test = df_test_ori


# In[46]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'TotalPricePerMonth', 'HasEverCancelled'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='PCA', n_dim=200)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[47]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[48]:


df_test = preparation_pipeline.transform(df_test)


# In[49]:


df_train


# In[50]:


pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[ ]:





# In[51]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW + TotalPricePerMonth + HasEverCancelled')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[52]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = df_score_cat_train,
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_onlybow.html') 


# # Model with all features and NCA

# In[53]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[54]:


df_train = df_train_ori
df_test = df_test_ori


# In[55]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='NCA', n_dim=200, labels_featurename='TotalPricePerMonth')),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[56]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[57]:


df_train


# In[58]:


df_train.info()


# In[59]:


pca = PCA(n_components=2,random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[60]:


X_transformed[:,1]


# In[61]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[62]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = df_train['TotalPricePerMonth'],
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_nca_allfeats.html') 


# # Model with all features and NCA, final representation with tSNE

# In[63]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[64]:


df_train = df_train_ori
df_test = df_test_ori


# In[65]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='NCA', n_dim=200, labels_featurename='TotalPricePerMonth')),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[66]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[67]:


df_train


# In[68]:


df_train.info()


# In[69]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[70]:


X_transformed[:,1]


# In[71]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[72]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = df_train['TotalPricePerMonth'],
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_nca_allfeats_final_tsne.html') 


# # Model with all features and PCA, final representation with tSNE

# In[73]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[74]:


df_train = df_train_ori
df_test = df_test_ori


# In[75]:


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


# In[76]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[77]:


df_train


# In[78]:


df_train.info()


# In[79]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[80]:


X_transformed[:,1]


# In[81]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[82]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = df_train['TotalPricePerMonth'],
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_pca_allfeats_final_tsne.html') 


# # Model with all features and tSNE, final representation with tSNE (best)

# In[83]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[84]:


df_train = df_train_ori
df_test = df_test_ori


# In[85]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='TSNE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[86]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[87]:


df_train


# In[88]:


df_train.info()


# In[89]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[90]:


X_transformed[:,1]


# In[91]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[92]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = df_train['TotalPricePerMonth'],
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_allfeats_final_tsne.html') 


# # Model with all features and tSNE, BoW not apart from the rest, final representation with tSNE

# In[112]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[113]:


df_train = df_train_ori
df_test = df_test_ori


# In[114]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized', 'TotalPricePerMonth', 'HasEverCancelled', 'BoughtTopValueProduct'], \
                                                        algorithm_to_use='TSNE', n_dim=2)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[115]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[116]:


df_train


# In[88]:


df_train.info()


# In[89]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[90]:


X_transformed[:,1]


# In[91]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[92]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = df_train['TotalPricePerMonth'],
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_allfeats_final_tsne.html') 


# # Model with all features and tSNE, final representation with tSNE 3D

# In[93]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[94]:


df_train = df_train_ori
df_test = df_test_ori


# In[95]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='TSNE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[96]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[97]:


df_train


# In[98]:


df_train.info()


# In[99]:


tsne = TSNE(n_components=3, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[100]:


X_transformed[:,1]


# In[101]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter3d(x = X_transformed[:,0], y = X_transformed[:,1], z = X_transformed[:,2],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = df_train['TotalPricePerMonth'],
                    )


layout = go.Layout(title = 'Représentation des clients en 3 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_allfeats_final_tsne_3d.html') 


# # Model with bow features + TotalPricePerMonth and NCA, final representation with tSNE

# In[102]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[103]:


df_train = df_train_ori
df_test = df_test_ori


# In[104]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='NCA', n_dim=200, labels_featurename='TotalPricePerMonth')),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[105]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[106]:


df_train


# In[107]:


df_train.info()


# In[108]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[109]:


X_transformed[:,1]


# In[110]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[111]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = df_train['TotalPricePerMonth'],
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_nca_bowandtotalpricepermonthfeats_final_tsne.html') 


# # Model with tSNE, then clustering algorithm KMeans

# In[138]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[123]:


df_train = df_train_ori
df_test = df_test_ori


# In[124]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='TSNE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[125]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[126]:


df_train


# In[129]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[130]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[131]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[135]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[163]:


gini_mean_score_per_k_train = []

for model in kmeans_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    gini_sum = 0
    for unique_label in unique_labels:
        gini_sum += gini(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    gini_sum = gini_sum / len(unique_labels)
    
    gini_mean_score_per_k_train.append(gini_sum)

    
gini_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    gini_sum = 0
    for unique_label in unique_labels:
        gini_sum += gini(df_test['TotalPricePerMonth'][labels_test == unique_label].to_numpy())
        
    gini_sum = gini_sum / len(unique_labels)
    
    gini_mean_score_per_k_test.append(gini_sum)    
    


# In[210]:


entropy_mean_score_per_k_train = []

for model in kmeans_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)

    
entropy_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_test['TotalPricePerMonth'][labels_test == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_test.append(entropy_sum)    
    


# In[133]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[136]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[214]:


print('Gini before clustering :')
gini(df_train['TotalPricePerMonth'].to_numpy())


# In[166]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), gini_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean gini score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[167]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), gini_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean gini score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[209]:


print('Entropy before clustering :')
entropy(df_train['TotalPricePerMonth'])


# In[215]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[216]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with tSNE, then clustering algorithm Ward
# No visualisation on test set because AgglomerativeClustering has no predict function, only fit_predict

# In[168]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[170]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[171]:


gini_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    gini_sum = 0
    for unique_label in unique_labels:
        gini_sum += gini(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    gini_sum = gini_sum / len(unique_labels)
    
    gini_mean_score_per_k_train.append(gini_sum)


    


# In[217]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[172]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[173]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), gini_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean gini score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[221]:


print('Entropy before clustering :')
entropy(df_train['TotalPricePerMonth'])


# In[219]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with tSNE, then clustering algorithm Ward, distance threshold
# No visualisation on test set because AgglomerativeClustering has no predict function, only fit_predict

# In[253]:


np.unique(AgglomerativeClustering(distance_threshold=1, n_clusters=None, affinity='euclidean').fit(df_train).labels_)


# In[271]:


clusterer_ward_per_thr = [AgglomerativeClustering(distance_threshold=thr, n_clusters=None, affinity='euclidean').fit(df_train) for thr in reversed(range(0,12))]


# In[234]:


np.unique(clusterer_ward.labels_)


# In[262]:


clusterer_ward_per_thr


# In[272]:


entropy_mean_score_per_k_train = []

for model in clusterer_ward_per_thr[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[276]:


plt.figure(figsize=(8, 3))
plt.plot(range(1,12), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$Ward threshold$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # => Around 50 clusters => entropy of TotalPrice around 4.5

# # Annex

# ## Display some data

# In[26]:


df_nocancel = df[df['InvoiceNo'].str.startswith('C') == False]
df_nocancel.reset_index(inplace=True)

df_gbproduct = df_nocancel[['StockCode', 'TotalPrice']].groupby('StockCode').sum()['TotalPrice']


# In[27]:


df_nocancel.head(2)


# In[28]:


df_nocancel.info()


# In[29]:


invoice_dates = pd.to_datetime(df_nocancel["InvoiceDate"], format="%Y-%m-%d ")


# In[30]:


invoice_dates = pd.to_datetime(df_nocancel["InvoiceDate"])


# In[31]:


np.maximum((pd.to_datetime('2011-12-09 12:50:00') - invoice_dates) / (np.timedelta64(1, "M")), 1)[123456]


# In[ ]:





# In[32]:


df_gbcustom_firstorder = df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()


# In[33]:


df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate']


# In[34]:


(   pd.to_datetime('2011-12-09 12:50:00')   - pd.to_datetime(df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
)\
  / (np.timedelta64(1, "M"))


# In[35]:


# Number of months between first order date and last date of the dataset
series_gbclient_nbmonths = np.maximum((
   (
   pd.to_datetime('2011-12-09 12:50:00')\
   - pd.to_datetime(df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
   )\
    / (np.timedelta64(1, "M"))
), 1)


# In[36]:


df_nocancel[['CustomerID', ]]


# In[37]:


df_gbcustom_firstorder


# In[38]:


df_nocancel[df_nocancel['CustomerID'] == '18281'].sort_values(by='InvoiceDate', ascending=True)


# In[39]:


invoice_dates[2000:2010]


# In[40]:


df_nocancel.loc[2000:2010,'InvoiceDate']


# In[ ]:





# In[41]:


df_nocancel.loc[100000:100010,'InvoiceMonth']


# In[42]:


df[df['InvoiceNo'].str.startswith('C') == True]['CustomerID'].unique()


# In[43]:


# Product codes that contain chars instead of numbers
df[df['StockCode'].str.isalpha()]['StockCode'].unique()


# # For debug / test (clean code is in functions.py)

# In[44]:


df_train = df_train_ori
df_test = df_test_ori


# In[45]:


df_train.head(6)


# In[46]:


df_train


# In[47]:


df_train_nocancel = df_train[df_train['InvoiceNo'].str.startswith('C') == False]
df_train_nocancel.reset_index(inplace=True)


# In[48]:


feat_list = ['CustomerID', 'TotalPrice']
feat_list_bow = [col for col in df_train_nocancel if col.startswith('DescriptionNormalized_')]
feat_list.extend(feat_list_bow)


# In[49]:


feat_list


# In[50]:


df_train_gbcust_nocancel = df_train_nocancel[feat_list].groupby('CustomerID').sum()


# In[51]:


df_train_gbcust_nocancel[feat_list_bow] = df_train_gbcust_nocancel[feat_list_bow].clip(upper=1)


# In[52]:


df_train_gbcust_nocancel


# In[53]:


# Number of months between first order date and last date of the dataset
series_train_gbclient_nbmonths = np.maximum((
   (
   pd.to_datetime('2011-12-09 12:50:00')\
   - pd.to_datetime(df_train_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
   )\
    / (np.timedelta64(1, "M"))
), 1)


# In[54]:


series_train_gbclient_nbmonths


# In[55]:


df_train_gbcust_nocancel['TotalPrice'] 


# In[56]:


df_train_gbcust_nocancel['TotalPrice'] = df_train_gbcust_nocancel['TotalPrice'] / series_train_gbclient_nbmonths


# In[57]:


df_train_gbcust_nocancel


# In[58]:


df_train


# In[59]:


custid_cancelled = df_train[df_train['InvoiceNo'].str.startswith('C') == True]['CustomerID'].unique()


# In[ ]:





# In[ ]:




