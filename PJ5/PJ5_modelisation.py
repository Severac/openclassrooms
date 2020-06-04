#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ5 : Online Retail dataset :  modelisation notebook 

# In[6]:


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
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import entropy

from sklearn.feature_selection import RFE


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

'''
RECOMPUTE_GRIDSEARCH = True  # CAUTION : computation is several hours long
SAVE_GRID_RESULTS = False # If True : grid results object will be saved to pickle files that have GRIDSEARCH_FILE_PREFIX
LOAD_GRID_RESULTS = False # If True : grid results object will be loaded from pickle files that have GRIDSEARCH_FILE_PREFIX
                          # Grid search results are loaded with full samples (SAMPLED_DATA must be False)
'''


RECOMPUTE_GRIDSEARCH = False  # CAUTION : computation is several hours long
SAVE_GRID_RESULTS = False # If True : grid results object will be saved to pickle files that have GRIDSEARCH_FILE_PREFIX
LOAD_GRID_RESULTS = True # If True : grid results object will be loaded from pickle files that have GRIDSEARCH_FILE_PREFIX

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

# In[7]:


df = load_data(DATA_PATH_FILE_INPUT)


# In[8]:


df.info()


# In[9]:


df, df_train, df_test = custom_train_test_split_sample(df, 'TotalPrice')


# In[10]:


df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)


# In[11]:


df_train.info()


# In[12]:


df_train_ori = df_train.copy(deep=True)
df_test_ori = df_test.copy(deep=True)


# # Top value products (must be saved with the model, and passed to it)

# In[13]:


df_nocancel = df_train[df_train['InvoiceNo'].str.startswith('C') == False]
df_nocancel.reset_index(inplace=True)

df_gbproduct = df_nocancel[['StockCode', 'TotalPrice']].groupby('StockCode').sum()['TotalPrice']


# In[14]:


TOP_VALUE_PRODUCT_THRESHOLD = 20
top_value_products = df_gbproduct.sort_values(ascending=False).head(TOP_VALUE_PRODUCT_THRESHOLD).index  # Get top value products


# In[15]:


top_value_products


# # Preparation pipeline : model with bow features + TotalPricePerMonth + BoughtTopValueProduct + HasEverCancelled

# In[16]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[17]:


df_train = df_train_ori
df_test = df_test_ori


# In[18]:


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


# In[19]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[20]:


df_test = preparation_pipeline.transform(df_test)


# In[21]:


df_train


# In[22]:


df_train.info()


# In[23]:


series_total_price_per_month_train = df_train['TotalPricePerMonth']
series_total_price_per_month_test = df_test['TotalPricePerMonth']

series_hasevercancelled_train = df_train['HasEverCancelled']
series_hasevercancelled_test = df_test['HasEverCancelled']

series_boughttopvalueproduct_train = df_train['BoughtTopValueProduct']
series_boughttopvalueproduct_test = df_test['BoughtTopValueProduct']


# # Explained variance of bag of words features

# In[24]:


from display_factorial import *
importlib.reload(sys.modules['display_factorial'])


# In[25]:


display_scree_plot(preparation_pipeline['dimensionality_reductor'].reductor)


# # 2D visualization

# In[26]:


pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[27]:


X_transformed[:,1]


# In[28]:


print('Binarisation of color categories')
bins = [-np.inf,df_train['TotalPricePerMonth'].quantile(0.25),        df_train['TotalPricePerMonth'].quantile(0.50),        df_train['TotalPricePerMonth'].quantile(0.75),        df_train['TotalPricePerMonth'].quantile(1)]

labels = [0, 1, 2, 3]

df_score_cat_train = pd.cut(df_train['TotalPricePerMonth'], bins=bins, labels=labels)


bins = [-np.inf,df_test['TotalPricePerMonth'].quantile(0.25),        df_test['TotalPricePerMonth'].quantile(0.50),        df_test['TotalPricePerMonth'].quantile(0.75),        df_test['TotalPricePerMonth'].quantile(1)]

labels = [0, 1, 2, 3]

df_score_cat_test = pd.cut(df_test['TotalPricePerMonth'], bins=bins, labels=labels)


# In[29]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[30]:


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


# In[31]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, test set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# # Generate bow colors

# In[32]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[33]:


df_train = df_train_ori
df_test = df_test_ori


# In[34]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'RfmScore'])),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='NCA', n_dim=1, labels_featurename='RfmScore')),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[35]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[36]:


df_train.loc[:, 0].to_numpy()


# In[37]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train.loc[:, 0].to_numpy().reshape(-1,1))
                for k in range(1, 50)]


# In[38]:


labels_test_per_k = [model.predict(df_test.loc[:, 0].to_numpy().reshape(-1,1)) for model in kmeans_per_k[1:]]


# In[39]:


silhouette_scores = [silhouette_score(df_train.loc[:, 0].to_numpy().reshape(-1,1), model.labels_)
                     for model in kmeans_per_k[1:]]


# In[40]:


silhouette_scores_test = [silhouette_score(df_test.loc[:, 0].to_numpy().reshape(-1,1), labels_test) for labels_test in labels_test_per_k]


# In[41]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[42]:


entropy_mean_score_per_k_train = []

for model in kmeans_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['RfmScore'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)

    
entropy_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_test['RfmScore'][labels_test == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_test.append(entropy_sum)    
    


# In[43]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[44]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[45]:


bow_labels_train = kmeans_per_k[10].labels_


# In[46]:


bow_labels_test = kmeans_per_k[10].predict(df_test.loc[:, 0].to_numpy().reshape(-1,1))


# # Model with only bag of word features, PCA

# In[185]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[186]:


df_train = df_train_ori
df_test = df_test_ori


# In[187]:


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


# In[188]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[189]:


df_test = preparation_pipeline.transform(df_test)


# In[190]:


df_train


# In[191]:


pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)

'''
tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)
'''


# In[192]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[193]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, test set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[194]:


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


# ## Clustering test

# In[195]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[196]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[197]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[198]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[199]:


entropy_mean_score_per_k_train = []

for model in kmeans_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)

    
entropy_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_test[labels_test == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_test.append(entropy_sum)    
    


# In[200]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[201]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[202]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[203]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[204]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## 2nd clustering : ward

# In[205]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[206]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[207]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[208]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[209]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[210]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with only bag of word features, TSNE

# In[211]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[212]:


df_train = df_train_ori
df_test = df_test_ori


# In[213]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='TSNE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[214]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[215]:


df_test = preparation_pipeline.transform(df_test)


# In[216]:


df_train


# In[217]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[218]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[219]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, test set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[220]:


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

py.offline.plot(fig, filename='clusters_plot_clients_onlybow_TSNE.html') 


# ## Clustering test

# In[221]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[222]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[223]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[224]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[225]:


entropy_mean_score_per_k_train = []

for model in kmeans_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)

    
entropy_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_test[labels_test == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_test.append(entropy_sum)    
    


# In[226]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[227]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[228]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[229]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[230]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## 2nd clustering : ward

# In[231]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[232]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[233]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[234]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[235]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[236]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with only bow features and TotalPricePerMonth, TSNE

# In[293]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[294]:


df_train = df_train_ori
df_test = df_test_ori


# In[295]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='TSNE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[296]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[297]:


df_test = preparation_pipeline.transform(df_test)


# In[298]:


df_train


# In[299]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[300]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[301]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, test set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[302]:


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

py.offline.plot(fig, filename='clusters_plot_clients_onlybow_TSNE.html') 


# ## Clustering test

# In[303]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[304]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[305]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[306]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[307]:


entropy_mean_score_per_k_train = []

for model in kmeans_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)

    
entropy_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_test[labels_test == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_test.append(entropy_sum)    
    


# In[308]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[309]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[310]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[311]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[312]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## 2nd clustering : ward

# In[313]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[314]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[315]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[316]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[317]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[318]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


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


# # Model with all features and tSNE, final representation with tSNE (2ND BEST)

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


# # Model with all features *except* BoW, and tSNE, final representation with tSNE

# In[237]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[238]:


df_train = df_train_ori
df_test = df_test_ori


# In[239]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    #('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    #('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
    #                                                    algorithm_to_use='TSNE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[240]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[252]:


df_train


# In[242]:


df_train.info()


# In[243]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[244]:


X_transformed[:,1]


# In[245]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, all feats except bow, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[253]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, all feats except bow, test set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[260]:


df_train.iloc[:, 0]


# In[261]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter3d(x = df_train.iloc[:,0], y = df_train.iloc[:,1], z = df_train.iloc[:, 2],
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

py.offline.plot(fig, filename='clusters_plot_clients_sne_allfeatsexceptbow_3d.html') 


# In[265]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter3d(x = df_train.iloc[:,0], y = df_train.iloc[:,1], z = df_train.iloc[:, 2],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_train['BoughtTopValueProduct']),
                    text = df_train['TotalPricePerMonth'],
                    )


layout = go.Layout(title = 'Représentation des clients en 3 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_allfeatsexceptbow_3d_color_topvalue.html') 


# In[266]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter3d(x = df_train.iloc[:,0], y = df_train.iloc[:,1], z = df_train.iloc[:, 2],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_train['HasEverCancelled']),
                    text = df_train['TotalPricePerMonth'],
                    )


layout = go.Layout(title = 'Représentation des clients en 3 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_allfeatsexceptbow_3d_color_cancel.html') 


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

# # Prepation model with LLE reduce to 200, then clustering algorithm Ward

# In[280]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[281]:


df_train = df_train_ori
df_test = df_test_ori


# In[282]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='LLE', n_dim=200, n_neighbors=10)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[283]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[284]:


df_test = preparation_pipeline.transform(df_test)


# In[285]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test) for model in clusterer_per_k[1:]]


# In[286]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[291]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[287]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[292]:


entropy_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_test['TotalPricePerMonth'][labels_test == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_test.append(entropy_sum)    


# In[288]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[293]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[289]:


print('Entropy before clustering :')
entropy(df_train['TotalPricePerMonth'])


# In[290]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[294]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Prepation model with LLE reduce to 3, then clustering algorithm Ward

# In[45]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[46]:


df_train = df_train_ori
df_test = df_test_ori


# In[47]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='LLE', n_dim=3, n_neighbors=10)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[48]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[49]:


df_test = preparation_pipeline.transform(df_test)


# In[50]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[53]:


labels_test_per_k = [model.predict(df_test) for model in clusterer_per_k[1:]]


# In[54]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[52]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[55]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[ ]:


entropy_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_test['TotalPricePerMonth'][labels_test == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_test.append(entropy_sum)    


# In[56]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[57]:


print('Entropy before clustering :')
entropy(df_train['TotalPricePerMonth'])


# In[58]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Preparation model BoW feats only, then LLE reduce to 200, then KMeans and Ward

# In[80]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[81]:


df_train = df_train_ori
df_test = df_test_ori


# In[82]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='LLE', n_dim=200, n_neighbors=10)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[83]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[84]:


df_test = preparation_pipeline.transform(df_test)


# In[85]:


df_train


# In[86]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[87]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[88]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[89]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[90]:


entropy_mean_score_per_k_train = []

for model in kmeans_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)

    
entropy_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_test[labels_test == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_test.append(entropy_sum)    
    


# In[91]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[92]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[93]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[94]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[95]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## 2nd clustering : Ward

# In[96]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[97]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[99]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[100]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[101]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[102]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Preparation model BoW feats only, then LLE reduce to 3 then KMeans and Ward

# In[126]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[104]:


df_train = df_train_ori
df_test = df_test_ori


# In[105]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='LLE', n_dim=3, n_neighbors=10)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[106]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[107]:


df_test = preparation_pipeline.transform(df_test)


# In[108]:


df_train


# In[109]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[110]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[111]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[112]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[113]:


entropy_mean_score_per_k_train = []

for model in kmeans_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)

    
entropy_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_test[labels_test == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_test.append(entropy_sum)    
    


# In[114]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[115]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[116]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[117]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[118]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## 2nd clustering : ward

# In[119]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[120]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[121]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[122]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[123]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[124]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Plot representation

# In[131]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter3d(x = df_train.loc[:,0], y = df_train.loc[:,1], z = df_train.loc[:,2],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = series_total_price_per_month_train,
                    )


layout = go.Layout(title = 'Représentation des clients en 3 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_lle_bowfeats_3d.html') 


# # Model with 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency', 'HasEverCancelled', 'BoughtTopValueProduct', 'DescriptionNormalized', and TSNE, with KMeans (INTERESTING)

# In[42]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[43]:


df_train = df_train_ori
df_test = df_test_ori


# In[44]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency', 'HasEverCancelled', 'BoughtTopValueProduct', 'DescriptionNormalized'])),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='TSNE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[45]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[46]:


rfm_scores_train = get_rfm_scores(df_train)


# In[47]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[48]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[49]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[50]:


rfm_scores_train_colors


# In[51]:


rfm_scores_train


# In[52]:


df_train


# In[53]:


df_train.info()


# In[54]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[55]:


X_transformed[:,1]


# In[56]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[57]:


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

py.offline.plot(fig, filename='clusters_plot_clients_sne_allfeatswithRFM_final_tsne.html') 


# In[58]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[59]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=rfm_scores_train_colors),
                    text = rfm_scores_train,
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_allfeatswithRFM_color_RFM_final_tsne.html') 


# In[60]:


## Add bow coloration


# In[62]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, BoW colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=bow_labels_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# ## RFM and bow (without BoughtTopValueProduct and HasEverCancelled)

# In[168]:


df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 0, 1, 2]]


# In[169]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 0, 1, 2]])
X_test_transformed = tsne.fit_transform(df_test[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 0, 1, 2]])


# In[170]:


X_transformed[:,1]


# In[171]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[172]:


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

py.offline.plot(fig, filename='clusters_plot_clients_sne_bowandRfm_final_tsne.html') 


# In[173]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# ## RFM and bow (without HasEverCancelled, but WITH BoughtTopValueProduct)

# In[210]:


df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 0, 1, 2]]


# In[211]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 0, 1, 2]])
X_test_transformed = tsne.fit_transform(df_test[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 0, 1, 2]])


# In[212]:


X_transformed[:,1]


# In[213]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[214]:


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

py.offline.plot(fig, filename='clusters_plot_clients_sne_bowandRfmandboughttopvalue_final_tsne.html') 


# In[215]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# ## RFM only (without bow, BoughtTopValueProduct and HasEverCancelled)

# In[174]:


df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth']]


# In[204]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth']])
X_test_transformed = tsne.fit_transform(df_test[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth']])


# In[205]:


X_transformed[:,1]


# In[206]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[207]:


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

py.offline.plot(fig, filename='clusters_plot_clients_sne_rfmOnly_final_tsne.html') 


# In[208]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1RFM without b')
plt.ylabel("Axe 2")

#plt.yscale('log')


# # RFM with 'TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled' (GOOD)

# In[187]:


df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled']]


# In[188]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled']])
X_test_transformed = tsne.fit_transform(df_test[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled']])


# In[189]:


X_transformed[:,1]


# In[190]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[191]:


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

py.offline.plot(fig, filename='clusters_plot_clients_sne_rfmBoughtTopValueProductsHasEverCancelled_withoutBow_final_tsne.html') 


# In[197]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# # Model with all bow features and RFM score (not individual feats), and TSNE, with KMeans

# In[120]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[121]:


df_train = df_train_ori
df_test = df_test_ori


# In[122]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'RfmScore'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='TSNE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[123]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[124]:


rfm_scores_train = df_train['RfmScore']


# In[125]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[126]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[127]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[128]:


rfm_scores_train_colors


# In[129]:


rfm_scores_train


# In[130]:


df_train


# In[131]:


df_train.info()


# In[132]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[133]:


X_transformed[:,1]


# In[137]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[138]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=rfm_scores_train_colors),
                    text = rfm_scores_train,
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_bowwithRFMscore_color_RFM_final_tsne.html') 


# # Model with 'DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore' (concat), and TSNE (BEST)

# In[27]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[28]:


df_train = df_train_ori
df_test = df_test_ori


# In[29]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='TSNE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[30]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[31]:


rfm_scores_train = df_train['RfmScore']


# In[32]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[33]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[34]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[35]:


rfm_scores_train_colors


# In[36]:


rfm_scores_train


# In[37]:


df_train


# In[38]:


df_train.info()


# In[39]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[40]:


X_transformed[:,1]


# In[41]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[42]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=rfm_scores_train_colors),
                    text = rfm_scores_train,
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_allFeatswithRFMscore_color_RFM_final_tsne.html') 


# # Model with only bow features without RfmScore, coloration with RfmScore

# In[43]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[44]:


df_train = df_train_ori
df_test = df_test_ori


# In[45]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='TSNE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[46]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[31]:


#rfm_scores_train = df_train['RfmScore']  # we reuse rfm_scores_train calculated on above model


# In[47]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[48]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[49]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[50]:


rfm_scores_train_colors


# In[51]:


rfm_scores_train


# In[52]:


df_train


# In[53]:


df_train.info()


# In[54]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[55]:


X_transformed[:,1]


# In[56]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[57]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=rfm_scores_train_colors),
                    text = rfm_scores_train,
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_onlybow_color_RFM_final_tsne.html') 


# # Model with 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore', coloration with RfmScore (INTERESTING)

# In[331]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[332]:


df_train = df_train_ori
df_test = df_test_ori


# In[333]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    #('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    #('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
    #                                                    algorithm_to_use='TSNE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[334]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[335]:


#rfm_scores_train = df_train['RfmScore']  # we reuse rfm_scores_train calculated on above model


# In[336]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[337]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[338]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[339]:


rfm_scores_train_colors


# In[340]:


rfm_scores_train


# In[341]:


df_train


# In[342]:


df_train.info()


# In[343]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[344]:


X_transformed[:,1]


# In[345]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[346]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=rfm_scores_train_colors),
                    text = rfm_scores_train,
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_allFeatsExceptBow_RfmScore_color_RFM_final_tsne.html') 


# # Model with all features and RFM score (not individual RFM feats), and LLE

# In[89]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[90]:


df_train = df_train_ori
df_test = df_test_ori


# In[91]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='LLE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[92]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[93]:


rfm_scores_train = df_train['RfmScore']


# In[94]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[95]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[96]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[97]:


rfm_scores_train_colors


# In[98]:


rfm_scores_train


# In[99]:


df_train


# In[100]:


df_train.info()


# In[101]:


'''
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_transformed = lle.fit_transform(df_train)
X_test_transformed = lle.fit_transform(df_test)
'''
tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[102]:


X_transformed[:,1]


# In[103]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[42]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=rfm_scores_train_colors),
                    text = rfm_scores_train,
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_allFeatswithRFMscore_color_RFM_final_lle.html') 


# # Model with all features and RFM score (not individual RFM feats), and Isomap

# In[109]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[110]:


df_train = df_train_ori
df_test = df_test_ori


# In[111]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='ISOMAP', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[112]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[113]:


rfm_scores_train = df_train['RfmScore']


# In[114]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[115]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[116]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[117]:


rfm_scores_train_colors


# In[118]:


rfm_scores_train


# In[119]:


df_train


# In[120]:


df_train.info()


# In[122]:


'''
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_transformed = lle.fit_transform(df_train)
X_test_transformed = lle.fit_transform(df_test)
'''
'''
tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)
'''
isomap = Isomap(n_components=2)
X_transformed = isomap.fit_transform(df_train)
X_test_transformed = isomap.fit_transform(df_test)


# In[123]:


X_transformed[:,1]


# In[124]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[125]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=rfm_scores_train_colors),
                    text = rfm_scores_train,
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_allFeatswithRFMscore_color_RFM_final_isomap.html') 


# # Model with all features and RFM score (cocncat) (not individual RFM feats), and NCA (BEST)

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
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='NCA', n_dim=3, labels_featurename='RfmScore')),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[115]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[116]:


rfm_scores_train = df_train['RfmScore']


# In[117]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[118]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[119]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[120]:


rfm_scores_train_colors


# In[121]:


rfm_scores_train


# In[131]:


df_train


# In[123]:


df_train.info()


# In[124]:


'''
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_transformed = lle.fit_transform(df_train)
X_test_transformed = lle.fit_transform(df_test)
'''

tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


'''
isomap = Isomap(n_components=2)
X_transformed = isomap.fit_transform(df_train)
X_test_transformed = isomap.fit_transform(df_test)
'''


# In[125]:


X_transformed[:,1]


# In[126]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[127]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, TotalPricePerMonth colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[130]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=rfm_scores_train_colors),
                    #text = rfm_scores_train,
                    #text = [('Bought top value product' if (boughttopvalueproduct == 1) else 'dit NOT buy top value product') for boughttopvalueproduct in df_train['BoughtTopValueProduct']],
                    text = list(map(str, zip('RFM: ' + rfm_scores_train.astype(str),\
                                             'BoughtTopValueProduct: ' + df_train['BoughtTopValueProduct'].astype(str),\
                                              'HasEverCancelled: '  + df_train['HasEverCancelled'].astype(str),\
                                              'Bow0: ' + df_train[0].astype(str),\
                                              'Bow1: ' + df_train[1].astype(str),\
                                              'Bow2: ' + df_train[2].astype(str),\
                                            ))\
                                        )
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_NCA_allFeatswithRFMscore_color_RFM_final_tsnelast.html') 


# # Model with all features and RFM score (SUM) (not individual RFM feats), and NCA (BEST)

# In[67]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[68]:


df_train = df_train_ori
df_test = df_test_ori


# In[69]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='NCA', n_dim=3, labels_featurename='RfmScore')),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[70]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[71]:


rfm_scores_train = df_train['RfmScore']


# In[72]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[73]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[74]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[75]:


rfm_scores_train_colors


# In[76]:


rfm_scores_train


# In[77]:


df_train


# In[78]:


df_train.info()


# In[79]:


'''
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_transformed = lle.fit_transform(df_train)
X_test_transformed = lle.fit_transform(df_test)
'''

tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


'''
isomap = Isomap(n_components=2)
X_transformed = isomap.fit_transform(df_train)
X_test_transformed = isomap.fit_transform(df_test)
'''


# In[80]:


X_transformed[:,1]


# In[81]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[82]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, TotalPricePerMonth colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[83]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=rfm_scores_train_colors),
                    #text = rfm_scores_train,
                    #text = [('Bought top value product' if (boughttopvalueproduct == 1) else 'dit NOT buy top value product') for boughttopvalueproduct in df_train['BoughtTopValueProduct']],
                    text = list(map(str, zip('RFM: ' + rfm_scores_train.astype(str),\
                                             'BoughtTopValueProduct: ' + df_train['BoughtTopValueProduct'].astype(str),\
                                              'HasEverCancelled: '  + df_train['HasEverCancelled'].astype(str),\
                                              'Bow0: ' + df_train[0].astype(str),\
                                              'Bow1: ' + df_train[1].astype(str),\
                                              'Bow2: ' + df_train[2].astype(str),\
                                            ))\
                                        )
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_NCA_allFeatswithRFMscoreSUM_color_RFM_final_tsnelast.html') 


# In[88]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=bow_labels_train),
                    #text = rfm_scores_train,
                    #text = [('Bought top value product' if (boughttopvalueproduct == 1) else 'dit NOT buy top value product') for boughttopvalueproduct in df_train['BoughtTopValueProduct']],
                    text = list(map(str, zip('RFM: ' + rfm_scores_train.astype(str),\
                                             'BoughtTopValueProduct: ' + df_train['BoughtTopValueProduct'].astype(str),\
                                              'HasEverCancelled: '  + df_train['HasEverCancelled'].astype(str),\
                                              'Bow0: ' + df_train[0].astype(str),\
                                              'Bow1: ' + df_train[1].astype(str),\
                                              'Bow2: ' + df_train[2].astype(str),\
                                            ))\
                                        )
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_NCA_allFeatswithRFMscoreSUM_color_BoW_final_tsnelast.html') 


# In[84]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, BoW colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=bow_labels_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# # Model with all features except RFM score, and TSNE, colored by RFM

# In[95]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[96]:


df_train = df_train_ori
df_test = df_test_ori


# In[97]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='TSNE', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[98]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[99]:


#rfm_scores_train = df_train['RfmScore'] # rfm_scores_train value has been got from code above


# In[100]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[101]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[102]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[103]:


rfm_scores_train_colors


# In[104]:


rfm_scores_train


# In[105]:


df_train


# In[106]:


df_train.info()


# In[107]:


'''
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_transformed = lle.fit_transform(df_train)
X_test_transformed = lle.fit_transform(df_test)
'''

tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


'''
isomap = Isomap(n_components=2)
X_transformed = isomap.fit_transform(df_train)
X_test_transformed = isomap.fit_transform(df_test)
'''


# In[108]:


X_transformed[:,1]


# In[109]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[110]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, TotalPricePerMonth colored, training set')

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
                    marker=dict(color=rfm_scores_train_colors),
                    #text = rfm_scores_train,
                    #text = [('Bought top value product' if (boughttopvalueproduct == 1) else 'dit NOT buy top value product') for boughttopvalueproduct in df_train['BoughtTopValueProduct']],
                    text = list(map(str, zip('rfm: ' + rfm_scores_train.astype(str), 'BoughtTopValueProduct: ' + df_train['BoughtTopValueProduct'].astype(str),\
                                              'HasEverCancelled: '  + df_train['HasEverCancelled'].astype(str),\
                                            ))\
                                        )
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_TSNE_allFeatswithOUTRFMscore_color_RFM_final_tsnelast.html') 


# # Model with 'DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore' (concat) with NCA up to 200 then KMeans then NCA to visualize clusters   (GOOD, ONLY NCA not TSNE)

# In[27]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[28]:


df_train = df_train_ori
df_test = df_test_ori


# In[29]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='NCA', n_dim=3, labels_featurename='RfmScore')),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[30]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[31]:


rfm_scores_train = df_train['RfmScore']
rfm_scores_test = df_test['RfmScore']


# In[32]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())
unique_rfm_scores_test = np.sort(rfm_scores_test.unique())


# In[33]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1

rfm_dict_colors_test = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores_test:
    rfm_dict_colors_test[unique_rfm_score] = cnt
    cnt += 1    


# In[34]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])
rfm_scores_test_colors = rfm_scores_test.apply(lambda x : rfm_dict_colors_test[x])


# In[35]:


rfm_scores_train_colors


# In[36]:


rfm_scores_test_colors


# In[37]:


rfm_scores_train


# In[38]:


df_train


# In[39]:


df_train.info()


# In[40]:


'''
df_train_rfmscore_distances = pairwise_distances(df_train['RfmScore'].to_numpy().reshape(-1, 1))
df_test_rfmscore_distances = pairwise_distances(df_test['RfmScore'].to_numpy().reshape(-1, 1))
'''


# In[41]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[42]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[43]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[44]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[45]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[46]:


entropy_mean_score_per_k_train = []

for model in kmeans_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['RfmScore'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)

    
entropy_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_test['RfmScore'][labels_test == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_test.append(entropy_sum)    
    


# In[47]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[48]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[49]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[50]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[51]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## Reduce and visualize

# In[414]:


nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
X_transformed = nca.fit_transform(df_train, pd.cut(df_train['RfmScore'], bins=range(1,10), right=True).astype(str).tolist())
X_test_transformed = nca.transform(df_test)


# In[415]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[416]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, cluster label colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[324]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_transformed)
                for k in range(1, 50)]


# In[325]:


labels_test_per_k = [model.predict(X_test_transformed) for model in kmeans_per_k[1:]]


# In[326]:


silhouette_scores = [silhouette_score(X_transformed, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[327]:


silhouette_scores_test = [silhouette_score(X_test_transformed, labels_test) for labels_test in labels_test_per_k]


# In[329]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[330]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[309]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[310]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[311]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with 'DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore' (SUM) with NCA up to 200 then KMeans then NCA to visualize clusters   (GOOD, ONLY NCA not TSNE)

# In[79]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[43]:


df_train = df_train_ori
df_test = df_test_ori


# In[44]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='NCA', n_dim=3, labels_featurename='RfmScore')),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[45]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[46]:


rfm_scores_train = df_train['RfmScore']
rfm_scores_test = df_test['RfmScore']


# In[47]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())
unique_rfm_scores_test = np.sort(rfm_scores_test.unique())


# In[48]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1

rfm_dict_colors_test = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores_test:
    rfm_dict_colors_test[unique_rfm_score] = cnt
    cnt += 1    


# In[49]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])
rfm_scores_test_colors = rfm_scores_test.apply(lambda x : rfm_dict_colors_test[x])


# In[50]:


rfm_scores_train_colors


# In[51]:


rfm_scores_test_colors


# In[52]:


rfm_scores_train


# In[53]:


df_train


# In[54]:


df_train.info()


# In[55]:


'''
df_train_rfmscore_distances = pairwise_distances(df_train['RfmScore'].to_numpy().reshape(-1, 1))
df_test_rfmscore_distances = pairwise_distances(df_test['RfmScore'].to_numpy().reshape(-1, 1))
'''


# In[56]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[57]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[58]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[59]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[60]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[61]:


entropy_mean_score_per_k_train = []

for model in kmeans_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['RfmScore'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)

    
entropy_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_test['RfmScore'][labels_test == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_test.append(entropy_sum)    
    


# In[62]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[63]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[64]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[65]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[66]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## Reduce and visualize

# In[67]:


nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
X_transformed = nca.fit_transform(df_train, pd.cut(df_train['RfmScore'], bins=range(1,10), right=True).astype(str).tolist())
X_test_transformed = nca.transform(df_test)


# In[68]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[69]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, cluster label colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[70]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_transformed)
                for k in range(1, 50)]


# In[71]:


labels_test_per_k = [model.predict(X_test_transformed) for model in kmeans_per_k[1:]]


# In[72]:


silhouette_scores = [silhouette_score(X_transformed, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[73]:


silhouette_scores_test = [silhouette_score(X_test_transformed, labels_test) for labels_test in labels_test_per_k]


# In[74]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[75]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[76]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[77]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[81]:


df_test


# In[151]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[152]:


clusterer = Clusterer(n_clusters=11, algorithm_to_use='WARD')
clusterer.fit(df_train)


# In[162]:


cluster_labels_test = clusterer.predict(df_test)


# In[163]:


cluster_labels_train = clusterer.predict(df_train)


# In[184]:


df_train.loc[0, :]


# In[188]:


df_train[df_train.index == 0]


# In[183]:


df_train[df_train.index=df_train.loc[0, :]]


# In[189]:


instance_prediction = clusterer.predict(df_train[df_train.index == 0])


# In[190]:


instance_prediction


# In[164]:


len(cluster_labels_test)


# In[167]:


df_train.loc[0, :]


# In[165]:


cluster_labels_test


# In[157]:


df_train


# In[158]:


cluster_labels_train


# In[159]:


clusterer.clusterer.labels_


# In[160]:


clusterer.score(df_train)


# In[161]:


silhouette_scores_test


# # Model with 'TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled' with NCA up to 200 then KMeans then NCA to visualize clusters

# In[351]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[352]:


df_train = df_train_ori
df_test = df_test_ori


# In[353]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth'])),
    #('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
    #                                                    algorithm_to_use='NCA', n_dim=3, labels_featurename='RfmScore')),
    #('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[354]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[355]:


#rfm_scores_train = df_train['RfmScore']  # Got from code above
#rfm_scores_test = df_test['RfmScore']


# In[356]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())
unique_rfm_scores_test = np.sort(rfm_scores_test.unique())


# In[357]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1

rfm_dict_colors_test = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores_test:
    rfm_dict_colors_test[unique_rfm_score] = cnt
    cnt += 1    


# In[358]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])
rfm_scores_test_colors = rfm_scores_test.apply(lambda x : rfm_dict_colors_test[x])


# In[359]:


rfm_scores_train_colors


# In[360]:


rfm_scores_test_colors


# In[361]:


rfm_scores_train


# In[362]:


df_train


# In[363]:


df_train.info()


# In[364]:


'''
df_train_rfmscore_distances = pairwise_distances(df_train['RfmScore'].to_numpy().reshape(-1, 1))
df_test_rfmscore_distances = pairwise_distances(df_test['RfmScore'].to_numpy().reshape(-1, 1))
'''


# In[365]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[366]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[367]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[368]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[370]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[371]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## Reduce and visualize

# In[373]:


nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
X_transformed = nca.fit_transform(df_train, pd.cut(rfm_scores_train, bins=range(1,10), right=True).astype(str).tolist())
X_test_transformed = nca.transform(df_test)


# In[374]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[375]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_transformed)
                for k in range(1, 50)]


# In[376]:


labels_test_per_k = [model.predict(X_test_transformed) for model in kmeans_per_k[1:]]


# In[377]:


silhouette_scores = [silhouette_score(X_transformed, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[378]:


silhouette_scores_test = [silhouette_score(X_test_transformed, labels_test) for labels_test in labels_test_per_k]


# In[379]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[380]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency' then KMeans then 3D visualisation

# In[429]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[430]:


df_train = df_train_ori
df_test = df_test_ori


# In[431]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    #('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    #('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
    #                                                    algorithm_to_use='NCA', n_dim=3, labels_featurename='RfmScore')),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[432]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[433]:


#rfm_scores_train = df_train['RfmScore']  # Got from above
#rfm_scores_test = df_test['RfmScore']


# In[434]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())
unique_rfm_scores_test = np.sort(rfm_scores_test.unique())


# In[435]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1

rfm_dict_colors_test = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores_test:
    rfm_dict_colors_test[unique_rfm_score] = cnt
    cnt += 1    


# In[436]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])
rfm_scores_test_colors = rfm_scores_test.apply(lambda x : rfm_dict_colors_test[x])


# In[437]:


rfm_scores_train_colors


# In[438]:


rfm_scores_test_colors


# In[439]:


rfm_scores_train


# In[440]:


df_train


# In[441]:


df_train.info()


# In[442]:


'''
df_train_rfmscore_distances = pairwise_distances(df_train['RfmScore'].to_numpy().reshape(-1, 1))
df_test_rfmscore_distances = pairwise_distances(df_test['RfmScore'].to_numpy().reshape(-1, 1))
'''


# In[443]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[444]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[445]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[446]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[447]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[449]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[450]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## Visualize

# In[452]:


df_train


# In[454]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter3d(x = df_train.loc[:,'Recency'], y = df_train.loc[:,'TotalPricePerMonth'], z = df_train.loc[:,'TotalQuantityPerMonth'],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=rfm_scores_train_colors),
                    #text = rfm_scores_train,
                    #text = [('Bought top value product' if (boughttopvalueproduct == 1) else 'dit NOT buy top value product') for boughttopvalueproduct in df_train['BoughtTopValueProduct']],
                    text = list(map(str, zip(
                                              '0: ' + df_train['Recency'].astype(str),\
                                              '1: ' + df_train['TotalPricePerMonth'].astype(str),\
                                              '2: ' + df_train['TotalQuantityPerMonth'].astype(str),\
                                            ))\
                                        )
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_NCA_3RFMfeats_color_RFM.html') 


# In[456]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter3d(x = df_train.loc[:,'Recency'], y = df_train.loc[:,'TotalPricePerMonth'], z = df_train.loc[:,'TotalQuantityPerMonth'],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_),
                    #text = rfm_scores_train,
                    #text = [('Bought top value product' if (boughttopvalueproduct == 1) else 'dit NOT buy top value product') for boughttopvalueproduct in df_train['BoughtTopValueProduct']],
                    text = list(map(str, zip(
                                              '0: ' + df_train['Recency'].astype(str),\
                                              '1: ' + df_train['TotalPricePerMonth'].astype(str),\
                                              '2: ' + df_train['TotalQuantityPerMonth'].astype(str),\
                                            ))\
                                        )
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_NCA_3RFMfeats_color_clusterlabels.html') 


# In[416]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, cluster label colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[324]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_transformed)
                for k in range(1, 50)]


# In[325]:


labels_test_per_k = [model.predict(X_test_transformed) for model in kmeans_per_k[1:]]


# In[326]:


silhouette_scores = [silhouette_score(X_transformed, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[327]:


silhouette_scores_test = [silhouette_score(X_test_transformed, labels_test) for labels_test in labels_test_per_k]


# In[329]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[330]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[309]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[310]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Correlations

# In[70]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[71]:


df_train = df_train_ori
df_test = df_test_ori


# In[72]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'RfmScore', 'BoughtTopValueProduct', 'HasEverCancelled'])),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='NCA', n_dim=3, labels_featurename='RfmScore')),
    #('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[73]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[75]:


corr_matrix = df_train.corr()


# In[76]:


corr_matrix


# In[77]:


plt.title('Corrélation entre les features')
sns.heatmap(corr_matrix, 
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.columns, cmap='coolwarm' ,center=0.20)


# In[27]:


'''
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols("BMI ~ 0 + 1 + 2", data=df_train).fit()
#print model.params
#print model.summary()
'''


# # Generate bow colors

# In[65]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[66]:


df_train = df_train_ori
df_test = df_test_ori


# In[67]:


preparation_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'RfmScore'])),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='NCA', n_dim=1, labels_featurename='RfmScore')),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[68]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[79]:


df_train.loc[:, 0].to_numpy()


# In[81]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train.loc[:, 0].to_numpy().reshape(-1,1))
                for k in range(1, 50)]


# In[82]:


labels_test_per_k = [model.predict(df_test.loc[:, 0].to_numpy().reshape(-1,1)) for model in kmeans_per_k[1:]]


# In[83]:


silhouette_scores = [silhouette_score(df_train.loc[:, 0].to_numpy().reshape(-1,1), model.labels_)
                     for model in kmeans_per_k[1:]]


# In[84]:


silhouette_scores_test = [silhouette_score(df_test.loc[:, 0].to_numpy().reshape(-1,1), labels_test) for labels_test in labels_test_per_k]


# In[45]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[85]:


entropy_mean_score_per_k_train = []

for model in kmeans_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['RfmScore'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)

    
entropy_mean_score_per_k_test = []

for labels_test in labels_test_per_k:
    unique_labels = np.unique(labels_test)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_test['RfmScore'][labels_test == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_test.append(entropy_sum)    
    


# In[86]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[87]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[90]:


bow_labels_train = kmeans_per_k[10].labels_


# In[91]:


bow_labels_test = kmeans_per_k[10].predict(df_test.loc[:, 0].to_numpy().reshape(-1,1))


# # Pipeline with clustering, and GridSearch to select the best model

# A GridSearch using pipeline is implemented separately, in .py files  
# To run the gridsearch, open a python3 console and :
# 
# 1/ Launch a python3 console  
# 2/ Run PJ5_GridSearch_prerequisites.py : exec(open('PJ5_GridSearch_prerequisites.py').read())  
# 3/ Run PJ5_GridSearch1.py : exec(open('PJ5_GridSearch1.py').read())  
# 
# Before 3, you can edit source code of PJ5_GridSearch1.py, and uncomment the GridSearch code you want to run.

# # Visualize best models found via GridSearch
# Generate pickle files with PJ5_GridSearch1.py before running this part  (see §above : Pipeline with clustering, and GridSearch to select the best model) 

# In[53]:


from functions import *
importlib.reload(sys.modules['functions'])
from functions import *


# ## First model

# In[115]:


model_agregate = AgregateToClientLevel(top_value_products, compute_rfm=True)


# In[116]:


model_agregate.fit(df_train)


# In[117]:


df_clients_test = model_agregate.transform(df_test)


# In[119]:


df_clients_test.shape


# In[105]:


model = Pipeline([
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    ('clusterer', Clusterer(n_clusters=4, algorithm_to_use='WARD'))
])


# In[106]:


model.fit(df_train)


# Score verification on final test set :

# In[109]:


score_test = model.score(df_test)


# In[110]:


score_test


# In[111]:


cluster_labels_test = model.predict(df_test)


# S'inspirer de : # Model with all features and RFM score (concat) (not individual RFM feats), and NCA (BEST)# 

# In[121]:


model_beforecluster = Pipeline([
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[122]:


model_beforecluster.fit(df_train)


# In[123]:


df_test_transformed = model_beforecluster.transform(df_test)


# In[124]:


df_test_transformed


# A FAIRE : renvoyer un dataframe en sorte de predict() du modèle final, 

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


# In[49]:


df_train_nocancel = df_train[df_train['InvoiceNo'].str.startswith('C') == False]
df_train_nocancel.reset_index(inplace=True)


# In[37]:


feat_list = ['CustomerID', 'TotalPrice']
feat_list_bow = [col for col in df_train_nocancel if col.startswith('DescriptionNormalized_')]
feat_list.extend(feat_list_bow)


# In[38]:


feat_list


# In[39]:


df_train_gbcust_nocancel = df_train_nocancel[feat_list].groupby('CustomerID').sum()


# In[51]:


df_train_gbcust_nocancel[feat_list_bow] = df_train_gbcust_nocancel[feat_list_bow].clip(upper=1)


# In[52]:


df_train_gbcust_nocancel


# In[40]:


# Number of months between first order date and last date of the dataset
series_train_gbclient_nbmonths = np.maximum((
   (
   pd.to_datetime('2011-12-09 12:50:00')\
   - pd.to_datetime(df_train_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
   )\
    / (np.timedelta64(1, "M"))
), 1)


# In[41]:


series_train_gbclient_nbmonths


# In[42]:


df_train_gbcust_nocancel['TotalPrice'] 


# In[43]:


df_train_gbcust_nocancel['TotalPrice'] = df_train_gbcust_nocancel['TotalPrice'] / series_train_gbclient_nbmonths


# In[44]:


df_train_gbcust_nocancel


# In[46]:


df_train_gbcust_nocancel['Recency'] = series_train_gbclient_nbmonths


# In[47]:


df_train_gbcust_nocancel


# In[65]:


df_train_nocancel[['CustomerID', 'Quantity']].groupby('CustomerID').sum()['Quantity']


# In[62]:


series_train_gbclient_nbmonths


# In[66]:


df_train_nocancel[['CustomerID', 'Quantity']].groupby('CustomerID').sum()['Quantity'] / series_train_gbclient_nbmonths


# In[53]:


df_train_nocancel[['CustomerID', 'Quantity']].groupby('CustomerID').sum() / series_train_gbclient_nbmonths


# In[55]:


df_train_gbcust_nocancel


# In[54]:


df_train_nocancel[['CustomerID', 'TotalPrice']].groupby('CustomerID').sum() / series_train_gbclient_nbmonths


# In[58]:


series_train_gbclient_nbmonths


# In[60]:


df_train_nocancel[['CustomerID', 'TotalPrice']].groupby('CustomerID').sum() / series_train_gbclient_nbmonths


# In[57]:


df_train_gbcust_nocancel['TotalPricePerMonth'] = df_train_nocancel[['CustomerID', 'TotalPrice']].groupby('CustomerID').sum() / series_train_gbclient_nbmonths


# In[51]:


df_train_gbcust_nocancel['TotalQuantityPerMonth'] = df_train_nocancel[['CustomerID', 'Quantity']].groupby('CustomerID').sum() / series_train_gbclient_nbmonths


# In[58]:


df_train


# In[59]:


custid_cancelled = df_train[df_train['InvoiceNo'].str.startswith('C') == True]['CustomerID'].unique()


# In[268]:


df


# In[272]:


df.groupby(['CustomerID', 'Quantity']).count()


# In[288]:


df_pivot = df.pivot_table(index=['CustomerID'], columns=['StockCode'], values='Quantity', aggfunc='sum', fill_value=0)


# In[289]:


df_pivot.loc['17850', '85123A']


# In[287]:


df[ (df['CustomerID'] == '17850') & (df['StockCode'] == '85123A')]['Quantity'].mean()


# In[292]:


df_pivot


# ## RFM table

# In[121]:


df_train


# In[92]:


quantiles = df_train[['TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']].quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[93]:


quantiles


# In[99]:


def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[94]:


df_rfmtable = df_train[['TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']]


# In[103]:


df_rfmtable


# In[107]:


df_rfmtable.loc[:, 'r_quartile'] = df_rfmtable.loc[:, 'Recency'].apply(RScore, args=('Recency',quantiles,))


# In[109]:


df_rfmtable.loc[:, 'r_quartile'] = df_rfmtable.loc[:, 'Recency'].apply(RScore, args=('Recency',quantiles,))
df_rfmtable.loc[:, 'f_quartile'] = df_rfmtable.loc[:, 'TotalQuantityPerMonth'].apply(FMScore, args=('TotalQuantityPerMonth',quantiles,))
df_rfmtable.loc[:, 'm_quartile'] = df_rfmtable.loc[:, 'TotalPricePerMonth'].apply(FMScore, args=('TotalPricePerMonth',quantiles,))
df_rfmtable.head()


# In[113]:


quantiles


# In[112]:


df_rfmtable.loc[:, 'RFMScore'] = df_rfmtable.r_quartile.map(str)                             + df_rfmtable.f_quartile.map(str)                             + df_rfmtable.m_quartile.map(str)
df_rfmtable.head()


# In[116]:


df_rfmtable.head(1)


# In[118]:


df_rfmtable


# In[123]:


df_train['RFMScore'] = df_rfmtable['RFMScore']


# In[124]:


df_train


# In[120]:


df_rfmtable.drop(columns=['r_quartile', 'f_quartile', 'm_quartile'], inplace=True)


# # Feature selection attempt

# In[244]:


# create the RFE model and select 3 attributes
rfe = RFE(Clusterer(n_clusters=11, algorithm_to_use='WARD'), 3)
rfe = rfe.fit(df_train, rfm_scores_train)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

