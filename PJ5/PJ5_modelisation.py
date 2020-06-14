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
  
import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

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

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

DATA_PATH = os.path.join("datasets", "onlineretail")
DATA_PATH = os.path.join(DATA_PATH, "out")

DATA_PATH_FILE_INPUT = os.path.join(DATA_PATH, "OnlineRetail_transformed.csv")


ALL_FEATURES = []

#MODEL_FEATURES=['InvoiceNo', 'InvoiceDate', 'CustomerID', 'TotalPrice', 'DescriptionNormalized', 'InvoiceMonth', 'StockCode']
MODEL_CLIENT_FEATURES = ['TotalPricePerMonth', 'DescriptionNormalized', 'HasEverCancelled', 'BoughtTopValueProduct' ]

plt.rcParams["figure.figsize"] = [16,9] # Taille par défaut des figures de matplotlib

import seaborn as sns
from seaborn import boxplot
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


# In[18]:


series_total_price_per_month_train = df_train['TotalPricePerMonth']
series_total_price_per_month_test = df_test['TotalPricePerMonth']

series_hasevercancelled_train = df_train['HasEverCancelled']
series_hasevercancelled_test = df_test['HasEverCancelled']

series_boughttopvalueproduct_train = df_train['BoughtTopValueProduct']
series_boughttopvalueproduct_test = df_test['BoughtTopValueProduct']


# # Explained variance of bag of words features

# In[19]:


from display_factorial import *
importlib.reload(sys.modules['display_factorial'])


# In[20]:


display_scree_plot(preparation_pipeline['dimensionality_reductor'].reductor)


# # 2D visualization

# In[21]:


pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[22]:


X_transformed[:,1]


# In[23]:


print('Binarisation of color categories')
bins = [-np.inf,df_train['TotalPricePerMonth'].quantile(0.25),        df_train['TotalPricePerMonth'].quantile(0.50),        df_train['TotalPricePerMonth'].quantile(0.75),        df_train['TotalPricePerMonth'].quantile(1)]

labels = [0, 1, 2, 3]

df_score_cat_train = pd.cut(df_train['TotalPricePerMonth'], bins=bins, labels=labels)


bins = [-np.inf,df_test['TotalPricePerMonth'].quantile(0.25),        df_test['TotalPricePerMonth'].quantile(0.50),        df_test['TotalPricePerMonth'].quantile(0.75),        df_test['TotalPricePerMonth'].quantile(1)]

labels = [0, 1, 2, 3]

df_score_cat_test = pd.cut(df_test['TotalPricePerMonth'], bins=bins, labels=labels)


# In[24]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[25]:


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


# In[26]:


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
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'RfmScore'])),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['RfmScore'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='NCA', n_dim=1, labels_featurename='RfmScore')),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[30]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[31]:


df_train.loc[:, 0].to_numpy()


# In[32]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train.loc[:, 0].to_numpy().reshape(-1,1))
                for k in range(1, 50)]


# In[33]:


labels_test_per_k = [model.predict(df_test.loc[:, 0].to_numpy().reshape(-1,1)) for model in kmeans_per_k[1:]]


# In[34]:


silhouette_scores = [silhouette_score(df_train.loc[:, 0].to_numpy().reshape(-1,1), model.labels_)
                     for model in kmeans_per_k[1:]]


# In[35]:


silhouette_scores_test = [silhouette_score(df_test.loc[:, 0].to_numpy().reshape(-1,1), labels_test) for labels_test in labels_test_per_k]


# In[36]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[37]:


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
    


# In[38]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[39]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[40]:


bow_labels_train = kmeans_per_k[10].labels_


# In[41]:


bow_labels_test = kmeans_per_k[10].predict(df_test.loc[:, 0].to_numpy().reshape(-1,1))


# # Model with only bag of word features, PCA

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
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='PCA', n_dim=200)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[45]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[46]:


df_test = preparation_pipeline.transform(df_test)


# In[47]:


df_train


# In[48]:


pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)

'''
tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)
'''


# In[49]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[50]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, test set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[51]:


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

# In[52]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[53]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[54]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[55]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[56]:


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
    


# In[57]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[58]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[59]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[60]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[61]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## 2nd clustering : ward

# In[62]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[63]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[64]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[65]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[66]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[67]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with only bag of word features, TSNE

# In[68]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[69]:


df_train = df_train_ori
df_test = df_test_ori


# In[70]:


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


# In[71]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[72]:


df_test = preparation_pipeline.transform(df_test)


# In[73]:


df_train


# In[74]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[75]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[76]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, test set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[77]:


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

# In[78]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[79]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[80]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[81]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[82]:


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
    


# In[83]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[84]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[85]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[86]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[87]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## 2nd clustering : ward

# In[88]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[89]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[90]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[91]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[92]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[93]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with only bow features and TotalPricePerMonth, TSNE

# In[94]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[95]:


df_train = df_train_ori
df_test = df_test_ori


# In[96]:


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


# In[97]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[98]:


df_test = preparation_pipeline.transform(df_test)


# In[99]:


df_train


# In[100]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[101]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[102]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, test set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[103]:


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

# In[104]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[105]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[106]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[107]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[108]:


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
    


# In[109]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[110]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[111]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[112]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[113]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## 2nd clustering : ward

# In[114]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[115]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[116]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[117]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[118]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[119]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with bow features + TotalPricePerMonth

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
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'TotalPricePerMonth'])),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                        algorithm_to_use='PCA', n_dim=200)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[123]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[124]:


df_test = preparation_pipeline.transform(df_test)


# In[125]:


df_train


# In[126]:


pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[ ]:





# In[127]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW + TotalPricePerMonth')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[128]:


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

# In[129]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[130]:


df_train = df_train_ori
df_test = df_test_ori


# In[131]:


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


# In[132]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[133]:


df_test = preparation_pipeline.transform(df_test)


# In[134]:


df_train


# In[135]:


pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[ ]:





# In[136]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW + TotalPricePerMonth + HasEverCancelled')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[137]:


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

# In[138]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[139]:


df_train = df_train_ori
df_test = df_test_ori


# In[140]:


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


# In[141]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[142]:


df_train


# In[143]:


df_train.info()


# In[144]:


pca = PCA(n_components=2,random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[145]:


X_transformed[:,1]


# In[146]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[147]:


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

# In[148]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[149]:


df_train = df_train_ori
df_test = df_test_ori


# In[150]:


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


# In[151]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[152]:


df_train


# In[153]:


df_train.info()


# In[154]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[155]:


X_transformed[:,1]


# In[156]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[157]:


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

# In[158]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[159]:


df_train = df_train_ori
df_test = df_test_ori


# In[160]:


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


# In[161]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[162]:


df_train


# In[163]:


df_train.info()


# In[164]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[165]:


X_transformed[:,1]


# In[166]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[167]:


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

# In[168]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[169]:


df_train = df_train_ori
df_test = df_test_ori


# In[170]:


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


# In[171]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[172]:


df_train


# In[173]:


df_train.info()


# In[174]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[175]:


X_transformed[:,1]


# In[176]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[177]:


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

# In[178]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[179]:


df_train = df_train_ori
df_test = df_test_ori


# In[180]:


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


# In[181]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[182]:


df_train


# In[183]:


df_train.info()


# In[184]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[185]:


X_transformed[:,1]


# In[186]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, all feats except bow, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[187]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, all feats except bow, test set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[188]:


df_train.iloc[:, 0]


# In[189]:


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


# In[190]:


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


# In[191]:


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

# In[207]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[208]:


df_train = df_train_ori
df_test = df_test_ori


# In[209]:


preparation_pipeline_agg = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products)),
    ('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale=['TotalPricePerMonth'])),
])


# In[210]:


df_train_agg = preparation_pipeline_agg.fit_transform(df_train)
df_test_agg = preparation_pipeline_agg.transform(df_test)


# In[211]:


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


# In[212]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[213]:


df_train


# In[214]:


df_train.info()


# In[215]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[216]:


X_transformed[:,1]


# In[217]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[219]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_transformed[:,0], y = X_transformed[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=df_score_cat_train),
                    text = df_train_agg['TotalPricePerMonth'],
                    )


layout = go.Layout(title = 'Représentation des clients en 2 dimensions',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_clients_sne_allfeats_final_tsne.html') 


# # Model with all features and tSNE, final representation with tSNE 3D

# In[220]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[221]:


df_train = df_train_ori
df_test = df_test_ori


# In[222]:


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


# In[223]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[224]:


df_train


# In[225]:


df_train.info()


# In[226]:


tsne = TSNE(n_components=3, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[227]:


X_transformed[:,1]


# In[228]:


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

# In[229]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[230]:


df_train = df_train_ori
df_test = df_test_ori


# In[231]:


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


# In[232]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[233]:


df_train


# In[234]:


df_train.info()


# In[235]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[236]:


X_transformed[:,1]


# In[237]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[238]:


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

# In[239]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[240]:


df_train = df_train_ori
df_test = df_test_ori


# In[241]:


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


# In[242]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[243]:


df_train


# In[244]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[245]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[246]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[247]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[248]:


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
    


# In[249]:


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
    


# In[250]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[251]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[252]:


print('Gini before clustering :')
gini(df_train['TotalPricePerMonth'].to_numpy())


# In[253]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), gini_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean gini score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[254]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), gini_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean gini score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[255]:


print('Entropy before clustering :')
entropy(df_train['TotalPricePerMonth'])


# In[256]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[257]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with tSNE, then clustering algorithm Ward
# No visualisation on test set because AgglomerativeClustering has no predict function, only fit_predict

# In[258]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[259]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[260]:


gini_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    gini_sum = 0
    for unique_label in unique_labels:
        gini_sum += gini(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    gini_sum = gini_sum / len(unique_labels)
    
    gini_mean_score_per_k_train.append(gini_sum)


    


# In[261]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[262]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[263]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), gini_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean gini score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[264]:


print('Entropy before clustering :')
entropy(df_train['TotalPricePerMonth'])


# In[265]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with tSNE, then clustering algorithm Ward, distance threshold
# No visualisation on test set because AgglomerativeClustering has no predict function, only fit_predict

# In[266]:


np.unique(AgglomerativeClustering(distance_threshold=1, n_clusters=None, affinity='euclidean').fit(df_train).labels_)


# In[270]:


clusterer_ward_per_thr = [AgglomerativeClustering(distance_threshold=thr, n_clusters=None, affinity='euclidean').fit(df_train) for thr in reversed(range(0,12))]


# In[271]:


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


# In[273]:


plt.figure(figsize=(8, 3))
plt.plot(range(1,12), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$Ward threshold$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # => Around 50 clusters => entropy of TotalPrice around 4.5

# # Prepation model with LLE reduce to 200, then clustering algorithm Ward

# In[274]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[275]:


df_train = df_train_ori
df_test = df_test_ori


# In[276]:


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


# In[277]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[278]:


df_test = preparation_pipeline.transform(df_test)


# In[279]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[281]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[282]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[283]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[284]:


print('Entropy before clustering :')
entropy(df_train['TotalPricePerMonth'])


# In[285]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Prepation model with LLE reduce to 3, then clustering algorithm Ward

# In[286]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[287]:


df_train = df_train_ori
df_test = df_test_ori


# In[288]:


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


# In[289]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[290]:


df_test = preparation_pipeline.transform(df_test)


# In[291]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[292]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[293]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[294]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[295]:


print('Entropy before clustering :')
entropy(df_train['TotalPricePerMonth'])


# In[296]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Preparation model BoW feats only, then LLE reduce to 200, then KMeans and Ward

# In[297]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[298]:


df_train = df_train_ori
df_test = df_test_ori


# In[299]:


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


# In[300]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[301]:


df_test = preparation_pipeline.transform(df_test)


# In[302]:


df_train


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


# ## 2nd clustering : Ward

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


# # Preparation model BoW feats only, then LLE reduce to 3 then KMeans and Ward

# In[319]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[320]:


df_train = df_train_ori
df_test = df_test_ori


# In[321]:


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


# In[322]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[323]:


df_test = preparation_pipeline.transform(df_test)


# In[324]:


df_train


# In[325]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[326]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[327]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[328]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[329]:


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
    


# In[330]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[331]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[332]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[333]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[334]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## 2nd clustering : ward

# In[335]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[336]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[337]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[338]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[339]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[340]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Plot representation

# In[342]:


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

# In[343]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[344]:


df_train = df_train_ori
df_test = df_test_ori


# In[345]:


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


# In[346]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[347]:


rfm_scores_train = get_rfm_scores(df_train)


# In[348]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[349]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[350]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[351]:


rfm_scores_train_colors


# In[352]:


rfm_scores_train


# In[353]:


df_train


# In[354]:


df_train.info()


# In[355]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[356]:


X_transformed[:,1]


# In[357]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[358]:


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


# In[359]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[360]:


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


# In[361]:


## Add bow coloration


# In[362]:


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

# In[363]:


df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 0, 1, 2]]


# In[364]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 0, 1, 2]])
X_test_transformed = tsne.fit_transform(df_test[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 0, 1, 2]])


# In[365]:


X_transformed[:,1]


# In[366]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[367]:


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


# In[368]:


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

# In[369]:


df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 0, 1, 2]]


# In[370]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 0, 1, 2]])
X_test_transformed = tsne.fit_transform(df_test[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 0, 1, 2]])


# In[371]:


X_transformed[:,1]


# In[372]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[373]:


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


# ## RFM only (without bow, BoughtTopValueProduct and HasEverCancelled)

# In[375]:


df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth']]


# In[376]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth']])
X_test_transformed = tsne.fit_transform(df_test[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth']])


# In[377]:


X_transformed[:,1]


# In[378]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[379]:


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


# In[380]:


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

# In[381]:


df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled']]


# In[382]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled']])
X_test_transformed = tsne.fit_transform(df_test[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled']])


# In[383]:


X_transformed[:,1]


# In[384]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[385]:


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


# In[386]:


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

# In[387]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[388]:


df_train = df_train_ori
df_test = df_test_ori


# In[389]:


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


# In[390]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[391]:


rfm_scores_train = df_train['RfmScore']


# In[392]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[393]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[394]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[395]:


rfm_scores_train_colors


# In[396]:


rfm_scores_train


# In[397]:


df_train


# In[398]:


df_train.info()


# In[399]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[400]:


X_transformed[:,1]


# In[401]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[402]:


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

# In[403]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[404]:


df_train = df_train_ori
df_test = df_test_ori


# In[405]:


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


# In[406]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[407]:


rfm_scores_train = df_train['RfmScore']


# In[408]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[409]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[410]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[411]:


rfm_scores_train_colors


# In[412]:


rfm_scores_train


# In[413]:


df_train


# In[414]:


df_train.info()


# In[415]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[416]:


X_transformed[:,1]


# In[417]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[418]:


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

# In[419]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[420]:


df_train = df_train_ori
df_test = df_test_ori


# In[421]:


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


# In[422]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[423]:


#rfm_scores_train = df_train['RfmScore']  # we reuse rfm_scores_train calculated on above model


# In[424]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[425]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[426]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[427]:


rfm_scores_train_colors


# In[428]:


rfm_scores_train


# In[429]:


df_train


# In[430]:


df_train.info()


# In[431]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[432]:


X_transformed[:,1]


# In[433]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[434]:


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

# In[435]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[436]:


df_train = df_train_ori
df_test = df_test_ori


# In[437]:


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


# In[438]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[439]:


#rfm_scores_train = df_train['RfmScore']  # we reuse rfm_scores_train calculated on above model


# In[440]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[441]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[442]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[443]:


rfm_scores_train_colors


# In[444]:


rfm_scores_train


# In[445]:


df_train


# In[446]:


df_train.info()


# In[447]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[448]:


X_transformed[:,1]


# In[449]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[450]:


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

# In[451]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[452]:


df_train = df_train_ori
df_test = df_test_ori


# In[453]:


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


# In[454]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[455]:


rfm_scores_train = df_train['RfmScore']


# In[456]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[457]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[458]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[459]:


rfm_scores_train_colors


# In[460]:


rfm_scores_train


# In[461]:


df_train


# In[462]:


df_train.info()


# In[463]:


'''
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_transformed = lle.fit_transform(df_train)
X_test_transformed = lle.fit_transform(df_test)
'''
tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[464]:


X_transformed[:,1]


# In[465]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[466]:


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

# In[467]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[468]:


df_train = df_train_ori
df_test = df_test_ori


# In[469]:


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


# In[470]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[471]:


rfm_scores_train = df_train['RfmScore']


# In[472]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[473]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[474]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[475]:


rfm_scores_train_colors


# In[476]:


rfm_scores_train


# In[477]:


df_train


# In[478]:


df_train.info()


# In[479]:


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


# In[480]:


X_transformed[:,1]


# In[481]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[482]:


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

# In[483]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[484]:


df_train = df_train_ori
df_test = df_test_ori


# In[485]:


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


# In[486]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[487]:


rfm_scores_train = df_train['RfmScore']


# In[488]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[489]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[490]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[491]:


rfm_scores_train_colors


# In[492]:


rfm_scores_train


# In[493]:


df_train


# In[494]:


df_train.info()


# In[495]:


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


# In[496]:


X_transformed[:,1]


# In[497]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[498]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, TotalPricePerMonth colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[499]:


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

# In[500]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[501]:


df_train = df_train_ori
df_test = df_test_ori


# In[502]:


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


# In[503]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[504]:


rfm_scores_train = df_train['RfmScore']


# In[505]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[506]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[507]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[508]:


rfm_scores_train_colors


# In[509]:


rfm_scores_train


# In[510]:


df_train


# In[511]:


df_train.info()


# In[512]:


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


# In[513]:


X_transformed[:,1]


# In[514]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[515]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, TotalPricePerMonth colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[516]:


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


# In[517]:


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


# In[518]:


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

# In[519]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[520]:


df_train = df_train_ori
df_test = df_test_ori


# In[521]:


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


# In[522]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[523]:


#rfm_scores_train = df_train['RfmScore'] # rfm_scores_train value has been got from code above


# In[524]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[525]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[526]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[527]:


rfm_scores_train_colors


# In[528]:


rfm_scores_train


# In[529]:


df_train


# In[530]:


df_train.info()


# In[531]:


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


# In[532]:


X_transformed[:,1]


# In[533]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[534]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, TotalPricePerMonth colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[535]:


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

# In[536]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[537]:


df_train = df_train_ori
df_test = df_test_ori


# In[538]:


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


# In[539]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[540]:


rfm_scores_train = df_train['RfmScore']
rfm_scores_test = df_test['RfmScore']


# In[541]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())
unique_rfm_scores_test = np.sort(rfm_scores_test.unique())


# In[542]:


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


# In[543]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])
rfm_scores_test_colors = rfm_scores_test.apply(lambda x : rfm_dict_colors_test[x])


# In[544]:


rfm_scores_train_colors


# In[545]:


rfm_scores_test_colors


# In[546]:


rfm_scores_train


# In[547]:


df_train


# In[548]:


df_train.info()


# In[549]:


'''
df_train_rfmscore_distances = pairwise_distances(df_train['RfmScore'].to_numpy().reshape(-1, 1))
df_test_rfmscore_distances = pairwise_distances(df_test['RfmScore'].to_numpy().reshape(-1, 1))
'''


# In[550]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[551]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[552]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[553]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[554]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[555]:


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
    


# In[556]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[557]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[558]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[559]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[560]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## Reduce and visualize

# In[561]:


nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
X_transformed = nca.fit_transform(df_train, pd.cut(df_train['RfmScore'], bins=range(1,10), right=True).astype(str).tolist())
X_test_transformed = nca.transform(df_test)


# In[562]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[563]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, cluster label colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[564]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_transformed)
                for k in range(1, 50)]


# In[565]:


labels_test_per_k = [model.predict(X_test_transformed) for model in kmeans_per_k[1:]]


# In[566]:


silhouette_scores = [silhouette_score(X_transformed, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[567]:


silhouette_scores_test = [silhouette_score(X_test_transformed, labels_test) for labels_test in labels_test_per_k]


# In[568]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[569]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[570]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[571]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[572]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with 'DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore' (SUM) with NCA up to 200 then KMeans then NCA to visualize clusters   (GOOD, ONLY NCA not TSNE)

# In[573]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[574]:


df_train = df_train_ori
df_test = df_test_ori


# In[575]:


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


# In[576]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[577]:


rfm_scores_train = df_train['RfmScore']
rfm_scores_test = df_test['RfmScore']


# In[578]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())
unique_rfm_scores_test = np.sort(rfm_scores_test.unique())


# In[579]:


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


# In[580]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])
rfm_scores_test_colors = rfm_scores_test.apply(lambda x : rfm_dict_colors_test[x])


# In[581]:


rfm_scores_train_colors


# In[582]:


rfm_scores_test_colors


# In[583]:


rfm_scores_train


# In[584]:


df_train


# In[585]:


df_train.info()


# In[586]:


'''
df_train_rfmscore_distances = pairwise_distances(df_train['RfmScore'].to_numpy().reshape(-1, 1))
df_test_rfmscore_distances = pairwise_distances(df_test['RfmScore'].to_numpy().reshape(-1, 1))
'''


# In[587]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[588]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[589]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[590]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[591]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[592]:


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
    


# In[593]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[594]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[595]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[596]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[597]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## Reduce and visualize

# In[598]:


nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
X_transformed = nca.fit_transform(df_train, pd.cut(df_train['RfmScore'], bins=range(1,10), right=True).astype(str).tolist())
X_test_transformed = nca.transform(df_test)


# In[599]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[600]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, cluster label colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[601]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_transformed)
                for k in range(1, 50)]


# In[602]:


labels_test_per_k = [model.predict(X_test_transformed) for model in kmeans_per_k[1:]]


# In[603]:


silhouette_scores = [silhouette_score(X_transformed, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[604]:


silhouette_scores_test = [silhouette_score(X_test_transformed, labels_test) for labels_test in labels_test_per_k]


# In[605]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[606]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[607]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[608]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[609]:


df_test


# In[610]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[611]:


clusterer = Clusterer(n_clusters=11, algorithm_to_use='WARD')
clusterer.fit(df_train)


# In[612]:


cluster_labels_test = clusterer.predict(df_test)


# In[613]:


cluster_labels_train = clusterer.predict(df_train)


# In[615]:


df_train[df_train.index == 0]


# In[686]:


len(cluster_labels_test)


# In[688]:


cluster_labels_test


# In[756]:


df_train


# In[690]:


cluster_labels_train


# In[691]:


clusterer.clusterer.labels_


# In[757]:


#clusterer.score(df_train)


# In[758]:


silhouette_scores_test


# # Model with 'TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled' with NCA up to 200 then KMeans then NCA to visualize clusters

# In[759]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[760]:


df_train = df_train_ori
df_test = df_test_ori


# In[761]:


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


# In[762]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[763]:


#rfm_scores_train = df_train['RfmScore']  # Got from code above
#rfm_scores_test = df_test['RfmScore']


# In[764]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())
unique_rfm_scores_test = np.sort(rfm_scores_test.unique())


# In[765]:


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


# In[766]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])
rfm_scores_test_colors = rfm_scores_test.apply(lambda x : rfm_dict_colors_test[x])


# In[767]:


rfm_scores_train_colors


# In[768]:


rfm_scores_test_colors


# In[769]:


rfm_scores_train


# In[770]:


df_train


# In[771]:


df_train.info()


# In[772]:


'''
df_train_rfmscore_distances = pairwise_distances(df_train['RfmScore'].to_numpy().reshape(-1, 1))
df_test_rfmscore_distances = pairwise_distances(df_test['RfmScore'].to_numpy().reshape(-1, 1))
'''


# In[773]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[774]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[775]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[776]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[777]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[778]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## Reduce and visualize

# In[779]:


nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
X_transformed = nca.fit_transform(df_train, pd.cut(rfm_scores_train, bins=range(1,10), right=True).astype(str).tolist())
X_test_transformed = nca.transform(df_test)


# In[780]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[781]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_transformed)
                for k in range(1, 50)]


# In[782]:


labels_test_per_k = [model.predict(X_test_transformed) for model in kmeans_per_k[1:]]


# In[783]:


silhouette_scores = [silhouette_score(X_transformed, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[784]:


silhouette_scores_test = [silhouette_score(X_test_transformed, labels_test) for labels_test in labels_test_per_k]


# In[785]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[786]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency' then KMeans then 3D visualisation

# In[787]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[788]:


df_train = df_train_ori
df_test = df_test_ori


# In[789]:


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


# In[790]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[791]:


#rfm_scores_train = df_train['RfmScore']  # Got from above
#rfm_scores_test = df_test['RfmScore']


# In[792]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())
unique_rfm_scores_test = np.sort(rfm_scores_test.unique())


# In[793]:


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


# In[794]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])
rfm_scores_test_colors = rfm_scores_test.apply(lambda x : rfm_dict_colors_test[x])


# In[795]:


rfm_scores_train_colors


# In[796]:


rfm_scores_test_colors


# In[797]:


rfm_scores_train


# In[798]:


df_train


# In[799]:


df_train.info()


# In[800]:


'''
df_train_rfmscore_distances = pairwise_distances(df_train['RfmScore'].to_numpy().reshape(-1, 1))
df_test_rfmscore_distances = pairwise_distances(df_test['RfmScore'].to_numpy().reshape(-1, 1))
'''


# In[801]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[802]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[803]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[804]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[805]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[806]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[807]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# ## Visualize

# In[808]:


df_train


# In[809]:


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


# In[810]:


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


# In[811]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, cluster label colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[812]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_transformed)
                for k in range(1, 50)]


# In[813]:


labels_test_per_k = [model.predict(X_test_transformed) for model in kmeans_per_k[1:]]


# In[814]:


silhouette_scores = [silhouette_score(X_transformed, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[815]:


silhouette_scores_test = [silhouette_score(X_test_transformed, labels_test) for labels_test in labels_test_per_k]


# In[816]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[817]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Correlations

# In[819]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[820]:


df_train = df_train_ori
df_test = df_test_ori


# In[821]:


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


# In[822]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[823]:


corr_matrix = df_train.corr()


# In[824]:


corr_matrix


# In[825]:


plt.title('Corrélation entre les features')
sns.heatmap(corr_matrix, 
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.columns, cmap='coolwarm' ,center=0.20)


# In[826]:


'''
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols("BMI ~ 0 + 1 + 2", data=df_train).fit()
#print model.params
#print model.summary()
'''


# # Generate bow colors

# In[827]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[828]:


df_train = df_train_ori
df_test = df_test_ori


# In[829]:


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


# In[830]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[831]:


df_train.loc[:, 0].to_numpy()


# In[832]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train.loc[:, 0].to_numpy().reshape(-1,1))
                for k in range(1, 50)]


# In[833]:


labels_test_per_k = [model.predict(df_test.loc[:, 0].to_numpy().reshape(-1,1)) for model in kmeans_per_k[1:]]


# In[834]:


silhouette_scores = [silhouette_score(df_train.loc[:, 0].to_numpy().reshape(-1,1), model.labels_)
                     for model in kmeans_per_k[1:]]


# In[835]:


silhouette_scores_test = [silhouette_score(df_test.loc[:, 0].to_numpy().reshape(-1,1), labels_test) for labels_test in labels_test_per_k]


# In[836]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[837]:


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
    


# In[838]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[839]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[840]:


bow_labels_train = kmeans_per_k[10].labels_


# In[841]:


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

# In[42]:


from functions import *
importlib.reload(sys.modules['functions'])
from functions import *


# In[43]:


df_train = df_train_ori
df_test = df_test_ori


# ## Agregate to client level to get text labels for visualisation, and interprete model with surrogate Decision Tree

# In[44]:


#model_agregate = AgregateToClientLevel(top_value_products, compute_rfm=True)
model_agregate = Pipeline([
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='ISOMAP', n_dim=3)),
])


# In[45]:


model_agregate.fit(df_train)


# In[46]:


df_clients_train_agreg = model_agregate.transform(df_train)


# In[47]:


df_clients_test_agreg = model_agregate.transform(df_test)


# In[48]:


df_clients_test_agreg


# In[49]:


df_clients_test_agreg.shape


# ## Loop on top 6 models

# In[50]:


df_train = df_train_ori
df_test = df_test_ori


# In[110]:


models = [
    Pipeline([
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    ('clusterer', Clusterer(n_clusters=4, algorithm_to_use='WARD'))
    ]),
    
    Pipeline([
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    ('clusterer', Clusterer(n_clusters=8, algorithm_to_use='KMEANS'))
    ]),
        
    Pipeline([
        ('bow_encoder', BowEncoder()),
        ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
        ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),

        ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='NCA', n_dim=3, labels_featurename=None)),
        ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('clusterer', Clusterer(n_clusters=4, algorithm_to_use='WARD'))
        ]),

    Pipeline([
        ('bow_encoder', BowEncoder()),
        ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
        ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
        #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
        ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='ISOMAP', n_dim=3)),
        ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('clusterer', Clusterer(n_clusters=6, algorithm_to_use='KMEANS'))
        ]),
    
    Pipeline([
        ('bow_encoder', BowEncoder()),
        ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
        ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
        #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
        ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='ISOMAP', n_dim=3)),
        ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('clusterer', Clusterer(n_clusters=6, algorithm_to_use='KMEANS'))
        ]),

    Pipeline([
        ('bow_encoder', BowEncoder()),
        ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
        ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'RfmScore'])),

        ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='NCA', n_dim=3, labels_featurename=None)),
        ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('clusterer', Clusterer(n_clusters=4, algorithm_to_use='KMEANS'))
        ]),

]


models_before_clustering = [
    Pipeline([
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    ]),
    
    Pipeline([
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    ]),
        
    Pipeline([
        ('bow_encoder', BowEncoder()),
        ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
        ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),

        ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='NCA', n_dim=3, labels_featurename=None)),
        ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ]),

    Pipeline([
        ('bow_encoder', BowEncoder()),
        ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
        ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
        #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
        ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='ISOMAP', n_dim=3)),
        ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ]),
    
    Pipeline([
        ('bow_encoder', BowEncoder()),
        ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
        ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
        #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
        ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='ISOMAP', n_dim=3)),
        ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ]),

    Pipeline([
        ('bow_encoder', BowEncoder()),
        ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
        ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'RfmScore'])),

        ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='NCA', n_dim=3, labels_featurename=None)),
        ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ]),

]

models_clustering = [
        Clusterer(n_clusters=4, algorithm_to_use='WARD'),
        Clusterer(n_clusters=8, algorithm_to_use='KMEANS'),
        Clusterer(n_clusters=4, algorithm_to_use='WARD'),
        Clusterer(n_clusters=6, algorithm_to_use='KMEANS'),
        Clusterer(n_clusters=6, algorithm_to_use='KMEANS'),
        Clusterer(n_clusters=4, algorithm_to_use='KMEANS'),
]


# In[111]:


# Below code can be uncommented to run only one model with code in next cell

'''
# Model 2 :
models = [
    Pipeline([
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    ('clusterer', Clusterer(n_clusters=8, algorithm_to_use='KMEANS'))
    ]),
]


models_before_clustering = [
    Pipeline([
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    ]),
]

models_clustering = [
        Clusterer(n_clusters=8, algorithm_to_use='KMEANS')
]
'''
'''
#Model 4 :
models = [
    Pipeline([
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                            algorithm_to_use='ISOMAP', n_dim=3)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    ('clusterer', Clusterer(n_clusters=6, algorithm_to_use='KMEANS'))
    ]),
]


models_before_clustering = [
    Pipeline([
        ('bow_encoder', BowEncoder()),
        ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
        ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
        #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
        ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='ISOMAP', n_dim=3)),
        ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ]),
]

models_clustering = [
        Clusterer(n_clusters=6, algorithm_to_use='KMEANS')
]

'''


# In[115]:


model_number = 0

for model_prep, model_clusterer in zip(models_before_clustering, models_clustering):
    model_number += 1
    df_train = df_train_ori
    df_test = df_test_ori    
    
    print('Model ' + str(model_number) + ':\n' + str(model_prep) + '\n' + str(model_clusterer))
    
    print('Running preparation model')
    
    if ((model_number == 3) or (model_number == 6)): # For NCA we must pass a dimensionality reductor label
        model_prep.fit(df_train, dimensionality_reductor__labels=df_clients_train_agreg['RfmScore'])
        
    else:
        model_prep.fit(df_train)
        
    df_clients_train = model_prep.transform(df_train)
    df_clients_test = model_prep.transform(df_test)
        
    print('Running clusterer model : calculating cluster labels and test set score')
    #print('1')
    model_clusterer.fit(df_clients_train)
    
    #print('2')
    df_predictions_test = model_clusterer.predict(df_clients_test)
    df_predictions_train = model_clusterer.predict(df_clients_train)
    
    #print('3')
    score_test = model_clusterer.score(df_clients_test)
    print(f'> Score on test set : {score_test}')
    
    print('Client representation')
    reductor = DimensionalityReductor(algorithm_to_use='TSNE', n_dim=2, features_totransform='ALL')
    df_clients_test_reduced = reductor.fit_transform(df_clients_test)


    py.offline.init_notebook_mode(connected=True)

    trace_1 = go.Scatter(x = df_clients_test_reduced.loc[:,0], y = df_clients_test_reduced.loc[:,1],
                        name = 'Clients',
                        mode = 'markers',
                        marker=dict(color=df_predictions_test),
                        #text = rfm_scores_train,
                        #text = [('Bought top value product' if (boughttopvalueproduct == 1) else 'dit NOT buy top value product') for boughttopvalueproduct in df_train['BoughtTopValueProduct']],
                        text = list(map(str, zip('RFM: ' + df_clients_test_agreg['RfmScore'].astype(str),\
                                                 'BoughtTopValueProduct: ' + df_clients_test_agreg['BoughtTopValueProduct'].astype(str),\
                                                  'HasEverCancelled: '  + df_clients_test_agreg['HasEverCancelled'].astype(str),\
                                                ))\
                                            )
                        )


    layout = go.Layout(title = 'Model ' + str(model_number) + ' : client representation, colored by cluster',
                       hovermode = 'closest',
    )

    fig = go.Figure(data = [trace_1], layout = layout)

    #py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

    py.offline.plot(fig, filename='clusters_model' + str(model_number) + '.html') 
    
    print('Interpreted clusters with decision tree:')
    dt = DecisionTreeClassifier(random_state=42)
    #dt = DecisionTreeClassifier(random_state=42)
    
    dt.fit(df_clients_test_agreg, df_predictions_test)
    #dt.fit(df_clients_train_agreg, df_predictions_train)
    
    for col, val in sorted(zip(df_clients_test_agreg.columns, dt.feature_importances_), key=lambda col_val: col_val[1], reverse=True):
        print(f'{col:10} {val:10.3f}')
        
    print('Draw decision tree')
    from io import StringIO
    import pydotplus
    dot_data = StringIO()
    tree.export_graphviz(
        dt,
        out_file=dot_data,
        feature_names=df_clients_test_agreg.columns,
        #feature_names=df_clients_train_agreg.columns,
        class_names=df_predictions_test.astype(str).unique(),
        #class_names=df_predictions_train.astype(str).unique(),
        max_depth=2,
        filled=True,
    )
    g = pydotplus.graph_from_dot_data(
        dot_data.getvalue()
    )    
    g.set_size('"6,6!"')
    g.write_png('graph_model' + str(model_number) + '.png',)
    
    #g.write_png('graph_model_current4_nomaxdepth.png') # Uncomment to run only 1 model (and also uncomment previous cell)
    
    # Save box plots of feature distributions
    for feature in df_clients_test_agreg.columns:
        fig, ax = plt.subplots(figsize=(8,6))
        plt.title('Feature distribution of ' + str(feature) + ' on clusters, test set')
        df_plot_test = df_clients_test_agreg.copy()
        df_plot_test['cluster'] = df_predictions_test
        boxplot(x='cluster', y=feature, data=df_plot_test)       

        fig.savefig(f'model{model_number}_featuredistribution_{feature}.png', dpi=300)

    


# # Export to pickle

# ## Export sample input file for the UI

# In[85]:


df_train_ori[['InvoiceNo','StockCode','Quantity','InvoiceDate','CustomerID','TotalPrice','DescriptionNormalized','InvoiceMonth']].sort_values(by='CustomerID').head(1000).to_csv('UI_input_sample.csv')


# ## Retrain model

# In[124]:


df_train = df_train_ori
df_test = df_test_ori    
    
#Model 2:
model_final =  Pipeline([
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    ('clusterer', Clusterer(n_clusters=8, algorithm_to_use='KMEANS'))
    ])


# In[125]:


model_final.fit(df_train)


# In[126]:


model_before_clustering_final = Pipeline([
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
    #('scaler', LogScalerMultiple(features_toscale=['TotalPricePerMonth', 'TotalQuantityPerMonth'])),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[127]:


model_before_clustering_final.fit(df_train)


# In[128]:


#For a model without bag of words
model_agregate_final = Pipeline([
        ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
        #('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'])), \
        ])

'''
#For a model with bag of words
model_agregate_final = Pipeline([
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='ISOMAP', n_dim=3)),
])
'''

model_agregate_final.fit(df_train)


# In[129]:


# Retrain decision tree before export ?
'''
dt = DecisionTreeClassifier(max_depth=4)
dt.fit(df_clients_test_agreg, df_predictions_test)

for col, val in sorted(zip(df_clients_test_agreg.columns, dt.feature_importances_), key=lambda col_val: col_val[1], reverse=True):
    print(f'{col:10} {val:10.3f}')

print('Draw decision tree')
from io import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(
    dt,
    out_file=dot_data,
    feature_names=df_clients_test_agreg.columns,
    #class_names=["0", "1"],
    class_names=df_predictions_test.astype(str).unique(),
    max_depth=4,
    filled=True,
)
g = pydotplus.graph_from_dot_data(
    dot_data.getvalue()
)
#g.write_png('graph_model' + str(model_number) + '.png')
g.write_png('graph_model_final.png') # Uncomment to run only 1 model (and also uncomment previous cell)
'''


# In[130]:


if (SAVE_API_MODEL == True):    
    API_model = {}
    API_model['model'] = model_final
    API_model['model_agregate'] = model_agregate_final
    API_model['model_before_clustering'] = model_before_clustering_final
    
    '''
    API_model = model_final
    '''
    
    with open(API_MODEL_PICKLE_FILE, 'wb') as f:
        pickle.dump(API_model, f, pickle.HIGHEST_PROTOCOL)  


# # Annex

# ## Display some data

# In[866]:


df_nocancel = df[df['InvoiceNo'].str.startswith('C') == False]
df_nocancel.reset_index(inplace=True)

df_gbproduct = df_nocancel[['StockCode', 'TotalPrice']].groupby('StockCode').sum()['TotalPrice']


# In[867]:


df_nocancel.head(2)


# In[868]:


df_nocancel.info()


# In[869]:


invoice_dates = pd.to_datetime(df_nocancel["InvoiceDate"], format="%Y-%m-%d ")


# In[870]:


invoice_dates = pd.to_datetime(df_nocancel["InvoiceDate"])


# In[871]:


np.maximum((pd.to_datetime('2011-12-09 12:50:00') - invoice_dates) / (np.timedelta64(1, "M")), 1)[123456]


# In[ ]:





# In[872]:


df_gbcustom_firstorder = df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()


# In[873]:


df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate']


# In[874]:


(   pd.to_datetime('2011-12-09 12:50:00')   - pd.to_datetime(df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
)\
  / (np.timedelta64(1, "M"))


# In[875]:


# Number of months between first order date and last date of the dataset
series_gbclient_nbmonths = np.maximum((
   (
   pd.to_datetime('2011-12-09 12:50:00')\
   - pd.to_datetime(df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
   )\
    / (np.timedelta64(1, "M"))
), 1)


# In[876]:


df_nocancel[['CustomerID', ]]


# In[877]:


df_gbcustom_firstorder


# In[878]:


df_nocancel[df_nocancel['CustomerID'] == '18281'].sort_values(by='InvoiceDate', ascending=True)


# In[879]:


invoice_dates[2000:2010]


# In[880]:


df_nocancel.loc[2000:2010,'InvoiceDate']


# In[ ]:





# In[881]:


df_nocancel.loc[100000:100010,'InvoiceMonth']


# In[882]:


df[df['InvoiceNo'].str.startswith('C') == True]['CustomerID'].unique()


# In[883]:


# Product codes that contain chars instead of numbers
df[df['StockCode'].str.isalpha()]['StockCode'].unique()


# ## RFM table

# In[884]:


df_train


# quantiles = df_train[['TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']].quantile(q=[0.25,0.5,0.75])
# quantiles = quantiles.to_dict()

# In[ ]:


quantiles


# In[ ]:


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


# In[ ]:


df_rfmtable = df_train[['TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']]


# In[ ]:


df_rfmtable


# In[ ]:


df_rfmtable.loc[:, 'r_quartile'] = df_rfmtable.loc[:, 'Recency'].apply(RScore, args=('Recency',quantiles,))


# In[ ]:


df_rfmtable.loc[:, 'r_quartile'] = df_rfmtable.loc[:, 'Recency'].apply(RScore, args=('Recency',quantiles,))
df_rfmtable.loc[:, 'f_quartile'] = df_rfmtable.loc[:, 'TotalQuantityPerMonth'].apply(FMScore, args=('TotalQuantityPerMonth',quantiles,))
df_rfmtable.loc[:, 'm_quartile'] = df_rfmtable.loc[:, 'TotalPricePerMonth'].apply(FMScore, args=('TotalPricePerMonth',quantiles,))
df_rfmtable.head()


# In[ ]:


quantiles


# In[ ]:


df_rfmtable.loc[:, 'RFMScore'] = df_rfmtable.r_quartile.map(str)                             + df_rfmtable.f_quartile.map(str)                             + df_rfmtable.m_quartile.map(str)
df_rfmtable.head()


# In[ ]:


df_rfmtable.head(1)


# In[ ]:


df_rfmtable


# In[ ]:


df_train['RFMScore'] = df_rfmtable['RFMScore']


# In[ ]:


df_train


# # Feature selection attempt

# In[ ]:


# create the RFE model and select 3 attributes
rfe = RFE(Clusterer(n_clusters=11, algorithm_to_use='WARD'), 3)
rfe = rfe.fit(df_train, rfm_scores_train)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)


# In[109]:


'''
for feature in df_clients_test_agreg.columns:
    fig, ax = plt.subplots(figsize=(8,6))
    plt.title('Feature distribution of ' + str(feature) + ' on clusters, test set')
    df_plot_test = df_clients_test_agreg.copy()
    df_plot_test['cluster'] = df_predictions_test
    boxplot(x='cluster', y=feature, data=df_plot_test)       
    
    fig.savefig(f'modelX_featuredistribution_{feature}.png', dpi=300)
'''


# In[855]:


df_clients_test.dtypes


# In[856]:


tsne = TSNE(n_components=2, random_state=42)


# In[857]:


tsne.fit_transform(df_clients_test.to_numpy())


# In[858]:


filter_cols = [col for col in df_clients_test]


# In[859]:


filter_cols


# In[860]:


filter_cols.sort(key=lambda v: (isinstance(v, str), v))


# In[861]:


df_clients_test_reduced.loc[:,0]


# S'inspirer de : # Model with all features and RFM score (concat) (not individual RFM feats), and NCA (BEST)# 

# In[862]:


model_beforecluster = Pipeline([
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[863]:


model_beforecluster.fit(df_train)


# In[864]:


df_test_transformed = model_beforecluster.transform(df_test)


# In[865]:


df_test_transformed


# In[901]:


trace_1 = go.Scatter(x = df_clients_test_reduced.loc[:,0], y = df_clients_test_reduced.loc[:,1],
                    name = 'Clients',
                    mode = 'markers',
                    marker=dict(color=bow_labels_test),
                    #text = rfm_scores_train,
                    #text = [('Bought top value product' if (boughttopvalueproduct == 1) else 'dit NOT buy top value product') for boughttopvalueproduct in df_train['BoughtTopValueProduct']],
                    text = list(map(str, zip('RFM: ' + df_clients_test_agreg['RfmScore'].astype(str),\
                                             'BoughtTopValueProduct: ' + df_clients_test_agreg['BoughtTopValueProduct'].astype(str),\
                                              'HasEverCancelled: '  + df_clients_test_agreg['HasEverCancelled'].astype(str),\
                                            ))\
                                        )
                    )


layout = go.Layout(title = 'Model ' + str(model_number) + ' : client representation, colored by bow',
                   hovermode = 'closest',
)

fig = go.Figure(data = [trace_1], layout = layout)

#py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_model' + str(model_number) + '.html') 

