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


# In[ ]:


df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, test set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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

# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


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
    


# In[ ]:


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


# In[ ]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[ ]:


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


# ## 2nd clustering : ward

# In[ ]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[ ]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[ ]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with only bow features and TotalPricePerMonth, TSNE

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[ ]:


df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, test set, BoW')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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

# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


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
    


# In[ ]:


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


# In[ ]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[ ]:


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


# ## 2nd clustering : ward

# In[ ]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[ ]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[ ]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with bow features + TotalPricePerMonth

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[ ]:


df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[ ]:





# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW + TotalPricePerMonth')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[ ]:


df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[ ]:





# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set, BoW + TotalPricePerMonth + HasEverCancelled')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


pca = PCA(n_components=2,random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, all feats except bow, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, all feats except bow, test set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_test_transformed[:,0], X_test_transformed[:,1], c=df_score_cat_test)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


df_train.iloc[:, 0]


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


tsne = TSNE(n_components=3, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


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
    


# In[ ]:


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
    


# In[ ]:


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


# In[ ]:


print('Gini before clustering :')
gini(df_train['TotalPricePerMonth'].to_numpy())


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), gini_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean gini score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), gini_mean_score_per_k_test, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean gini score on test set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[ ]:


print('Entropy before clustering :')
entropy(df_train['TotalPricePerMonth'])


# In[ ]:


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


# # Model with tSNE, then clustering algorithm Ward
# No visualisation on test set because AgglomerativeClustering has no predict function, only fit_predict

# In[ ]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[ ]:


gini_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    gini_sum = 0
    for unique_label in unique_labels:
        gini_sum += gini(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    gini_sum = gini_sum / len(unique_labels)
    
    gini_mean_score_per_k_train.append(gini_sum)


    


# In[ ]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), gini_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean gini score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[ ]:


print('Entropy before clustering :')
entropy(df_train['TotalPricePerMonth'])


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Model with tSNE, then clustering algorithm Ward, distance threshold
# No visualisation on test set because AgglomerativeClustering has no predict function, only fit_predict

# In[ ]:


np.unique(AgglomerativeClustering(distance_threshold=1, n_clusters=None, affinity='euclidean').fit(df_train).labels_)


# In[ ]:


clusterer_ward_per_thr = [AgglomerativeClustering(distance_threshold=thr, n_clusters=None, affinity='euclidean').fit(df_train) for thr in reversed(range(0,12))]


# In[ ]:


np.unique(clusterer_ward.labels_)


# In[ ]:


clusterer_ward_per_thr


# In[ ]:


entropy_mean_score_per_k_train = []

for model in clusterer_ward_per_thr[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(df_train['TotalPricePerMonth'][model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(1,12), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$Ward threshold$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # => Around 50 clusters => entropy of TotalPrice around 4.5

# # Prepation model with LLE reduce to 200, then clustering algorithm Ward

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[ ]:


df_test = preparation_pipeline.transform(df_test)


# In[ ]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test) for model in clusterer_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


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


# In[ ]:


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


# In[ ]:


print('Entropy before clustering :')
entropy(df_train['TotalPricePerMonth'])


# In[ ]:


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


# # Prepation model with LLE reduce to 3, then clustering algorithm Ward

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[ ]:


df_test = preparation_pipeline.transform(df_test)


# In[ ]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test) for model in clusterer_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


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


# In[ ]:


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


# In[ ]:


print('Entropy before clustering :')
entropy(df_train['TotalPricePerMonth'])


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[ ]:


df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


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
    


# In[ ]:


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


# In[ ]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[ ]:


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


# ## 2nd clustering : Ward

# In[ ]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[ ]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[ ]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Preparation model BoW feats only, then LLE reduce to 3 then KMeans and Ward

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)


# In[ ]:


df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train


# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


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
    


# In[ ]:


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


# In[ ]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[ ]:


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


# ## 2nd clustering : ward

# In[ ]:


clusterer_per_k = [AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df_train) for k in range(1,50)]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in clusterer_per_k[1:]]


# In[ ]:


entropy_mean_score_per_k_train = []

for model in clusterer_per_k[1:]:
    unique_labels = np.unique(model.labels_)
    
    entropy_sum = 0
    for unique_label in unique_labels:
        entropy_sum += entropy(series_total_price_per_month_train[model.labels_ == unique_label].to_numpy())
        
    entropy_sum = entropy_sum / len(unique_labels)
    
    entropy_mean_score_per_k_train.append(entropy_sum)


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[ ]:


print('Entropy before clustering :')
entropy(series_total_price_per_month_train)


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Plot representation

# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


rfm_scores_train = get_rfm_scores(df_train)


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[ ]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


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


# In[ ]:


## Add bow coloration


# In[ ]:


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

# In[ ]:


df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 0, 1, 2]]


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 0, 1, 2]])
X_test_transformed = tsne.fit_transform(df_test[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 0, 1, 2]])


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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


# In[ ]:


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

# In[ ]:


df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 0, 1, 2]]


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 0, 1, 2]])
X_test_transformed = tsne.fit_transform(df_test[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 0, 1, 2]])


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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


# In[ ]:


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

# In[ ]:


df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth']]


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth']])
X_test_transformed = tsne.fit_transform(df_test[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth']])


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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


# In[ ]:


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

# In[ ]:


df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled']]


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled']])
X_test_transformed = tsne.fit_transform(df_test[['TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled']])


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")
#plt.yscale('log')


# In[ ]:


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


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


rfm_scores_train = df_train['RfmScore']


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[ ]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


rfm_scores_train = df_train['RfmScore']


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[ ]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


#rfm_scores_train = df_train['RfmScore']  # we reuse rfm_scores_train calculated on above model


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[ ]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


#rfm_scores_train = df_train['RfmScore']  # we reuse rfm_scores_train calculated on above model


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[ ]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


rfm_scores_train = df_train['RfmScore']


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[ ]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


'''
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_transformed = lle.fit_transform(df_train)
X_test_transformed = lle.fit_transform(df_test)
'''
tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(df_train)
X_test_transformed = tsne.fit_transform(df_test)


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


rfm_scores_train = df_train['RfmScore']


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[ ]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


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


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


rfm_scores_train = df_train['RfmScore']


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[ ]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


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


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, TotalPricePerMonth colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


rfm_scores_train = df_train['RfmScore']


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[ ]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


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


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, TotalPricePerMonth colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


#rfm_scores_train = df_train['RfmScore'] # rfm_scores_train value has been got from code above


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())


# In[ ]:


rfm_dict_colors = {}
cnt = 0

for unique_rfm_score in unique_rfm_scores:
    rfm_dict_colors[unique_rfm_score] = cnt
    cnt += 1
    


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


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


# In[ ]:


X_transformed[:,1]


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, TotalPricePerMonth colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=df_score_cat_train)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


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

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


rfm_scores_train = df_train['RfmScore']
rfm_scores_test = df_test['RfmScore']


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())
unique_rfm_scores_test = np.sort(rfm_scores_test.unique())


# In[ ]:


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


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])
rfm_scores_test_colors = rfm_scores_test.apply(lambda x : rfm_dict_colors_test[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_test_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


'''
df_train_rfmscore_distances = pairwise_distances(df_train['RfmScore'].to_numpy().reshape(-1, 1))
df_test_rfmscore_distances = pairwise_distances(df_test['RfmScore'].to_numpy().reshape(-1, 1))
'''


# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[ ]:


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
    


# In[ ]:


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


# In[ ]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[ ]:


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


# ## Reduce and visualize

# In[ ]:


nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
X_transformed = nca.fit_transform(df_train, pd.cut(df_train['RfmScore'], bins=range(1,10), right=True).astype(str).tolist())
X_test_transformed = nca.transform(df_test)


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, cluster label colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_transformed)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(X_test_transformed) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(X_transformed, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(X_test_transformed, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


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


# In[ ]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[ ]:


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


# # Model with 'DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore' (SUM) with NCA up to 200 then KMeans then NCA to visualize clusters   (GOOD, ONLY NCA not TSNE)

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


rfm_scores_train = df_train['RfmScore']
rfm_scores_test = df_test['RfmScore']


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())
unique_rfm_scores_test = np.sort(rfm_scores_test.unique())


# In[ ]:


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


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])
rfm_scores_test_colors = rfm_scores_test.apply(lambda x : rfm_dict_colors_test[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_test_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


'''
df_train_rfmscore_distances = pairwise_distances(df_train['RfmScore'].to_numpy().reshape(-1, 1))
df_test_rfmscore_distances = pairwise_distances(df_test['RfmScore'].to_numpy().reshape(-1, 1))
'''


# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[ ]:


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
    


# In[ ]:


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


# In[ ]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[ ]:


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


# ## Reduce and visualize

# In[ ]:


nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
X_transformed = nca.fit_transform(df_train, pd.cut(df_train['RfmScore'], bins=range(1,10), right=True).astype(str).tolist())
X_test_transformed = nca.transform(df_test)


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, cluster label colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_transformed)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(X_test_transformed) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(X_transformed, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(X_test_transformed, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


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


# In[ ]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[ ]:


df_test


# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


clusterer = Clusterer(n_clusters=11, algorithm_to_use='WARD')
clusterer.fit(df_train)


# In[ ]:


cluster_labels_test = clusterer.predict(df_test)


# In[ ]:


cluster_labels_train = clusterer.predict(df_train)


# In[ ]:


df_train.loc[0, :]


# In[ ]:


df_train[df_train.index == 0]


# In[ ]:


df_train[df_train.index=df_train.loc[0, :]]


# In[ ]:


instance_prediction = clusterer.predict(df_train[df_train.index == 0])


# In[ ]:


instance_prediction


# In[ ]:


len(cluster_labels_test)


# In[ ]:


df_train.loc[0, :]


# In[ ]:


cluster_labels_test


# In[ ]:


df_train


# In[ ]:


cluster_labels_train


# In[ ]:


clusterer.clusterer.labels_


# In[ ]:


clusterer.score(df_train)


# In[ ]:


silhouette_scores_test


# # Model with 'TotalPricePerMonth', 'Recency', 'TotalQuantityPerMonth', 'BoughtTopValueProduct', 'HasEverCancelled' with NCA up to 200 then KMeans then NCA to visualize clusters

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


#rfm_scores_train = df_train['RfmScore']  # Got from code above
#rfm_scores_test = df_test['RfmScore']


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())
unique_rfm_scores_test = np.sort(rfm_scores_test.unique())


# In[ ]:


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


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])
rfm_scores_test_colors = rfm_scores_test.apply(lambda x : rfm_dict_colors_test[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_test_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


'''
df_train_rfmscore_distances = pairwise_distances(df_train['RfmScore'].to_numpy().reshape(-1, 1))
df_test_rfmscore_distances = pairwise_distances(df_test['RfmScore'].to_numpy().reshape(-1, 1))
'''


# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


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


# ## Reduce and visualize

# In[ ]:


nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
X_transformed = nca.fit_transform(df_train, pd.cut(rfm_scores_train, bins=range(1,10), right=True).astype(str).tolist())
X_test_transformed = nca.transform(df_test)


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, RFM colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=rfm_scores_train_colors)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_transformed)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(X_test_transformed) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(X_transformed, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(X_test_transformed, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


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


# # Model with 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency' then KMeans then 3D visualisation

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


#rfm_scores_train = df_train['RfmScore']  # Got from above
#rfm_scores_test = df_test['RfmScore']


# In[ ]:


unique_rfm_scores = np.sort(rfm_scores_train.unique())
unique_rfm_scores_test = np.sort(rfm_scores_test.unique())


# In[ ]:


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


# In[ ]:


rfm_scores_train_colors = rfm_scores_train.apply(lambda x : rfm_dict_colors[x])
rfm_scores_test_colors = rfm_scores_test.apply(lambda x : rfm_dict_colors_test[x])


# In[ ]:


rfm_scores_train_colors


# In[ ]:


rfm_scores_test_colors


# In[ ]:


rfm_scores_train


# In[ ]:


df_train


# In[ ]:


df_train.info()


# In[ ]:


'''
df_train_rfmscore_distances = pairwise_distances(df_train['RfmScore'].to_numpy().reshape(-1, 1))
df_test_rfmscore_distances = pairwise_distances(df_test['RfmScore'].to_numpy().reshape(-1, 1))
'''


# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[ ]:


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


# ## Visualize

# In[ ]:


df_train


# In[ ]:


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


# In[ ]:


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


# In[ ]:


fig = plt.figure()
fig.suptitle('Customers 2d representation, cluster label colored, training set')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_)
#ax.set_xlim([0,500])
plt.xlabel('Axe 1')
plt.ylabel("Axe 2")

#plt.yscale('log')


# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_transformed)
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(X_test_transformed) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(X_transformed, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(X_test_transformed, labels_test) for labels_test in labels_test_per_k]


# In[ ]:


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


# In[ ]:


print('Entropy before clustering :')
entropy(df_train['RfmScore'])


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), entropy_mean_score_per_k_train, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Mean entropy score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# # Correlations

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


corr_matrix = df_train.corr()


# In[ ]:


corr_matrix


# In[ ]:


plt.title('Corrélation entre les features')
sns.heatmap(corr_matrix, 
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.columns, cmap='coolwarm' ,center=0.20)


# In[ ]:


'''
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols("BMI ~ 0 + 1 + 2", data=df_train).fit()
#print model.params
#print model.summary()
'''


# # Generate bow colors

# In[ ]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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


# In[ ]:


df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)


# In[ ]:


df_train.loc[:, 0].to_numpy()


# In[ ]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train.loc[:, 0].to_numpy().reshape(-1,1))
                for k in range(1, 50)]


# In[ ]:


labels_test_per_k = [model.predict(df_test.loc[:, 0].to_numpy().reshape(-1,1)) for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores = [silhouette_score(df_train.loc[:, 0].to_numpy().reshape(-1,1), model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_test = [silhouette_score(df_test.loc[:, 0].to_numpy().reshape(-1,1), labels_test) for labels_test in labels_test_per_k]


# In[ ]:


# Model corresponding to max silhouette score. We add +1 because "for model in kmeans_per_k[1:] above has suppressed one indice"
# kmeans_per_k[silhouette_scores.index(max(silhouette_scores)) + 1].labels_


# In[ ]:


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
    


# In[ ]:


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


# In[ ]:


bow_labels_train = kmeans_per_k[10].labels_


# In[ ]:


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

# In[ ]:


from functions import *
importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# ## Agregate to client level to get text labels for visualisation, and interprete model with surrogate Decision Tree

# In[ ]:


#model_agregate = AgregateToClientLevel(top_value_products, compute_rfm=True)
model_agregate = Pipeline([
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='ISOMAP', n_dim=3)),
])


# In[ ]:


model_agregate.fit(df_train)


# In[ ]:


df_clients_train_agreg = model_agregate.transform(df_train)


# In[ ]:


df_clients_test_agreg = model_agregate.transform(df_test)


# In[ ]:


df_clients_test_agreg


# In[ ]:


df_clients_test_agreg.shape


# ## Loop on top 4 models

# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


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

        ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='ISOMAP', n_dim=3)),
        ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('clusterer', Clusterer(n_clusters=6, algorithm_to_use='KMEANS'))
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

        ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                                algorithm_to_use='ISOMAP', n_dim=3)),
        ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
        ]),
]

models_clustering = [
        Clusterer(n_clusters=4, algorithm_to_use='WARD'),
        Clusterer(n_clusters=8, algorithm_to_use='KMEANS'),
        Clusterer(n_clusters=4, algorithm_to_use='WARD'),
        Clusterer(n_clusters=6, algorithm_to_use='KMEANS')
]


# In[ ]:


model_number = 0

for model_prep, model_clusterer in zip(models_before_clustering, models_clustering):
    model_number += 1
    df_train = df_train_ori
    df_test = df_test_ori    
    
    print('Model ' + str(model_number) + ':\n' + str(model_prep) + '\n' + str(model_clusterer))
    
    print('Running preparation model')
    
    if (model_number == 3): # For NCA we must pass a dimensionality reductor label
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
    dt = DecisionTreeClassifier()
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
        max_depth=2,
        filled=True,
    )
    g = pydotplus.graph_from_dot_data(
        dot_data.getvalue()
    )
    g.write_png('graph_model' + str(model_number) + '.png')

    


# In[ ]:


df_predictions_test.astype(str).unique()


# In[ ]:


reductor


# In[ ]:


df_clients_test.dtypes


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)


# In[ ]:


tsne.fit_transform(df_clients_test.to_numpy())


# In[ ]:


filter_cols = [col for col in df_clients_test]


# In[ ]:


filter_cols


# In[ ]:


filter_cols.sort(key=lambda v: (isinstance(v, str), v))


# In[ ]:


df_clients_test_reduced.loc[:,0]


# S'inspirer de : # Model with all features and RFM score (concat) (not individual RFM feats), and NCA (BEST)# 

# In[ ]:


model_beforecluster = Pipeline([
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['BoughtTopValueProduct', 'HasEverCancelled', 'RfmScore'])),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
])


# In[ ]:


model_beforecluster.fit(df_train)


# In[ ]:


df_test_transformed = model_beforecluster.transform(df_test)


# In[ ]:


df_test_transformed


# A FAIRE : renvoyer un dataframe en sorte de predict() du modèle final, indexé avec les identifiants de client

# # Annex

# ## Display some data

# In[ ]:


df_nocancel = df[df['InvoiceNo'].str.startswith('C') == False]
df_nocancel.reset_index(inplace=True)

df_gbproduct = df_nocancel[['StockCode', 'TotalPrice']].groupby('StockCode').sum()['TotalPrice']


# In[ ]:


df_nocancel.head(2)


# In[ ]:


df_nocancel.info()


# In[ ]:


invoice_dates = pd.to_datetime(df_nocancel["InvoiceDate"], format="%Y-%m-%d ")


# In[ ]:


invoice_dates = pd.to_datetime(df_nocancel["InvoiceDate"])


# In[ ]:


np.maximum((pd.to_datetime('2011-12-09 12:50:00') - invoice_dates) / (np.timedelta64(1, "M")), 1)[123456]


# In[ ]:





# In[ ]:


df_gbcustom_firstorder = df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()


# In[ ]:


df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate']


# In[ ]:


(   pd.to_datetime('2011-12-09 12:50:00')   - pd.to_datetime(df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
)\
  / (np.timedelta64(1, "M"))


# In[ ]:


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


# In[ ]:


df_gbcustom_firstorder


# In[ ]:


df_nocancel[df_nocancel['CustomerID'] == '18281'].sort_values(by='InvoiceDate', ascending=True)


# In[ ]:


invoice_dates[2000:2010]


# In[ ]:


df_nocancel.loc[2000:2010,'InvoiceDate']


# In[ ]:





# In[ ]:


df_nocancel.loc[100000:100010,'InvoiceMonth']


# In[ ]:


df[df['InvoiceNo'].str.startswith('C') == True]['CustomerID'].unique()


# In[ ]:


# Product codes that contain chars instead of numbers
df[df['StockCode'].str.isalpha()]['StockCode'].unique()


# ## RFM table

# In[ ]:


df_train


# In[ ]:


quantiles = df_train[['TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']].quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


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

