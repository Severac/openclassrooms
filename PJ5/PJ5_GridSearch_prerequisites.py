#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:23:51 2020

@author: francois

PJ5 Openclassrooms : this script is prerequisites for Grid search validation and saves results
To be used as prerequisite of PJ5_GridSearch1.py

"""

#%matplotlib inline

#%load_ext autoreload  # Autoreload has a bug : when you modify function in source code and run again, python kernel hangs :(
#%autoreload 2

import datetime as dt

import sys, importlib

from functions import *
#importlib.reload(sys.modules['functions'])

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


RECOMPUTE_GRIDSEARCH = True  # CAUTION : computation is several hours long
SAVE_GRID_RESULTS = True # If True : grid results object will be saved to pickle files that have GRIDSEARCH_FILE_PREFIX
LOAD_GRID_RESULTS = False # If True : grid results object will be loaded from pickle files that have GRIDSEARCH_FILE_PREFIX

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

# Load data
print('Load data')

df = load_data(DATA_PATH_FILE_INPUT)

df, df_train, df_test = custom_train_test_split_sample(df, 'TotalPrice')
df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)
df_train_ori = df_train.copy(deep=True)
df_test_ori = df_test.copy(deep=True)

# Top value products (must be saved with the model, and passed to it)
print('Top value products (must be saved with the model, and passed to it)')
df_nocancel = df_train[df_train['InvoiceNo'].str.startswith('C') == False]
df_nocancel.reset_index(inplace=True)

df_gbproduct = df_nocancel[['StockCode', 'TotalPrice']].groupby('StockCode').sum()['TotalPrice']

TOP_VALUE_PRODUCT_THRESHOLD = 20
top_value_products = df_gbproduct.sort_values(ascending=False).head(TOP_VALUE_PRODUCT_THRESHOLD).index  # Get top value products

# Preparation pipeline : model with bow features + TotalPricePerMonth + BoughtTopValueProduct + HasEverCancelled
print('Preparation pipeline : model with bow features + TotalPricePerMonth + BoughtTopValueProduct + HasEverCancelled')

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
    
df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)

series_total_price_per_month_train = df_train['TotalPricePerMonth']
series_total_price_per_month_test = df_test['TotalPricePerMonth']

series_hasevercancelled_train = df_train['HasEverCancelled']
series_hasevercancelled_test = df_test['HasEverCancelled']

series_boughttopvalueproduct_train = df_train['BoughtTopValueProduct']
series_boughttopvalueproduct_test = df_test['BoughtTopValueProduct']

# Generation of TotalPricePerMonth labels
pca = PCA(n_components=2, random_state=42)
X_transformed = pca.fit_transform(df_train)
X_test_transformed = pca.fit_transform(df_test)

print('Binarisation of color categories')
bins = [-np.inf,df_train['TotalPricePerMonth'].quantile(0.25),\
        df_train['TotalPricePerMonth'].quantile(0.50),\
        df_train['TotalPricePerMonth'].quantile(0.75),\
        df_train['TotalPricePerMonth'].quantile(1)]

labels = [0, 1, 2, 3]

df_score_cat_train = pd.cut(df_train['TotalPricePerMonth'], bins=bins, labels=labels)


bins = [-np.inf,df_test['TotalPricePerMonth'].quantile(0.25),\
        df_test['TotalPricePerMonth'].quantile(0.50),\
        df_test['TotalPricePerMonth'].quantile(0.75),\
        df_test['TotalPricePerMonth'].quantile(1)]

labels = [0, 1, 2, 3]

df_score_cat_test = pd.cut(df_test['TotalPricePerMonth'], bins=bins, labels=labels)

# Generation of Bag Of Words labels and columns
print('Generation of Bag Of Words labels and columns')

df_train = df_train_ori
df_test = df_test_ori

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
    
df_train = preparation_pipeline.fit_transform(df_train)
df_test = preparation_pipeline.transform(df_test)

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train.loc[:, 0].to_numpy().reshape(-1,1))
                for k in range(1, 50)]

labels_test_per_k = [model.predict(df_test.loc[:, 0].to_numpy().reshape(-1,1)) for model in kmeans_per_k[1:]]

silhouette_scores = [silhouette_score(df_train.loc[:, 0].to_numpy().reshape(-1,1), model.labels_)
                     for model in kmeans_per_k[1:]]

silhouette_scores_test = [silhouette_score(df_test.loc[:, 0].to_numpy().reshape(-1,1), labels_test) for labels_test in labels_test_per_k]

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
    
bow_labels_train = kmeans_per_k[10].labels_

bow_labels_test = kmeans_per_k[10].predict(df_test.loc[:, 0].to_numpy().reshape(-1,1))