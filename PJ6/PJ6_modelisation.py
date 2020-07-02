#!/usr/bin/env python
# coding: utf-8

# # Global settings

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


import nltk
import codecs

#from nltk.corpus.reader.api import CorpusReader
#from nltk.corpus.reader.api import CategorizedCorpusReader

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

import pandas_profiling

from bs4 import BeautifulSoup

DATA_PATH = os.path.join("datasets", "stackexchange")
#DATA_PATH = os.path.join(DATA_PATH, "out")

#DATA_PATH_FILE_INPUT = os.path.join(DATA_PATH, "QueryResults_20190101-20200620.csv")
#DATA_PATH_FILE_INPUT = os.path.join(DATA_PATH, "QueryResults 20200301-20200620_1.csv")

DATA_PATH_FILE = os.path.join(DATA_PATH, "*.csv")
ALL_FILES_LIST = glob.glob(DATA_PATH_FILE)

ALL_FEATURES = []

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

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from nltk.cluster import KMeansClusterer # NLTK algorithm will be useful for cosine distance

SAVE_API_MODEL = True # If True : API model ill be saved
API_MODEL_PICKLE_FILE = 'API_model_PJ6.pickle'


# # Doc2vec settings

# In[2]:


DOC2VEC_TRAINING_SAVE_FILE = 'doc2vec_model'
#doc2vec_fname = get_tmpfile(DOC2VEC_TRAINING_SAVE_FILE)

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.parsing.preprocessing import remove_stopwords

import time

from gensim.test.utils import get_tmpfile

import gensim

#model.save(fname)
#model = Doc2Vec.load(fname)  # you can continue training with the loaded model!


# In[3]:


ALL_FILES_LIST


# # Load data

# In[4]:


import pandas as pd

pd.set_option('display.max_columns', None)

feats_list = ['Title', 'Body', 'Tags']

def load_data(data_path=DATA_PATH):
    csv_path = DATA_PATH_FILE
    df_list = []
    
    for f in ALL_FILES_LIST:
        print(f'Loading file {f}')
        
        df_list.append(pd.read_csv(f, sep=',', header=0, encoding='utf-8', usecols=feats_list))
        
    return pd.concat(df_list)


# In[5]:


df = load_data()
df.reset_index(inplace=True, drop=True)


# In[6]:


df


# ## Drop NA

# In[7]:


df.dropna(subset=['Body'], axis=0, inplace=True)
df.dropna(subset=['Tags'], axis=0, inplace=True)


# In[8]:


df.shape


# # Encode labels (strip < and >, then 1 hot encode)

# In[9]:


# Converting tags from <tag 1><tag2><tag3> to tag1 tag2 tag3
df.loc[:, 'Tags'] = df['Tags'].str.replace('<', '') 
df.loc[:, 'Tags'] = df.loc[:, 'Tags'].str.replace('>', ' ') 
df.loc[:, 'Tags'] = df.loc[:, 'Tags'].str.rstrip()


# In[10]:


df.info()


# In[11]:


#df_train.dropna(subset=['Tags'], axis=0, inplace=True)  # Can be removed later  (NA already dropped on df first place)
#df_test.dropna(subset=['Tags'], axis=0, inplace=True)  # Can be removed later  (NA already dropped on df first place)


# In[12]:


bowencoder = BowEncoder()


# In[13]:


bowencoder.fit(df, categorical_features_totransform=['Tags'])


# In[14]:


df = bowencoder.transform(df)


# In[15]:


df[['Body', 'Tags', 'Tags_javascript', 'Tags_jquery', 'Tags_python', 'Tags_html', 'Tags_java', 'Tags_docker', 'Tags_android', 'Tags_cordova']]


# In[16]:


filter_col_labels = [col for col in df if col.startswith('Tags')]


# In[17]:


df_labels = df[filter_col_labels].copy(deep=True)


# In[18]:


df_labels.drop(columns=['Tags'], inplace=True)


# In[19]:


df_labels


# In[20]:


df.drop(columns=filter_col_labels, inplace=True)


# In[21]:


df_labels.shape


# # Split training set, test set, and split labels

# In[22]:


df, df_train, df_test, df_train_labels, df_test_labels = custom_train_test_split_with_labels(df, df_labels, None)


# In[23]:


df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
df_train_labels.reset_index(drop=True, inplace=True)
df_test_labels.reset_index(drop=True, inplace=True)


# In[24]:


df_train


# In[25]:


df_test


# In[26]:


df_train_labels


# In[27]:


df_test_labels


# In[28]:


df_train_ori = df_train.copy(deep=True)
df_test_ori = df_test.copy(deep=True)


# In[29]:


df_train.shape


# In[30]:


df_test.shape


# In[31]:


df_train_labels.shape


# In[32]:


df_test_labels.shape


# # Prepare text data (remove html in Body, and regroup Body + title)

# In[33]:


df_train = df_train_ori
df_test = df_test_ori


# In[34]:


dataprep = PrepareTextData()


# In[35]:


df_train = dataprep.fit_transform(df_train)


# In[36]:


df_test = dataprep.transform(df_test)


# In[37]:


df_train


# In[38]:


df_test


# # Doc2Vec training (launch only the 1st time)

# In[67]:


cnt_label = 0
InputDocs = []
for document in df_train['all_text']:  # TO DO : relaunch this training with df_train
    #InputDocs.append(TaggedDocument(document,[cnt_label]))
    
    doc_transformed = remove_stopwords(document)
    doc_toappend = gensim.utils.simple_preprocess(doc_transformed)
    
    InputDocs.append(TaggedDocument(doc_toappend,[cnt_label]))    
    cnt_label += 1


# In[68]:


InputDocs


# In[69]:


start = time.time()
model_doc2vec = Doc2Vec(InputDocs, vector_size=200, window=5, min_count=5, workers=4)  # All input docs loaded in memory
end = time.time()

print('Durée doc2vec training: ' + str(end - start) + ' secondes')    


# In[70]:


#model_doc2vec.save(doc2vec_fname)
model_doc2vec.save(DOC2VEC_TRAINING_SAVE_FILE)


# In[71]:


TaggedDocument(gensim.utils.simple_preprocess(df_train.iloc[0]['all_text']), [0])


# In[72]:


gensim.utils.simple_preprocess("Hello this is a new text")


# In[73]:


[model_doc2vec.infer_vector(gensim.utils.simple_preprocess(text)) for text in ['hello this is', 'second text']]


# In[49]:


#a = [document for document in df_train.loc[:,'all_text'] ] 


# In[51]:


#df_train.shape


# In[50]:


#len(a)


# In[74]:


#X_vectorized = [model_doc2vec.infer_vector(TaggedDocument(document)) for document in df_train.loc[:, 'all_text']]  # Too slow on training set


# In[76]:


#X_vectorized


# # Doc2vec loading

# In[39]:


df_train = df_train_ori
df_test = df_test_ori


# In[40]:


df


# In[41]:


from functions import *
importlib.reload(sys.modules['functions'])
from functions import *


# In[42]:


doc2vec = Doc2Vec_Vectorizer(model_path=DOC2VEC_TRAINING_SAVE_FILE, feature_totransform='all_text')


# In[43]:


doc2vec.fit(df_train)


# In[44]:


doc2vec.model.docvecs.vectors_docs


# In[45]:


# Use this to infer vectors :  needed for test set, but not mandatory for training set (we alredy have trained vectors available)
#df_train_transformed = doc2vec.transform(df_train.loc[0:1000, :])
#df_train_transformed = doc2vec.transform(df_train)

# Use this to get already trained vectors in training set
df_train_transformed = doc2vec.model.docvecs.vectors_docs


# In[46]:


df_train_transformed.shape


# In[47]:


df_train_transformed


# In[48]:


df_train = df_train_transformed


# # First clustering attempts (launch only once)

# In[37]:


df_train.shape


# In[68]:


df_train_ori.info()


# In[69]:


df_train


# In[34]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_train)
                for k in range(1, 20)]


# In[39]:


'''
kmeans_model_nltk = KMeansClusterer(
            10, distance=nltk.cluster.util.cosine_distance, avoid_empty_clusters=True, repeats=10) 
'''


# In[40]:


'''
clusters = kmeans_model_nltk.cluster(df_train.to_numpy(), assign_clusters = True)    
'''


# In[41]:


kmeans_nltk_per_k = [KMeansClusterer(
            k, distance=nltk.cluster.util.cosine_distance, avoid_empty_clusters=True, repeats=10) for k in range(1,20)]


# In[ ]:


# If df_train is a dataframe  (when vectors have been infered) :
#kmeans_nltk_labels_train_per_k = [model.cluster(df_train.to_numpy(), assign_clusters = True) for model in kmeans_nltk_per_k]

# If df_train is an ndarray (when we directly got training labels):
kmeans_nltk_labels_train_per_k = [model.cluster(df_train, assign_clusters = True) for model in kmeans_nltk_per_k]


# In[ ]:


#labels_test_per_k = [model.predict(df_test) for model in kmeans_per_k[1:]]


# In[35]:


silhouette_scores = [silhouette_score(df_train, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[ ]:


silhouette_scores_nltk = [silhouette_score(df_train, labels)
                     for labels in kmeans_nltk_labels_train_per_k[1:]]


# In[247]:


#silhouette_scores_test = [silhouette_score(df_test, labels_test) for labels_test in labels_test_per_k]


# In[36]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 20), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 20), silhouette_scores_nltk, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score of KMeans NLTK (with cosine distance) on training set", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# plt.figure(figsize=(8, 3))
# plt.plot(range(2, 50), silhouette_scores_test, "bo-")
# plt.xlabel("$k$", fontsize=14)
# plt.ylabel("Silhouette score on test set", fontsize=14)
# #plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
# #save_fig("silhouette_score_vs_k_plot")
# plt.show()

# # Compare 1 document to closest neighbours

# In[61]:


df_train_ori


# In[63]:


df_train.shape


# In[64]:


df_train_ori.shape


# In[49]:


doc_to_compare = df_train_ori.loc[500]['Body']


# In[50]:


gensim.utils.simple_preprocess(remove_stopwords(doc_to_compare))


# In[51]:


print(doc_to_compare)


# In[52]:


doc2vec.model.infer_vector(gensim.utils.simple_preprocess(remove_stopwords(doc_to_compare)))


# In[56]:


doc2vec.model.docvecs.most_similar([doc2vec.model.infer_vector(gensim.utils.simple_preprocess(remove_stopwords(doc_to_compare)))])


# In[60]:


print(df_train_ori.loc[266450]['Body'])


# # First implementation of a KNN classification algorithm

# In[68]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[69]:


df_train = df_train_ori
df_test = df_test_ori


# In[70]:


preparation_pipeline = Pipeline([
    ('scaler', StandardScalerMultiple(features_toscale=['TotalPricePerMonth'])),
    
    
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


# In[ ]:





# # Annex (old code)

# ## Remove html tags in Body, and regroup Body + title

# In[82]:


dataprep = PrepareTextData()


# In[83]:


df = dataprep.fit_transform(df)


# ## Or with beautifulsoup
# df.loc[:, 'Body'] = df['Body'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

# In[93]:


pd.set_option('display.max_colwidth', None)
print(df.loc[45000]['all_text'])


# In[10]:


# Converting tags from <tag 1><tag2><tag3> to tag1 tag2 tag3
df.loc[:, 'Tags'] = df['Tags'].str.replace('<', '') 
df.loc[:, 'Tags'] = df.loc[:, 'Tags'].str.replace('>', ' ') 
df.loc[:, 'Tags'] = df.loc[:, 'Tags'].str.rstrip()


# In[ ]:





# In[11]:


df.info()


# In[12]:


df.sample(100)


# ## Regroup text features and clean

# df.loc[:, 'Title'].fillna(value='', inplace=True)

# df['all_text'] = df['Title'].astype(str) + '. ' +  df['Body'].astype(str)

# ## Split training set, test set (old)

# In[104]:


df, df_train, df_test = custom_train_test_split_sample(df, None)


# In[105]:


df_train.reset_index(drop=True, inplace=True)


# In[106]:


df_train


# In[107]:


df_test


# In[108]:


df_test.reset_index(drop=True, inplace=True)


# df['all_text']

# In[ ]:




