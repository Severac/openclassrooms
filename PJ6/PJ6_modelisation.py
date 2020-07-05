#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ6 : Categorize answers to questions :  modelisation notebook 

# # Global settings

# In[67]:


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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn import tree

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

from yellowbrick.classifier import ROCAUC
from sklearn.metrics import roc_auc_score

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


LEARNING_CURVE_STEP_SIZE = 100


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

df_train_labels_ori = df_train_labels.copy(deep=True)
df_test_labels_ori = df_test_labels.copy(deep=True)


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

# In[ ]:


cnt_label = 0
InputDocs = []
for document in df_train['all_text']:  # TO DO : relaunch this training with df_train
    #InputDocs.append(TaggedDocument(document,[cnt_label]))
    
    doc_transformed = remove_stopwords(document)
    doc_toappend = gensim.utils.simple_preprocess(doc_transformed)
    
    InputDocs.append(TaggedDocument(doc_toappend,[cnt_label]))    
    cnt_label += 1


# In[ ]:


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

# In[ ]:


df_train = df_train_ori
df_test = df_test_ori


# In[ ]:


df


# In[ ]:


from functions import *
importlib.reload(sys.modules['functions'])
from functions import *


# In[ ]:


doc2vec = Doc2Vec_Vectorizer(model_path=DOC2VEC_TRAINING_SAVE_FILE, feature_totransform='all_text')


# In[ ]:


doc2vec.fit(df_train)


# In[ ]:


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

# In[39]:


df_train.shape


# In[40]:


df_train_ori.info()


# In[41]:


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

# In[48]:


df_train_ori


# In[49]:


df_train.shape


# In[50]:


df_train_ori.shape


# In[122]:


doc_to_compare = df_train_ori.loc[1000]['Body']


# In[123]:


print(doc_to_compare)


# In[124]:


doc2vec.model.docvecs.most_similar([doc2vec.model.infer_vector(gensim.utils.simple_preprocess(remove_stopwords(doc_to_compare)))])


# In[125]:


doc_ids = [doc[0] for doc in doc2vec.model.docvecs.most_similar([doc2vec.model.infer_vector(gensim.utils.simple_preprocess(remove_stopwords(doc_to_compare)))])]


# In[126]:


doc_ids_str = [str(doc_id) for doc_id in doc_ids]


# In[127]:


doc_ids_str


# In[128]:


col_names_with_value_1 = [col for col in df_train_labels[df_train_labels.index.isin(doc_ids_str)]                          if (df_train_labels[df_train_labels.index.isin(doc_ids_str)][col] == 1).any()]


# In[129]:


df_train_labels[df_train_labels.index.isin(doc_ids_str)]    .loc[:,col_names_with_value_1]


# # First implementation of a KNN classification algorithm

# In[39]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[40]:


df_train = df_train_ori
df_test = df_test_ori

df_train_labels = df_train_labels_ori
df_test_labels = df_test_labels_ori


# In[41]:


'''
df_train = df_train.loc[0:1000, :]
df_train_labels = df_train_labels.loc[0:1000, :]
'''


# In[42]:


df_train


# In[43]:


prediction_pipeline = Pipeline([
    ('doc2vec', Doc2Vec_Vectorizer(model_path=DOC2VEC_TRAINING_SAVE_FILE, feature_totransform='all_text')),
    #('features_selector', FeaturesSelector(features_toselect=['Tags'])),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(),)
    ])


# In[44]:


prediction_pipeline.fit(df_train, df_train_labels)


# In[55]:


predictions_train = prediction_pipeline.predict(df_train)


# In[56]:


df_predictions_train = pd.DataFrame(predictions_train, columns=df_train_labels.columns)


# In[45]:


predictions_test = prediction_pipeline.predict(df_test)


# In[46]:


df_predictions_test = pd.DataFrame(predictions_test, columns=df_test_labels.columns)


# In[57]:


df_predictions_train


# In[47]:


df_predictions_test


# In[51]:


df_train


# In[52]:


print(df_train.loc[0, 'all_text'])


# In[53]:


doc_index = 0
col_names_tags_value_1 = [col for col in df_train_labels[df_predictions_train.index.isin([doc_index])]                          if (df_predictions_train[df_train_labels.index.isin([doc_index])][col] == 1).any()]


# In[54]:


df_train.loc[df_predictions_train[df_predictions_train['Tags_python'] == 1].index, :]


# In[55]:


col_names_tags_value_1


# In[59]:


df_train_labels


# In[60]:


df_predictions_train


# ## Performance measures

# In[61]:


precision_score(df_train_labels, df_predictions_train, average='micro')


# In[64]:


precision_score(df_train_labels, df_predictions_train, average='macro')


# In[72]:


# Shows exact matchs of all tags
accuracy_score(df_train_labels, df_predictions_train)


# In[62]:


recall_score(df_train_labels, df_predictions_train, average='micro')


# In[63]:


recall_score(df_train_labels, df_predictions_train, average='macro')


# In[67]:


roc_auc_score(df_train_labels, df_predictions_train)


# In[48]:


precision_score(df_test_labels, df_predictions_test, average='micro')


# In[49]:


precision_score(df_test_labels, df_predictions_test, average='macro')


# In[50]:


# Shows exact matchs of all tags
accuracy_score(df_test_labels, df_predictions_test)


# In[51]:


recall_score(df_test_labels, df_predictions_test, average='micro')


# In[52]:


recall_score(df_test_labels, df_predictions_test, average='macro')


# In[53]:


roc_auc_score(df_test_labels, df_predictions_test)


# In[54]:


predictions_test.shape


# In[ ]:


fig, ax = plt.subplots(figsize=(6, 6))
roc_viz = ROCAUC(prediction_pipeline)
roc_viz.score(X_test, y_test)
roc_viz.poof()


# In[72]:


df_test = preparation_pipeline.transform(df_test)


# # Analyis of how much data the model uses to effecively learn more

# For each step, giving more data input to the model and see evolution of performance

# ## Dataset of size 1000

# In[86]:


importlib.reload(sys.modules['functions'])
from functions import *
importlib.reload(sys.modules['functions'])


# In[87]:


df_train = df_train_ori
df_test = df_test_ori

df_train_labels = df_train_labels_ori
df_test_labels = df_test_labels_ori


# In[88]:


DATASET_SIZE = 1000

df_train = df_train.loc[0:DATASET_SIZE, :]
df_train_labels = df_train_labels.loc[0:DATASET_SIZE, :]

df_test = df_test.loc[0:DATASET_SIZE, :]
df_test_labels = df_test_labels.loc[0:DATASET_SIZE, :]


# In[89]:


prediction_pipeline = Pipeline([
    ('doc2vec', Doc2Vec_Vectorizer(model_path=DOC2VEC_TRAINING_SAVE_FILE, feature_totransform='all_text')),
    #('features_selector', FeaturesSelector(features_toselect=['Tags'])),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(),)
    ])


# In[90]:


plot_learning_curves(prediction_pipeline, df_train, df_test, df_train_labels, df_test_labels, 200)


# ## Dataset of size 10000

# In[106]:


importlib.reload(sys.modules['functions'])
from functions import *
importlib.reload(sys.modules['functions'])


# In[107]:


df_train = df_train_ori
df_test = df_test_ori

df_train_labels = df_train_labels_ori
df_test_labels = df_test_labels_ori


# In[108]:


DATASET_SIZE = 10000
DATASET_TEST_SIZE = 1000

df_train = df_train.loc[0:DATASET_SIZE, :]
df_train_labels = df_train_labels.loc[0:DATASET_SIZE, :]

df_test = df_test.loc[0:DATASET_TEST_SIZE, :]
df_test_labels = df_test_labels.loc[0:DATASET_TEST_SIZE, :]


# In[109]:


prediction_pipeline = Pipeline([
    ('doc2vec', Doc2Vec_Vectorizer(model_path=DOC2VEC_TRAINING_SAVE_FILE, feature_totransform='all_text')),
    #('features_selector', FeaturesSelector(features_toselect=['Tags'])),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(),)
    ])


# In[110]:


plot_learning_curves(prediction_pipeline, df_train, df_test, df_train_labels, df_test_labels, 2000)


# ## Dataset of size 100000

# In[39]:


importlib.reload(sys.modules['functions'])
from functions import *
importlib.reload(sys.modules['functions'])


# In[40]:


df_train = df_train_ori
df_test = df_test_ori

df_train_labels = df_train_labels_ori
df_test_labels = df_test_labels_ori


# In[41]:


DATASET_SIZE = 100000
DATASET_TEST_SIZE = 10000

df_train = df_train.loc[0:DATASET_SIZE-1, :]
df_train_labels = df_train_labels.loc[0:DATASET_SIZE-1, :]

df_test = df_test.loc[0:DATASET_TEST_SIZE-1, :]
df_test_labels = df_test_labels.loc[0:DATASET_TEST_SIZE-1, :]


# In[42]:


prediction_pipeline = Pipeline([
    ('doc2vec', Doc2Vec_Vectorizer(model_path=DOC2VEC_TRAINING_SAVE_FILE, feature_totransform='all_text')),
    #('features_selector', FeaturesSelector(features_toselect=['Tags'])),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(),)
    ])


# In[43]:


plot_learning_curves(prediction_pipeline, df_train, df_test, df_train_labels, df_test_labels, 20000)


# # Implementation of a KNN classification algorithm on 90000 instances and check predictions

# In[44]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[45]:


df_train = df_train_ori
df_test = df_test_ori

df_train_labels = df_train_labels_ori
df_test_labels = df_test_labels_ori


# In[46]:


DATASET_SIZE = 90000
DATASET_TEST_SIZE = 10000

df_train = df_train.loc[0:DATASET_SIZE-1, :]
df_train_labels = df_train_labels.loc[0:DATASET_SIZE-1, :]

df_test = df_test.loc[0:DATASET_TEST_SIZE-1, :]
df_test_labels = df_test_labels.loc[0:DATASET_TEST_SIZE-1, :]


# In[47]:


df_train


# In[48]:


prediction_pipeline = Pipeline([
    ('doc2vec', Doc2Vec_Vectorizer(model_path=DOC2VEC_TRAINING_SAVE_FILE, feature_totransform='all_text')),
    #('features_selector', FeaturesSelector(features_toselect=['Tags'])),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(),)
    ])


# In[49]:


prediction_pipeline.fit(df_train, df_train_labels)


# In[50]:


get_ipython().run_cell_magic('time', '', 'predictions_train = prediction_pipeline.predict(df_train)')


# In[51]:


df_predictions_train = pd.DataFrame(predictions_train, columns=df_train_labels.columns)


# In[52]:


get_ipython().run_cell_magic('time', '', 'predictions_test = prediction_pipeline.predict(df_test)')


# In[53]:


df_predictions_test = pd.DataFrame(predictions_test, columns=df_test_labels.columns)


# In[54]:


df_predictions_train


# In[55]:


df_predictions_test


# In[56]:


df_train


# ## Performance measures

# In[57]:


precision_score(df_train_labels, df_predictions_train, average='micro')


# In[58]:


precision_score(df_train_labels, df_predictions_train, average='macro')


# In[59]:


# Shows exact matchs of all tags
accuracy_score(df_train_labels, df_predictions_train)


# In[60]:


recall_score(df_train_labels, df_predictions_train, average='micro')


# In[61]:


recall_score(df_train_labels, df_predictions_train, average='macro')


# In[67]:


roc_auc_score(df_train_labels, df_predictions_train)


# In[62]:


precision_score(df_test_labels, df_predictions_test, average='micro')


# In[63]:


precision_score(df_test_labels, df_predictions_test, average='macro')


# In[64]:


# Shows exact matchs of all tags
accuracy_score(df_test_labels, df_predictions_test)


# In[65]:


recall_score(df_test_labels, df_predictions_test, average='micro')


# In[66]:


recall_score(df_test_labels, df_predictions_test, average='macro')


# In[53]:


roc_auc_score(df_test_labels, df_predictions_test)


# In[54]:


predictions_test.shape


# ## Check how many instances have at least 1 tag predicted

# In[73]:


df_test_labels_sum = df_test_labels.sum(axis=1)


# In[76]:


df_test_labels_sum.shape


# In[77]:


df_test_labels_sum[df_test_labels_sum > 0]


# => 90% of instances have at least 1 predicted class to true (61% precision on them)

# ## Number of instances per class

# In[87]:


pd.set_option('display.max_rows', 400)
df_train_labels.sum().sort_values(ascending=False)


# In[97]:


df_test_labels.sum().sort_values(ascending=False)


# ## Score per class

# In[96]:


from sklearn.metrics import classification_report

print(classification_report(df_test_labels, df_predictions_test, target_names=df_test_labels.columns.tolist()))


# In[118]:


pd.DataFrame(classification_report(df_train_labels, df_predictions_train, target_names=df_train_labels.columns.tolist(), output_dict=True)).transpose()    .sort_values(by='precision', ascending=False).shape


# In[112]:


pd.DataFrame(classification_report(df_train_labels, df_predictions_train, target_names=df_train_labels.columns.tolist(), output_dict=True)).transpose()    .sort_values(by='precision', ascending=False)


# => La majorité des classes sont au dessus de 70% de précision

# In[117]:


pd.DataFrame(classification_report(df_test_labels, df_predictions_test, target_names=df_test_labels.columns.tolist(), output_dict=True)).transpose()    .sort_values(by='precision', ascending=False)


# => Overfit :  
# Tags_python 	0.535714 	0.300000 	0.384615 	1200.0

# ## Number of correct labels predicted per sample

# In[126]:


df_train_nb_correct_labels_predicted = (df_train_labels * df_predictions_train).sum(axis=1)


# In[128]:


df_train_nb_correct_labels_predicted[df_train_nb_correct_labels_predicted > 0]


# In[122]:


df_test_nb_correct_labels_predicted = (df_test_labels * df_predictions_test).sum(axis=1)


# In[124]:


df_test_nb_correct_labels_predicted[df_test_nb_correct_labels_predicted > 0]


# In[125]:


df_test_nb_correct_labels_predicted


# In[139]:


df_train_nb_correct_labels_predicted[df_train_nb_correct_labels_predicted == 0]


# In[136]:


index_train_noclue = df_train_nb_correct_labels_predicted[df_train_nb_correct_labels_predicted == 0].index


# In[137]:


index_train_noclue


# For instance 0, model is not totally out of the truth :  tag pycharm predicted :

# In[74]:


df_train_labels.loc[0, :]['Tags_python']


# In[82]:


df_predictions_train.loc[0,:][df_predictions_train.loc[0,:] > 0]


# In[138]:


df_train.loc[index_train_noclue, :]


# In[141]:


doc_index = 0
col_names_tags_value_1 = [col for col in df_train_labels[df_predictions_train.index.isin([doc_index])]                          if (df_predictions_train[df_train_labels.index.isin([doc_index])][col] == 1).any()]


# In[150]:


df_train_labels[df_train_labels == 1]


# In[155]:


(df_train_labels == 1)


# In[151]:


df_train_labels


# In[148]:


# Labels that model had no clue to predict (all labels that were missed, plus the model missed all of the other labels for the instance) => too slow
col_names_tags_value_1_labels = [[col for col in df_train_labels[df_predictions_train.index.isin([doc_index])]                        if (df_train_labels[df_train_labels.index.isin([doc_index])][col] == 1).any()]  for doc_index in index_train_noclue]


# In[147]:


col_names_tags_value_1_labels


# In[143]:


'''
#Too slow
col_names_tags_value_1_allsamples = [[col for col in df_train_labels[df_predictions_train.index.isin([doc_index])]\
                          if (df_predictions_train[df_train_labels.index.isin([doc_index])][col] == 1).any()] for doc_index in index_train_noclue]

'''


# In[142]:


col_names_tags_value_1


# In[ ]:


df_predictions_test


# In[116]:


df_predictions_train['Tags_electron'].sum()


# In[104]:


print(classification_report(df_train_labels, df_predictions_train, target_names=df_train_labels.columns.tolist()))


# In[ ]:





# ## Precision / Recall curve

# In[ ]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

disp = plot_precision_recall_curve(prediction_pipeline, df_test, df_test_labels)


# # Implementation of a Decision Tree classification algorithm

# In[39]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[40]:


df_train = df_train_ori
df_test = df_test_ori

df_train_labels = df_train_labels_ori
df_test_labels = df_test_labels_ori


# In[41]:


DATASET_SIZE = 90000
DATASET_TEST_SIZE = 10000

df_train = df_train.loc[0:DATASET_SIZE-1, :]
df_train_labels = df_train_labels.loc[0:DATASET_SIZE-1, :]

df_test = df_test.loc[0:DATASET_TEST_SIZE-1, :]
df_test_labels = df_test_labels.loc[0:DATASET_TEST_SIZE-1, :]


# In[42]:


df_train


# In[43]:


prediction_pipeline = Pipeline([
    ('doc2vec', Doc2Vec_Vectorizer(model_path=DOC2VEC_TRAINING_SAVE_FILE, feature_totransform='all_text')),
    #('features_selector', FeaturesSelector(features_toselect=['Tags'])),
    ('scaler', StandardScaler()),
    ('knn', DecisionTreeClassifier(),)
    ])


# In[44]:


get_ipython().run_cell_magic('time', '', 'prediction_pipeline.fit(df_train, df_train_labels)')


# In[45]:


get_ipython().run_cell_magic('time', '', 'predictions_train = prediction_pipeline.predict(df_train)')


# In[46]:


get_ipython().run_cell_magic('time', '', 'df_predictions_train = pd.DataFrame(predictions_train, columns=df_train_labels.columns)')


# In[47]:


get_ipython().run_cell_magic('time', '', 'predictions_test = prediction_pipeline.predict(df_test)')


# In[48]:


df_predictions_test = pd.DataFrame(predictions_test, columns=df_test_labels.columns)


# In[49]:


df_predictions_train


# In[50]:


df_predictions_test


# In[51]:


df_train


# ## Performance measures

# In[52]:


precision_score(df_train_labels, df_predictions_train, average='micro')


# In[53]:


precision_score(df_train_labels, df_predictions_train, average='macro')


# In[54]:


# Shows exact matchs of all tags
accuracy_score(df_train_labels, df_predictions_train)


# In[55]:


recall_score(df_train_labels, df_predictions_train, average='micro')


# In[56]:


recall_score(df_train_labels, df_predictions_train, average='macro')


# In[ ]:


roc_auc_score(df_train_labels, df_predictions_train)


# In[57]:


precision_score(df_test_labels, df_predictions_test, average='micro')


# In[58]:


precision_score(df_test_labels, df_predictions_test, average='macro')


# In[ ]:


# Shows exact matchs of all tags
accuracy_score(df_test_labels, df_predictions_test)


# In[59]:


recall_score(df_test_labels, df_predictions_test, average='micro')


# In[60]:


recall_score(df_test_labels, df_predictions_test, average='macro')


# In[ ]:


roc_auc_score(df_test_labels, df_predictions_test)


# In[61]:


predictions_test.shape


# In[ ]:





# In[ ]:





# # Implementation of a MultinomialNB with partial fit

# In[50]:


importlib.reload(sys.modules['functions'])
from functions import *


# In[51]:


df_train = df_train_ori
df_test = df_test_ori

df_train_labels = df_train_labels_ori
df_test_labels = df_test_labels_ori


# In[86]:


DATASET_SIZE = 90000
DATASET_TEST_SIZE = 10000
STEP_SIZE = 10000 # Mini batch step size

df_train = df_train.loc[0:DATASET_SIZE-1, :]
df_train_labels = df_train_labels.loc[0:DATASET_SIZE-1, :]

df_test = df_test.loc[0:DATASET_TEST_SIZE-1, :]
df_test_labels = df_test_labels.loc[0:DATASET_TEST_SIZE-1, :]


# In[52]:


DATASET_SIZE = 1000
DATASET_TEST_SIZE = 100
STEP_SIZE = 200 # Mini batch step size

df_train = df_train.loc[0:DATASET_SIZE-1, :]
df_train_labels = df_train_labels.loc[0:DATASET_SIZE-1, :]

df_test = df_test.loc[0:DATASET_TEST_SIZE-1, :]
df_test_labels = df_test_labels.loc[0:DATASET_TEST_SIZE-1, :]


# In[53]:


df_train


# In[68]:


preparation_pipeline = Pipeline([
    ('doc2vec', Doc2Vec_Vectorizer(model_path=DOC2VEC_TRAINING_SAVE_FILE, feature_totransform='all_text')),
    #('features_selector', FeaturesSelector(features_toselect=['Tags'])),
    #('scaler', StandardScaler()),
    ('scaler', MinMaxScaler()),  # MinMaxScaler to have only positive values required by Naive Bayes
    ])

prediction_model = MultiOutputClassifier(MultinomialNB())


# In[55]:


get_ipython().run_cell_magic('time', '', 'df_train = preparation_pipeline.fit_transform(df_train, df_train_labels)')


# In[56]:


get_ipython().run_cell_magic('time', '', 'df_test = preparation_pipeline.transform(df_test)')


# In[81]:


minibatch_indexes = minibatch_generate_indexes(df_train, STEP_SIZE)


# In[82]:


np.unique(df_train_labels.loc[left_index:right_index])


# In[83]:


get_ipython().run_cell_magic('time', '', 'nb_iter = int(DATASET_SIZE / STEP_SIZE)\nprogbar = tqdm(range(nb_iter))\n\ntrain_errors, val_errors = [], []\n\nfor (left_index, right_index) in minibatch_indexes:\n    print(\'Partial fit\')\n    # right_index+1 because df_train is in numpy format, right bound is not included (but in pandas, right bound is included)\n    \n    # Below does not work : 3rd parameters (classes) can\'t be strings :(\n    #prediction_model.partial_fit(df_train[left_index:right_index+1], df_train_labels.loc[left_index:right_index], df_train_labels.columns.tolist())\n    \n    prediction_model.partial_fit(df_train[left_index:right_index+1],\\\n                                 df_train_labels.loc[left_index:right_index],\\\n                                 np.unique(df_train_labels.loc[left_index:right_index]))\n    \n    print(\'Intermediate prediction\')\n    predictions_train = prediction_model.predict(df_train[left_index:right_index+1])\n    predictions_test = prediction_model.predict(df_test)\n    \n    train_errors.append(precision_score(df_train_labels.loc[left_index:right_index], predictions_train, average=\'micro\'))\n    val_errors.append(precision_score(df_test_labels.loc[left_index:right_index], predictions_test, average=\'micro\'))   \n    \n    print(\'Train errors : \' + str(train_errors))\n    print(\'Test errors : \' + str(val_errors))\n    \n    progbar.update(1)\n    \nplt.plot(train_errors, "r-+", linewidth=2, label="train")\nplt.plot(val_errors, "b-", linewidth=3, label="test")\nplt.legend(loc="upper right", fontsize=14)   # not shown in the book\nplt.xlabel("Training set iterations", fontsize=14) # not shown\n\nplt.ylabel("precision_micro", fontsize=14)      ')


# In[96]:


np.unique(df_train_labels.loc[0:5])


# In[98]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=df_train_labels.columns.tolist())


# In[102]:


df_train_ori


# In[100]:


df_train_labels


# In[99]:


df_train_labels_binarized = mlb.fit_transform(df_train_labels)


# In[97]:


prediction_model.partial_fit(df_train[0:5+1],                             df_train_labels.loc[0:5],                             np.unique(df_train_labels.loc[0:5]))


# In[101]:


df_train


# In[61]:


df_train_labels.loc[left_index:right_index].shape


# ### MultinomialNB does not seem to support multi label :(

# In[63]:


predictions_train


# In[66]:


pd.DataFrame(prediction_model.predict_proba(df_train[left_index:right_index+1])).shape


# In[59]:


precision_score(df_train_labels.loc[left_index:right_index], predictions_train, average='micro')


# In[ ]:


right_index


# In[ ]:


df_train.shape[0]


# In[ ]:


len(df_train[0:5])


# In[ ]:


df_train[left_index:right_index].shape[0]


# In[ ]:


df_train_labels.loc[left_index:right_index].shape[0]


# In[ ]:


df_train_labels.loc[left_index:right_index]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'predictions_train = prediction_model.predict(df_train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_predictions_train = pd.DataFrame(predictions_train, columns=df_train_labels.columns)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'predictions_test = prediction_model.predict(df_test)')


# In[ ]:


df_predictions_test = pd.DataFrame(predictions_test, columns=df_test_labels.columns)


# In[ ]:


df_predictions_train


# In[ ]:


df_predictions_test


# In[ ]:


df_train


# ## Performance measures

# In[ ]:


precision_score(df_train_labels, df_predictions_train, average='micro')


# In[ ]:


precision_score(df_train_labels, df_predictions_train, average='macro')


# In[ ]:


# Shows exact matchs of all tags
accuracy_score(df_train_labels, df_predictions_train)


# In[ ]:


recall_score(df_train_labels, df_predictions_train, average='micro')


# In[ ]:


recall_score(df_train_labels, df_predictions_train, average='macro')


# In[ ]:


roc_auc_score(df_train_labels, df_predictions_train)


# In[ ]:


precision_score(df_test_labels, df_predictions_test, average='micro')


# In[ ]:


precision_score(df_test_labels, df_predictions_test, average='macro')


# In[ ]:


# Shows exact matchs of all tags
accuracy_score(df_test_labels, df_predictions_test)


# In[ ]:


recall_score(df_test_labels, df_predictions_test, average='micro')


# In[ ]:


recall_score(df_test_labels, df_predictions_test, average='macro')


# In[ ]:


roc_auc_score(df_test_labels, df_predictions_test)


# In[ ]:


predictions_test.shape


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
