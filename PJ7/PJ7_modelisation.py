#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ7 : implement automatic image indexing

# In[2]:


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
from sklearn.linear_model import Perceptron
from sklearn import tree

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

#from yellowbrick.classifier import ROCAUC
from sklearn.metrics import roc_auc_score

import nltk
import codecs

from sklearn.decomposition import LatentDirichletAllocation

#from nltk.corpus.reader.api import CorpusReader
#from nltk.corpus.reader.api import CategorizedCorpusReader

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

#import pandas_profiling

from bs4 import BeautifulSoup

DATA_PATH = os.path.join("datasets", "stanforddogs")
DATA_PATH = os.path.join(DATA_PATH, "Images")

#DATA_PATH_FILE_INPUT = os.path.join(DATA_PATH, "QueryResults_20190101-20200620.csv")
#DATA_PATH_FILE_INPUT = os.path.join(DATA_PATH, "QueryResults 20200301-20200620_1.csv")

DATA_PATH_FILE = os.path.join(DATA_PATH, "*.csv")
ALL_FILES_LIST = glob.glob(DATA_PATH_FILE)

ALL_FEATURES = []

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

# Set this to load (or train again / save) KNN model to disk
SAVE_KNN_MODEL = False
LOAD_KNN_MODEL = True

KNN_FILE_MODEL_PREFIX = 'knn_model'


# Set this to load (or train again / save) second KNN model to disk  
#  (in "Implementation of KNN classification algorithm on 9000 instances all of which have at least 1 label" part)
#  (and also in "Same as 1/ but with predict proba instead of predict" part)
SAVE_KNN_MODEL2 = False
LOAD_KNN_MODEL2 = True

# Set this to load (or predict again / save) probabilites prediction on model 2 to disk  (in "Same as 1/ but with predict proba instead of predict" part)
SAVE_KNN_MODEL2_PROBA = False
LOAD_KNN_MODEL2_PROBA = True


KNN_FILE_MODEL_PREFIX2 = 'knn_model2'

SAVE_BESTGRIDSEARCH_MODEL = False
LOAD_BESTGRIDSEARCH_MODEL = True
BESTGRIDSEARCH_FILE_MODEL_PREFIX = 'bestgridsearch_model_'

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
API_MODEL_PICKLE_FILE = 'API_model_PJ7.pickle'


LEARNING_CURVE_STEP_SIZE = 100


# # Image settings

# In[3]:


from PIL import Image
from io import BytesIO


# In[4]:


import cv2

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()


# # Exploration of 1 image from OC course

# In[5]:


img = Image.open("simba.png") 


# In[6]:


np_img = np.array(img)


# In[7]:


# Pour le normaliser : argument density=True dans plt.hist
# Pour avoir l'histogramme cumulé : argument cumulative=True

n, bins, patches = plt.hist(np_img.flatten(), bins=range(256))
plt.show()


# In[8]:


# Pour le normaliser : argument density=True dans plt.hist
# Pour avoir l'histogramme cumulé : argument cumulative=True

n, bins, patches = plt.hist(np_img.flatten(), bins=range(256), edgecolor='blue')
plt.show()


# In[9]:


# Pour le normaliser : argument density=True dans plt.hist
# Pour avoir l'histogramme cumulé : argument cumulative=True

n, bins, patches = plt.hist(np_img.flatten(), bins=range(256), histtype='stepfilled')
plt.show()


# # Exploration of 1 image

# In[10]:


PATH_TESTIMAGE = DATA_PATH + "/n02111889-Samoyed/" + "n02111889_1363.jpg"
img = Image.open(PATH_TESTIMAGE) 


# In[11]:


'n02108422-bull_mastiff'.split('-')[1]


# In[12]:


display(img)


# In[13]:


img.size


# In[14]:


img.mode


# In[15]:


img.getpixel((20, 100))


# In[16]:


np_image = np.array(img)


# In[17]:


np_image.shape


# In[18]:


np.array([1,2,3])


# In[19]:


np_img = np.array(img)


# In[20]:


np_img[:, :, 0]


# In[21]:


# Pour le normaliser : argument density=True dans plt.hist
# Pour avoir l'histogramme cumulé : argument cumulative=True

n, bins, patches = plt.hist(np_img.flatten(), bins=range(256))
plt.show()


# In[22]:


# Pour le normaliser : argument density=True dans plt.hist
# Pour avoir l'histogramme cumulé : argument cumulative=True

n, bins, patches = plt.hist(np_img[:, :, 0].flatten(), bins=range(256))
plt.show()


# In[23]:


# Pour le normaliser : argument density=True dans plt.hist
# Pour avoir l'histogramme cumulé : argument cumulative=True

n, bins, patches = plt.hist(np_img[:, :, 1].flatten(), bins=range(256))
plt.show()


# In[24]:


# Pour le normaliser : argument density=True dans plt.hist
# Pour avoir l'histogramme cumulé : argument cumulative=True

n, bins, patches = plt.hist(np_img[:, :, 2].flatten(), bins=range(256))
plt.show()


# # Image preprocessing tests

# In[25]:


imgcv = cv2.imread(PATH_TESTIMAGE)


# In[26]:


imgcv_gray = cv2.cvtColor(imgcv,cv2.COLOR_BGR2GRAY) # Gray scaling


# In[27]:


imgcv_gray.shape


# In[28]:


Image.fromarray(imgcv_gray)


# In[29]:


kp = sift.detect(imgcv_gray,None)

imgcv_keypoints = cv2.drawKeypoints(imgcv_gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)

cv2.imwrite('sift_keypoints.jpg',imgcv_keypoints)


# In[30]:


Image.fromarray(imgcv_keypoints)


# In[31]:


kp, des = sift.detectAndCompute(imgcv_gray,None)


# In[32]:


len(kp)


# In[33]:


des.shape


# In[34]:


flann_params = dict(algorithm = 1, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {}) 
bow_extract = cv2.BOWImgDescriptorExtractor( sift , matcher )
#bow_extract.setVocabulary( vocab ) # the 64x20 dictionary, you made before # <= selon la doc:  Each row of the vocabulary is a visual word (cluster center). 


# In[35]:


#bowsig = bow_extract.compute(imgcv_gray, kp)


# # Load images and labels

# In[90]:


filename_images = []
np_images = []
labels = []

deslist = []

NB_DOGS_PER_RACE = 5
i = 0

np_des = np.empty((0, 1+128), np.uint8)  # We add 1 column to store the image number


# In[91]:


np_des


# In[92]:


np_toy = np.full((5, 10), 1, dtype=np.uint8)


# In[93]:


np_toy.nbytes


# In[94]:


get_ipython().run_cell_magic('time', '', '\ncnt_files = sum([len(files) for r, d, files in os.walk(DATA_PATH)])\nprogbar = tqdm(range(cnt_files))\n\npic_number = 0\nfor root, dirs, files in os.walk(DATA_PATH):\n    path = root.split(os.sep)\n    \n    i = 0    \n    for file in files:\n        if (i > NB_DOGS_PER_RACE - 1):\n            break\n        \n        # Append filename to global list\n        filename_images.append(os.path.join(root, file))\n        \n        img = cv2.imread(os.path.join(root, file))\n        \n        kp, des = sift.detectAndCompute(img, None)\n        \n        np_picnum = np.full((des.shape[0], 1), pic_number, dtype=np.uint16) # Add 1 column to store picture number\n        des = np.hstack((np_picnum, des.astype(np.uint8)))\n        \n        #deslist.append(des)\n        #np_des = np.append(np_des, des, axis=0)\n        np_des = np.vstack((np_des, des)) # Equivalent of line above\n        \n        progbar.update(1)\n        \n        i += 1\n        pic_number += 1')


# In[113]:


np_des.dtype


# In[97]:


np_des.shape


# In[114]:


df_des = pd.DataFrame(np_des)


# In[86]:


df_des.loc[:, 1].max()


# In[89]:


df_des.max()


# In[98]:


np_des.nbytes


# In[ ]:


np_des = deslist[0]

for descriptor in deslist:
    np_des = np.vstack((np_des, descriptor))  


# In[16]:


sys.getsizeof(np_images)


# In[22]:


np_images[100].nbytes


# In[23]:


np_images[100].dtype


# In[24]:


np_images[100].shape


# In[32]:


np_images[50]


# In[27]:


len(filename_images)


# In[16]:


labels[50]


# ## With Pandas

# In[144]:


filename_images = []
np_images = []
labels = []

deslist = []

NB_DOGS_PER_RACE = 10
i = 0

df_des = pd.DataFrame(np.empty((0, 1+128), np.uint8), columns=['picnum'] +  [colname for colname in range(0,128)])  # We add 1 column to store the image number


# In[145]:


df_des


# In[146]:


get_ipython().run_cell_magic('time', '', "\ncnt_files = sum([len(files) for r, d, files in os.walk(DATA_PATH)])\nprogbar = tqdm(range(cnt_files))\n\npic_number = 0\nfor root, dirs, files in os.walk(DATA_PATH):\n    path = root.split(os.sep)\n    \n    i = 0    \n    for file in files:\n        if (i > NB_DOGS_PER_RACE - 1):\n            break\n        \n        # Append filename to global list\n        filename_images.append(os.path.join(root, file))\n        \n        img = cv2.imread(os.path.join(root, file))\n        \n        kp, des = sift.detectAndCompute(img, None)\n        \n        df_picnum = pd.DataFrame(np.full((des.shape[0], 1), pic_number, dtype=np.uint16), columns=['picnum']) # Add 1 column to store picture number\n        #print(des.shape)\n        #print(df_picnum.shape)\n        \n        df_des_1pic = pd.concat([df_picnum, pd.DataFrame(des.astype(np.uint8))], axis=1)\n        #print(df_des_1pic.shape)\n        #print(df_des_1pic.columns)\n        \n        #print('df_des.shape avant concat ')\n        #print(df_des.shape)\n        \n        df_des = pd.concat([df_des, df_des_1pic], axis=0)\n\n        #print('df_des.shape apres concat ')\n        #print(df_des.shape)\n        #print('!')\n        \n        progbar.update(1)\n        \n        i += 1\n        pic_number += 1")


# In[147]:


df_des


# In[148]:


df_des.memory_usage()


# In[149]:


df_des.info()


# In[154]:


clusterer = Clusterer(n_clusters=200)


# In[ ]:


get_ipython().run_line_magic('time', '')

clusterer.fit(df_des)


# In[112]:


len(filename_images)


# # Annex (old code)

# In[6]:


img = Image.open(DATA_PATH + "/n02111889-Samoyed/" + "n02111889_1363.jpg") 

temp = BytesIO()
img.save(temp, format="png")
display(Image.open(temp))


# In[26]:


# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(DATA_PATH):
    path = root.split(os.sep)
    #print((len(path) - 1) * 'p---', os.path.basename(root))
    
    for file in files:
        print(os.path.join(root, file))
        print(len(path) * '---', file)
        print(len(path) * '---', os.path.basename(root).split('-')[1])


# In[ ]:


for root, dirs, files in os.walk(DATA_PATH):
    path = root.split(os.sep)
    
    for file in files:
        # Append filename to global list
        filename_images.append(os.path.join(root, file))
        
        # Append image numpy array to global list
        im = Image.open(os.path.join(root, file))
        np_images.append(np.array(im))  # <= Takes too much memory (compared to jpeg disk space usage) : for later, either fix memory problem, or stream files from disk
        im.close()
        
        # Append image label to global list (we use the part of directory name after the '-'character as label)
        labels.append(os.path.basename(root).split('-')[1])

        # Limitation for now, because of memory
        '''
        i += 1
        
        if (i > 10):
            break
        '''
        

