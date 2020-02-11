#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ3 : IMDB dataset :  data cleaning notebook 

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import zipfile
import urllib

import matplotlib.pyplot as plt

import numpy as np

DOWNLOAD_ROOT = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Moteur+de+recommandation+de+films/"
DATA_PATH = os.path.join("/home/francois/coding/OC/PJ3/datasets", "imdb")
DATA_PATH_FILE = os.path.join(DATA_PATH, "movie_metadata.csv")
DATA_URL = DOWNLOAD_ROOT + "imdb-5000-movie-dataset.zip"

DATA_PATH_FILE_OUTPUT = os.path.join(DATA_PATH, "movie_metadata_transformed.csv")

DOWNLOAD_DATA = True  # A la première exécution du notebook, ou pour rafraîchir les données, mettre cette variable à True


import seaborn as sns
sns.set()


# # Téléchargement et décompression des données

# In[14]:


#PROXY_DEF = 'BNP'
PROXY_DEF = None

def fetch_dataset(data_url=DATA_URL, data_path=DATA_PATH):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    archive_path = os.path.join(data_path, "imdb-5000-movie-dataset.zip")
    
    if (PROXY_DEF == 'BNP'):
        #create the object, assign it to a variable
        proxy = urllib.request.ProxyHandler({'https': 'https:/login:password@ncproxy:8080'})
        # construct a new opener using your proxy settings
        opener = urllib.request.build_opener(proxy)
        # install the openen on the module-level
        urllib.request.install_opener(opener)    
    
    urllib.request.urlretrieve(data_url, archive_path)
    data_archive = zipfile.ZipFile(archive_path)
    data_archive.extractall(path=data_path)
    data_archive.close()


# In[15]:


if (DOWNLOAD_DATA == True):
    fetch_dataset()


# In[ ]:




