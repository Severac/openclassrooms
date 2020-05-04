#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ5 : Online Retail dataset :  data exploration notebook 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from functions import *


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

import os
import zipfile
import urllib

import matplotlib.pyplot as plt

import numpy as np

import qgrid

import glob

from pandas.plotting import scatter_matrix

DOWNLOAD_ROOT = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/"
DATA_FILENAME = "Online%20Retail.xlsx"
DATA_PATH = os.path.join("datasets", "onlineretail")

DATA_PATH_OUT = os.path.join(DATA_PATH, "out")
DATA_PATH_FILE_OUTPUT = os.path.join(DATA_PATH_OUT, "OnlineRetail_transformed.csv")

DATA_URL = DOWNLOAD_ROOT + DATA_FILENAME
ARCHIVE_PATH_FILE = os.path.join(DATA_PATH, DATA_FILENAME)


DOWNLOAD_DATA = True  # A la première exécution du notebook, ou pour rafraîchir les données, mettre cette variable à True

plt.rcParams["figure.figsize"] = [16,9] # Taille par défaut des figures de matplotlib

import seaborn as sns
sns.set()

#import common_functions


### For progress bar :
from tqdm import tqdm_notebook as tqdm
                                        


# # Download and decompression of data

# In[3]:


#PROXY_DEF = 'BNP'
PROXY_DEF = None

def fetch_dataset(data_url=DATA_URL, data_path=DATA_PATH):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    archive_path = ARCHIVE_PATH_FILE
    
    if (PROXY_DEF == 'BNP'):
        #create the object, assign it to a variable
        proxy = urllib.request.ProxyHandler({'https': 'https://user:pass@ncproxy:8080'})
        # construct a new opener using your proxy settings
        opener = urllib.request.build_opener(proxy)
        # install the openen on the module-level
        urllib.request.install_opener(opener)    
    
    urllib.request.urlretrieve(data_url, archive_path)
    #data_archive = zipfile.ZipFile(archive_path)
    #data_archive.extractall(path=data_path)
    #data_archive.close()


# In[4]:


if (DOWNLOAD_DATA == True):
    fetch_dataset()


# # Data load

# In[5]:


DATA_PATH_FILE = os.path.join(DATA_PATH, "*.xlsx")
ALL_FILES_LIST = glob.glob(DATA_PATH_FILE)


# ## Import of Excel file

# In[6]:


import pandas as pd

pd.set_option('display.max_columns', None)

def load_data(data_path=DATA_PATH):
    file_path = DATA_PATH_FILE
    df_list = []
    
    for f in ALL_FILES_LIST:
        print(f'Loading file {f}')

        df_list.append(pd.read_excel(f, encoding='utf-8', converters={'InvoiceNo': str, 'StockCode':str, 'Description': str,                                        'CustomerID':str, 'Country': str})
        )

        
    return pd.concat(df_list)


# In[7]:


df = load_data()


# ## Display some data and basic information

# In[8]:


df.head(10)


# In[9]:


df.info()


# In[10]:


df.describe()


# # Check for duplicates, and drop them

# In[11]:


df[df.duplicated()]


# In[12]:


df.drop_duplicates(inplace=True)


# In[13]:


df.info()


# In[14]:


df.shape


# # Check quality of data (% complete)

# In[15]:


display_percent_complete(df)


# # Analysis of qualitative values

# ## Display of different possible values for qualitative features

# In[16]:


for column_name in df.select_dtypes(include=['object']).columns:
    print_column_information(df, column_name)


# ### Trim all text values and check possible values again

# In[17]:


df_obj = df.select_dtypes(['object'])

df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())


# In[18]:


for column_name in df.select_dtypes(include=['object']).columns:
    print_column_information(df, column_name)


# => Description now has 4211 distinct values instead of 4223

# ## Customer ID

# In[19]:


df.shape


# In[20]:


df[df['CustomerID'].isnull()]


# ### Drop null customer values
# Our goal is to segment customers, so orders without customer values are useless

# In[21]:


df.drop(index=df[df['CustomerID'].isnull()].index, axis=0, inplace=True)


# In[158]:


df.reset_index(drop=True, inplace=True)


# In[22]:


df.shape


# ## Check quality of data again (% complete)

# In[23]:


display_percent_complete(df)


# => Now every value is set

# ## InvoiceNo : analysis of cancellations

# ### Display cancellations (InvoiceNo starting with C according to dataset description)

# In[35]:


df[df['InvoiceNo'].str.startswith('C')]


# In[64]:


print('{:.2f}% of orders are cancellations'.format((len(df[df['InvoiceNo'].str.startswith('C')])/df.shape[0])*100))


# ### Check all orders from a client that has cancelled 1 order

# In[55]:


df[df['CustomerID'] == '17548']


# => We see that cancelled products appear only on the line with InvoiceNo starting with C  
# => We can remove cancellations ?

# ### Mean number of orders for clients :

# In[101]:


df['CustomerID'].value_counts().mean()


# ### Mean number of orders for clients that have cancelled at least 1 order :

# In[97]:


df[df['CustomerID'].isin(df[df['InvoiceNo'].str.startswith('C')]['CustomerID'].unique())]['CustomerID'].value_counts().mean()


# => We'll keep this information of cancellations for the model

# ## Invoice date (min and max values)

# In[24]:


print('Minimum Invoice Date : ' + str(df['InvoiceDate'].min()))
print('Maximum Invoice Date : ' + str(df['InvoiceDate'].max()))


# ## Comparison of StockCode and Description

# In[25]:


df[['StockCode', 'Description']].sort_values(by='StockCode')


# => Description is the text corresponding to StockCode

# In[26]:


print('Number of unique Description : ' + str(len(df['Description'].unique())))
print('Number of unique StockCode : ' + str(len(df['StockCode'].unique())))


# => There are more descriptions than stock codes. Are there inconsistencies with some description texts ?

# In[116]:


progbar = tqdm(range(len(df['StockCode'].unique())))

stockcodes_defaults = []

for stockcode_id in df['StockCode'].unique():    
    # If number of unique description values is different from 1, we have some anomaly in description
    if ((len(df[df['StockCode'] == stockcode_id]['Description'].unique())) != 1):
        stockcodes_defaults.append(stockcode_id)   
    
    progbar.update(1)


# In[128]:


print('=> ' + str(len(stockcodes_defaults)) + ' products do not always have the same description text for each order')


# Let's explore that : we have some differences due to coma or added words/letters

# In[123]:


qgrid_show(df[df['StockCode'].isin(stockcodes_defaults)].sort_values(by='StockCode'))


# In[148]:


# Print description that has most occurences for a stock code :
df[df['StockCode'] == '21232']['Description'].value_counts().sort_values(ascending=False).index[0]


# In[150]:


progbar = tqdm(range(len(df['StockCode'].unique())))

ref_descriptions = {}

# For each stock code : assign most represented description value in the dataset
for stockcode_id in df['StockCode'].unique():    
    ref_descriptions[stockcode_id] = df[df['StockCode'] == stockcode_id]['Description'].value_counts().sort_values(ascending=False).index[0]
    
    progbar.update(1)


# In[161]:


df['DescriptionNormalized'] = df['StockCode'].apply(lambda val : ref_descriptions[val] )


# In[163]:


qgrid_show(df[df['StockCode'].isin(stockcodes_defaults)].sort_values(by='StockCode'))


# In[112]:


len(df[df['StockCode'] == '22733']['Description'].unique())


# In[66]:


df[df['StockCode'].isna()]


# In[63]:


df[['StockCode']].sort_values(by='StockCode')


# In[67]:


df[['StockCode', 'Description']].sort_values(by='StockCode')


# In[18]:


qgrid_show(df)

