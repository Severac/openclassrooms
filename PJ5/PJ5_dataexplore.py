#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ5 : Online Retail dataset :  data exploration notebook 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from functions import *
from display_factorial import *

import datetime as dt


# In[2]:


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


DOWNLOAD_DATA = False  # A la première exécution du notebook, ou pour rafraîchir les données, mettre cette variable à True

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


# In[22]:


df.reset_index(drop=True, inplace=True)


# In[23]:


df.shape


# ## Check quality of data again (% complete)

# In[24]:


display_percent_complete(df)


# => Now every value is set

# ## InvoiceNo : analysis of cancellations

# ### Display cancellations (InvoiceNo starting with C according to dataset description)

# In[26]:


df[df['InvoiceNo'].str.startswith('C')]


# In[27]:


print('{:.2f}% of orders are cancellations'.format((len(df[df['InvoiceNo'].str.startswith('C')])/df.shape[0])*100))


# ### Check all orders from a client that has cancelled 1 order

# In[28]:


df[df['CustomerID'] == '17548']


# => We see that cancelled products appear only on the line with InvoiceNo starting with C  
# => Can we remove cancellations ?

# ### Mean number of orders and total price for clients :

# In[29]:


df['CustomerID'].value_counts().mean()


# In[30]:


df['TotalPrice'] = df['Quantity'] * df['UnitPrice']


# In[31]:


df[['CustomerID', 'TotalPrice']].groupby('CustomerID').sum().mean()


# In[32]:


# Mean price must be calculated without taking cancellations into account :
df[df['InvoiceNo'].str.startswith('C') == False][['CustomerID', 'TotalPrice']].groupby('CustomerID').sum().mean()


# ### Mean number of orders and total price for clients that have cancelled at least 1 order :

# In[33]:


df[df['CustomerID'].isin(df[df['InvoiceNo'].str.startswith('C')]['CustomerID'].unique())]['CustomerID'].value_counts().mean()


# In[34]:


# Take cancellations into account
df[(df['CustomerID'].isin(df[df['InvoiceNo'].str.startswith('C')]['CustomerID'].unique()))    & (df['InvoiceNo'].str.startswith('C') == False)
  ]['CustomerID'].value_counts().mean()


# In[35]:


df[df['CustomerID'].isin(df[df['InvoiceNo'].str.startswith('C')]['CustomerID'].unique())][['CustomerID', 'TotalPrice']].groupby('CustomerID').sum().mean()


# In[36]:


# Take cancellations into account
df[(df['CustomerID'].isin(df[df['InvoiceNo'].str.startswith('C')]['CustomerID'].unique()))    & (df['InvoiceNo'].str.startswith('C') == False)
  ][['CustomerID', 'TotalPrice']].groupby('CustomerID').sum().mean()


# => Clients that have cancelled at least 1 order earn 2x more value  (4243 / 2048  = 2.07)  
# => We'll keep this information of cancellations for the model

# In[37]:


df_nocancel = df[df['InvoiceNo'].str.startswith('C') == False]


# ### Number of clients that got at least one discount

# In[47]:


df[df['StockCode'] == 'D']['CustomerID'].unique()


# ### Total earned values for clients that got at least one discount

# In[52]:


df[df['CustomerID'].isin(df[df['StockCode'] == 'D']['CustomerID'].unique())]['TotalPrice'].sum()


# In[55]:


df[df['CustomerID'].isin(df[df['StockCode'] == 'D']['CustomerID'].unique()) & (df['StockCode'] == 'D')]['TotalPrice'].sum()


# => 24 clients that got at least 1 discount earn 1M £ !  'StockCode' == 'D' (discount) feature will be useful

# ### Mean number of orders and total price for clients that have got a discount

# In[38]:


# Take cancellations into account
df[(df['CustomerID'].isin(df[df['StockCode'] == 'D']['CustomerID'].unique()))    & (df['InvoiceNo'].str.startswith('C') == False)
  ][['CustomerID', 'TotalPrice']].groupby('CustomerID').sum().mean()


# In[40]:


df[(df['CustomerID'].isin(df[df['StockCode'] == 'D']['CustomerID'].unique()))    & (df['StockCode'] == 'D')
  ][['CustomerID', 'TotalPrice']].groupby('CustomerID').sum().mean()


# ### Drop false cancellations

# In[25]:


# Drop all alphanumeric product codes except 'D' that are Discounts which is a useful feature
df.drop(index=df[df['StockCode'].isin(['POST', 'M', 'PADS', 'DOT', 'CRUK'])].index, axis=0, inplace=True)


# ## Invoice date (min and max values)

# In[37]:


print('Minimum Invoice Date : ' + str(df['InvoiceDate'].min()))
print('Maximum Invoice Date : ' + str(df['InvoiceDate'].max()))


# ## StockCode and Description analysis

# In[38]:


df[['StockCode', 'Description']].sort_values(by='StockCode')


# => Description is the text corresponding to StockCode

# In[39]:


print('Number of unique Description : ' + str(len(df['Description'].unique())))
print('Number of unique StockCode : ' + str(len(df['StockCode'].unique())))


# => There are more descriptions than stock codes. Are there inconsistencies with some description texts ?

# In[40]:


progbar = tqdm(range(len(df['StockCode'].unique())))

stockcodes_defaults = []

for stockcode_id in df['StockCode'].unique():    
    # If number of unique description values is different from 1, we have some anomaly in description
    if ((len(df[df['StockCode'] == stockcode_id]['Description'].unique())) != 1):
        stockcodes_defaults.append(stockcode_id)   
    
    progbar.update(1)


# In[41]:


print('=> ' + str(len(stockcodes_defaults)) + ' products do not always have the same description text for each order')


# Let's explore that : we have some differences due to coma or added words/letters

# In[42]:


qgrid_show(df[df['StockCode'].isin(stockcodes_defaults)].sort_values(by='StockCode'))


# In[43]:


# Print description that has most occurences for a stock code :
df[df['StockCode'] == '21232']['Description'].value_counts().sort_values(ascending=False).index[0]


# ### Assign 1 unique description for each product

# In[44]:


progbar = tqdm(range(len(df['StockCode'].unique())))

ref_descriptions = {}

# For each stock code : assign most represented description value in the dataset
for stockcode_id in df['StockCode'].unique():    
    ref_descriptions[stockcode_id] = df[df['StockCode'] == stockcode_id]['Description'].value_counts().sort_values(ascending=False).index[0]
    
    progbar.update(1)


# In[45]:


df['DescriptionNormalized'] = df['StockCode'].apply(lambda val : ref_descriptions[val] )


# In[46]:


qgrid_show(df[df['StockCode'].isin(stockcodes_defaults)].sort_values(by='StockCode'))


# In[47]:


print('Number of unique Description : ' + str(len(df['DescriptionNormalized'].unique())))
print('Number of unique StockCode : ' + str(len(df['StockCode'].unique())))


# # Add some features

# In[48]:


def get_month(x) : return dt.datetime(x.year,x.month,1)
df['InvoiceMonth'] = df['InvoiceDate'].apply(get_month)
#df_nocancel['InvoiceMonth'] = df_nocancel['InvoiceDate'].apply(get_month)
df_nocancel = df[df['InvoiceNo'].str.startswith('C') == False]
df_nocancel.reset_index(inplace=True)


# In[49]:


df_nocancel


# # Quantitative analysis

# ## Customer value analysis

# In[50]:


print('Total value (in £) : {:.2f}'.format(df_nocancel['TotalPrice'].sum()))


# In[51]:


print('Number of clients : ' + str(len(df_nocancel['CustomerID'].unique())))


# In[52]:


print('Number of products : ' + str(len(df_nocancel['StockCode'].unique())))


# In[53]:


print('Mean total price per client :')
df_nocancel[['CustomerID', 'TotalPrice']].groupby('CustomerID').sum().mean()


# In[54]:


print('Example of orders for 1 client and 1 product :')
pd.set_option('display.max_rows', 100)
df_nocancel[(df_nocancel['CustomerID'] == '17850') & (df_nocancel['StockCode'] == '85123A')] 


# In[55]:


fig = plt.figure()
fig.suptitle('Distribution of total price')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.hist(df_nocancel['TotalPrice'], bins=50)
#ax.set_xlim([0,4000])
plt.ylabel("Number of orders")
plt.xlabel('Total price of order')
plt.yscale('log')


# In[56]:


fig = plt.figure()
fig.suptitle('Distribution of total price')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
sns.boxplot(df_nocancel['TotalPrice'])
#ax.set_xlim([0,4000])
plt.xlabel("TotalPrice of order")
plt.xscale('log')


# In[57]:


### Distribution of TotalPrice of customers


# In[58]:


df_nocancel[['CustomerID', 'TotalPrice']].groupby('CustomerID').sum()['TotalPrice']


# In[59]:


fig = plt.figure()
fig.suptitle('Distribution of total price of customers')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.hist(df_nocancel[['CustomerID', 'TotalPrice']].groupby('CustomerID').sum()['TotalPrice'], bins=50)
#ax.set_xlim([0,4000])
plt.xlabel('Total price paid by customer')
plt.ylabel("Number of customers")
plt.yscale('log')


# In[60]:


fig = plt.figure()
fig.suptitle('Distribution of total price of customers')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
sns.distplot(df_nocancel[['CustomerID', 'TotalPrice']].groupby('CustomerID').sum()['TotalPrice'], kde=False, rug=True)
#ax.set_xlim([0,4000])
plt.xlabel('Total price paid by customer')
plt.ylabel("Number of customers")
plt.yscale('log')


# In[61]:


fig = plt.figure()
fig.suptitle('Distribution of total price of customers')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
sns.distplot(np.log(1+df_nocancel[['CustomerID', 'TotalPrice']].groupby('CustomerID').sum()['TotalPrice']), kde=False, rug=True)
#ax.set_xlim([0,4000])
plt.xlabel('Total price paid by customer')
plt.ylabel("Number of customers")
plt.yscale('log')


# ### Number of customer that represent 80% of value

# In[62]:


df_gbcustom = df_nocancel[['CustomerID', 'TotalPrice']].groupby('CustomerID').sum()['TotalPrice']
df_gbproduct = df_nocancel[['StockCode', 'TotalPrice']].groupby('StockCode').sum()['TotalPrice']


# In[63]:


value_80p = 0.80*df_gbcustom.sum()


# In[64]:


value_80p


# In[65]:


print('Number of clients that represent 80% of value : {:d}'      .format(df_gbcustom[df_gbcustom.sort_values(ascending=False).cumsum() < value_80p].sort_values(ascending=False).shape[0]))


# In[66]:


print('Top 20 earned value customers')
display(df_gbcustom[df_gbcustom.sort_values(ascending=False).cumsum() < value_80p].sort_values(ascending=False).head(20))
print('Earn value of top 20 customers : {:.2f} £'.format(df_gbcustom[df_gbcustom.sort_values(ascending=False).cumsum() < value_80p].sort_values(ascending=False).head(20).sum()))


# In[67]:


len(df_gbcustom.sort_values(ascending=False))


# In[68]:


fig = plt.figure()
fig.suptitle('Earn value by customers')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(range(0,len(df_gbcustom.sort_values(ascending=False))), df_gbcustom.sort_values(ascending=False))
ax.set_xlim([0,500])
plt.xlabel('Customers (ordered by earned value descending)')
plt.ylabel("Earned value")
#plt.yscale('log')


# => Elbow around 200

# In[69]:


fig = plt.figure()
fig.suptitle('Earn value by product')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(range(0,len(df_gbproduct.sort_values(ascending=False))), df_gbproduct.sort_values(ascending=False))
ax.set_xlim([0,1000])
plt.xlabel('Products (ordered by earned value descending)')
plt.ylabel("Earned value")
#plt.yscale('log')


# In[70]:


df_nocancel[df_nocancel['StockCode'].isin(df_gbproduct.sort_values(ascending=False).head(800).index)]['TotalPrice'].sum()


# In[71]:


print('Value earned by top 200 customers represent 50% of total value:')
df_nocancel[df_nocancel['CustomerID'].isin(df_gbcustom.sort_values(ascending=False).head(200).index)]['TotalPrice'].sum()


# In[72]:


df_top200 = df_nocancel[df_nocancel['CustomerID'].isin(df_gbcustom.sort_values(ascending=False).head(200).index)]


# In[73]:


fig = plt.figure()
fig.suptitle('Total price earned by product on top 200 customers')

ax = plt.gca()
#plt.hist(df_nocancel['TotalPrice'], bins=50, range=(0,100))
plt.scatter(range(0,len(df_top200['StockCode'].unique())), df_top200[['StockCode', 'TotalPrice']].groupby('StockCode').sum())
#ax.set_xlim([0,500])
plt.xlabel('Product id')
plt.ylabel("Earned value")
#plt.yscale('log')


# ## Value accross time

# In[74]:


df_nocancel['InvoiceMonth'].sort_values(ascending=True).unique()


# In[75]:


df_nocancel[['InvoiceMonth', 'TotalPrice']].sort_values(by='InvoiceMonth', ascending=True).groupby('InvoiceMonth').sum().plot.bar(title='Total amount of orders accross months')


# ## Value by country

# In[76]:


df_nocancel[['Country', 'TotalPrice']].groupby('Country').sum().sort_values(by='TotalPrice', ascending=False).plot.bar(title='Value by country')


# In[77]:


df_top200[['Country', 'TotalPrice']].groupby('Country').sum().sort_values(by='TotalPrice', ascending=False).plot.bar(title='Value by country on top 200 customers')


# In[78]:


df_nocancel[df_nocancel['Country'] != 'United Kingdom']['DescriptionNormalized'].sample(100)


# => Other countries than UK represent 10% of earned value. Product descriptions are in English and common to the others  
# => We can keep orders from all countries, but we will not keep country to avoid unbalance

# # Feature engineering

# ## CustomerID, Quantity, TotalPrice paid

# In[79]:


df_clients = df_nocancel[['CustomerID', 'Quantity', 'TotalPrice']].groupby('CustomerID').sum().copy(deep=True)


# In[80]:


df_clients


# ## Country

# In[81]:


df_clients = pd.concat([df_clients, df_nocancel[['CustomerID', 'Country']].groupby('CustomerID')['Country'].unique().str[0]], axis=1)


# ## Flag clients that have cancelled at least 1 command

# In[82]:


df_clients['HasEverCancelled'] = False


# In[83]:


df_clients.loc[df_clients.index.isin(df[df['InvoiceNo'].str.startswith('C')]['CustomerID'].unique().tolist()), 'HasEverCancelled'] = True


# In[84]:


df_clients


# ## Product description bag of words

# In[85]:


df_nocancel.shape


# In[86]:


df_nocancel['DescriptionNormalized']


# In[87]:


df_nocancel.head(10)


# In[88]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=0.001)

matrix_vectorized = vectorizer.fit_transform(df_nocancel['DescriptionNormalized'])


# In[89]:


# Ordered column names :
#[k for k, v in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])]


# In[90]:


matrix_vectorized


# In[91]:


bow_features = ['desc_' + str(s) for s in vectorizer.get_feature_names()]


# In[92]:


df_vectorized = pd.DataFrame(matrix_vectorized.todense(), columns=bow_features, dtype='int8')
del matrix_vectorized


# In[93]:


df_vectorized.info()


# In[94]:


df_nocancel.shape


# In[95]:


df_vectorized


# In[96]:


#df_vectorized.todense()


# In[97]:


df_nocancel.shape


# In[98]:


df_vectorized.shape


# In[99]:


df_nocancel.index


# In[100]:


df_nocancel_bow = pd.concat([df_nocancel, df_vectorized], axis=1)


# In[101]:


df_nocancel_bow.shape


# In[102]:


df_nocancel.head(2)


# ### Add a feature for top 200 products (that earn 50% of value)

# In[103]:


df_nocancel[df_nocancel['StockCode'].isin(df_gbproduct.sort_values(ascending=False).head(200).index)]['TotalPrice'].sum()


# In[104]:


df_nocancel_bow['Top200Value'] = 0


# In[105]:


df_nocancel_bow.loc[df_nocancel_bow['StockCode'].isin(df_gbproduct.sort_values(ascending=False).head(200).index), 'Top200Value'] = 1


# # Representation of products (dimensionality reduction)

# In[106]:


other_features = ['TotalPrice', 'Top200Value']


# In[107]:


ORDER_FEATS = bow_features + other_features


# In[108]:


df_nocancel_bow.info()


# In[109]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScalerMultiple(features_toscale=other_features)

df_nocancel_bow = scaler.fit_transform(df_nocancel_bow)


# In[110]:


df_nocancel_bow['TotalPrice'].sort_values(ascending=False)


# In[111]:


df_nocancel_bow['TotalPrice'].quantile(0.50)


# In[112]:


df_nocancel_bow.head(5)


# In[113]:


print('Start')
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

# choix du nombre de composantes à calculer
n_comp = 6

# import de l'échantillon
data = df_nocancel_bow

# selection des colonnes à prendre en compte dans l'ACP
#data_pca = df [numerical_features]

print('Start of reduction jobs')
#data_pca = df_nocancel_bow[ORDER_FEATS].copy()
data_pca = df_nocancel_bow[ORDER_FEATS]
 
print('Binarisation of color categories...')
bins = [-np.inf,df_nocancel_bow['TotalPrice'].quantile(0.25),        df_nocancel_bow['TotalPrice'].quantile(0.50),        df_nocancel_bow['TotalPrice'].quantile(0.75),        df_nocancel_bow['TotalPrice'].quantile(1)]

labels = [0, 1, 2, 3]
df_score_cat = pd.cut(data_pca['TotalPrice'], bins=bins, labels=labels)

# préparation des données pour l'ACP
#data_pca = data_pca.dropna()

X = data_pca.values
#names = data["idCours"] # ou data.index pour avoir les intitulés

#features = data.columns
features = data_pca.columns

# Centrage et Réduction

#std_scale = preprocessing.StandardScaler().fit(X)
#X_scaled = std_scale.transform(X)
X_scaled = X

print('PCA reduction...')
# Calcul des composantes principales
pca = decomposition.PCA(n_components=n_comp)
print('fit...')
pca.fit(X_scaled)

# Eboulis des valeurs propres
#display_scree_plot(pca)

# Cercle des corrélations
pcs = pca.components_
#plt.figure(figsize=(16,10))
plt.rcParams["figure.figsize"] = [16,9]
#display_circles(pcs, n_comp, pca, [(0,1),(2,3),(4,5)], labels = np.array(features))


# Projection des individus
print('Transform...')
X_projected = pca.transform(X_scaled)


# In[114]:


print('Display factorial planes')
display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3),(4,5)], illustrative_var=df_score_cat)
#display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3),(4,5)])

plt.show()


# In[115]:


display_scree_plot(pca)


# In[116]:



q1 = df_nocancel_bow['TotalPrice'].quantile(0.25)
q2 = df_nocancel_bow['TotalPrice'].quantile(0.50)
q3 = df_nocancel_bow['TotalPrice'].quantile(0.75)
q4 = df_nocancel_bow['TotalPrice'].quantile(1)


# In[117]:


from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#clusterer = KMeans(n_clusters=4, random_state=42).fit(data_pca)


# # Export cleaned data

# In[120]:


if not os.path.isdir(DATA_PATH_OUT):
    os.makedirs(DATA_PATH_OUT)

df.to_csv(DATA_PATH_FILE_OUTPUT, index=False)

