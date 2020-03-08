#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ4 : transats dataset :  data exploration notebook 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import zipfile
import urllib

import matplotlib.pyplot as plt

import numpy as np

import qgrid

import glob

from pandas.plotting import scatter_matrix

DOWNLOAD_ROOT = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Anticipez+le+retard+de+vol+des+avions+-+109/"
DATA_PATH = os.path.join("datasets", "transats")

DATA_PATH_FILE = os.path.join(DATA_PATH, "*.csv")

ALL_FILES_LIST = glob.glob(DATA_PATH_FILE)

DATA_URL = DOWNLOAD_ROOT + "Dataset+Projet+4.zip"

ARCHIVE_PATH_FILE = os.path.join(DATA_PATH, "Dataset+Projet+4.zip")

DATA_PATH_FILE_OUTPUT = os.path.join(DATA_PATH, "transats_metadata_transformed.csv")

DOWNLOAD_DATA = False  # A la première exécution du notebook, ou pour rafraîchir les données, mettre cette variable à True

plt.rcParams["figure.figsize"] = [16,9] # Taille par défaut des figures de matplotlib

import seaborn as sns
sns.set()

#import common_functions




# In[2]:


def qgrid_show(df):
    display(qgrid.show_grid(df, grid_options={'forceFitColumns': False, 'defaultColumnWidth': 170}))


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
        proxy = urllib.request.ProxyHandler({'https': 'https://login:password@ncproxy:8080'})
        # construct a new opener using your proxy settings
        opener = urllib.request.build_opener(proxy)
        # install the openen on the module-level
        urllib.request.install_opener(opener)    
    
    urllib.request.urlretrieve(data_url, archive_path)
    data_archive = zipfile.ZipFile(archive_path)
    data_archive.extractall(path=data_path)
    data_archive.close()


# In[4]:


if (DOWNLOAD_DATA == True):
    fetch_dataset()


# # Import of CSV file

# ## Inspection de quelques lignes du fichier pour avoir un aperçu visuel du texte brut :

# In[5]:


def read_raw_file(nblines, data_path = DATA_PATH):
    csv_path = ALL_FILES_LIST[0]
    
    fp = open(csv_path)
    
    line = ""
    
    for cnt_lines in range(nblines+1):
        line = fp.readline()
        
    print(">>>>>> Line %d" % (cnt_lines))
    print(line)
    
    


# In[6]:


read_raw_file(0)
read_raw_file(1)
read_raw_file(2)


# ## Data load

# In[7]:


import pandas as pd

pd.set_option('display.max_columns', None)

def load_data(data_path=DATA_PATH):
    csv_path = DATA_PATH_FILE
    df_list = []
    
    for f in ALL_FILES_LIST:
        print(f'Loading file {f}')
        df_list.append(pd.read_csv(f, sep=',', header=0, encoding='utf-8', error_bad_lines=False, low_memory=False))
    
    return pd.concat(df_list)


# In[8]:


df = load_data()


# In[9]:


df.reset_index(drop=True, inplace=True)


# ###  On vérifie que le nombre de lignes intégrées dans le Dataframe correspond au nombre de lignes du fichier

# In[10]:


num_lines = 0

for f in ALL_FILES_LIST:
    num_lines += sum(1 for line in open(f, encoding='utf-8'))
    
message = (
f'Nombre de lignes total (en comptant les entêtes): {num_lines} \n'
f"Nombre d'instances dans le dataframe: {df.shape[0]}"
)
print(message)


# ### Puis on affiche quelques instances de données :

# In[11]:


df.head(10)


# In[12]:


df['Unnamed: 64'].value_counts()


# => Aucune valeur pour la dernière colonne "Unnamed: 64":  on la drop donc

# In[13]:


df.drop(labels='Unnamed: 64', axis=1, inplace=True)


# In[14]:


df['YEAR'].value_counts()


# => Une seule valeur pour la colonne YEAR (2016) sauf une ligne au 16/03/04 : l'information n'apportera donc rien pour les prédictions

# In[15]:


df.drop(labels='YEAR', axis=1, inplace=True)


# ### Liste des colonnes

# In[16]:


df.info()


# In[17]:


df.describe()


# ### Vérification s'il y a des doublons

# In[18]:


df[df.duplicated()]


# ### Pas de suppression de doublons nécessaire

# In[19]:


#df.drop_duplicates(inplace=True)


# # Quality of data analysis and first removals of useless data

# ## Affichage des champs renseignés (non NA) avec leur pourcentage de complétude
# L'objectif est de voir quelles sont les features qui seront les plus fiables en terme de qualité de donnée, et quelles sont celles pour lesquelles on devra faire des choix

# In[20]:


pd.set_option('display.max_rows', 100)
(df.count()/df.shape[0]).sort_values(axis=0, ascending=False)


# ## Affichage des différentes valeurs possibles pour les features qualitatives

# In[21]:


def print_column_information(df, column_name):
    print(f'Column {column_name}')
    print('--------------------------')

    print(df[[column_name]].groupby(column_name).size().sort_values(ascending=False))
    print(df[column_name].unique())    
    print('\n')

for column_name in df.select_dtypes(include=['object']).columns:
    #print(df[column_name].value_counts)
    print_column_information(df, column_name)


# ## Identifiant de la compagnie : examen des champs et voir quel champ conserver

# In[22]:


df[['AIRLINE_ID']].groupby('AIRLINE_ID').size().sort_values(ascending=False)


# In[23]:


df[['UNIQUE_CARRIER']].groupby('UNIQUE_CARRIER').size().sort_values(ascending=False)


# => Les deux champs sont équivalents. On conservera UNIQUE_CARRIER, et on enlèvera la ligne qui contient la valeur 10397 (outlier)

# In[24]:


df.drop(index=df[df['UNIQUE_CARRIER'] == '10397'].index, axis=0, inplace=True)


# In[25]:


df[df['UNIQUE_CARRIER'] == '10397']


# ## Constat que les informations MONTH et DAY_OF_MONTH sont équivalentes à FL_DATE (sans l'année)
# On pourra donc conserver MONTH et DAY_OF_MONTH à la place de FL_DATE

# In[26]:


df[['FL_DATE', 'MONTH', 'DAY_OF_MONTH']].sample(10)


# ## Observation of DELAY_NEW to see what this variable means

# In[27]:


df['DEP_DELAY_NEW'].unique()


# In[28]:


df['DEP_DELAY_NEW'].hist(bins=50)


# In[29]:


df[df['DEP_DELAY_NEW'] < 100]['DEP_DELAY_NEW'].hist(bins=50)


# In[30]:


df[df['DEP_DELAY_NEW'] == 0]['DEP_DELAY_NEW'].count()


# In[31]:


df[df['DEP_DELAY_NEW'] > 0][['DEP_DELAY_NEW', 'DEP_DELAY']].sample(10)


# In[32]:


df[['DEP_DELAY_NEW', 'DEP_DELAY']].sample(10)


# In[33]:


(df[df['DEP_DELAY'] > 0]['DEP_DELAY'] - df[df['DEP_DELAY'] > 0]['DEP_DELAY_NEW']).unique()


# In[34]:


s_delay = (df[df['DEP_DELAY'] > 0]['DEP_DELAY'] - df[df['DEP_DELAY'] > 0]['DEP_DELAY_NEW']) != 0


# In[35]:


s_delay[s_delay == True]


# => Only one row has different value for DEP_DELAY and DEP_DELAY_NEW when DEP_DELAY > 0

# In[36]:


df[df['DEP_DELAY'] > 0].loc[[3376972]]


# => We see that DEP_DELAY_NEW is the same as DEP_DELAY when DEP_DELAY >=0,  and that DEP_DELAY_NEW is 0 when DEP_DELAY is < 0
# => We'll not keep DEP_DELAY_NEW since we're not interested in predicting negative delays  (= planes that arrive before schedule)

# In[37]:


df[df['ARR_DEL15'] == 1][['ARR_DEL15','TAIL_NUM']].groupby(['ARR_DEL15','TAIL_NUM']).size().sort_values(ascending=False).hist(bins=50)


# # Some information about delays

# ## Display of delays grouped by tail number (plane identifier)

# In[38]:


pd.set_option('display.max_rows', 50)
df_delays_groupby_tails = df[df['ARR_DEL15'] == 1][['ARR_DEL15','TAIL_NUM']].groupby(['ARR_DEL15','TAIL_NUM']).size().sort_values(ascending=False)
df_delays_groupby_tails


# In[39]:


X_tails = range(df_delays_groupby_tails.shape[0])
Y_tails = df_delays_groupby_tails.to_numpy()


# In[40]:


X_tails


# In[41]:


plt.title('Plane delays')
plt.ylabel("Number of delays")
plt.xlabel("Tails ID of plane")
plt.plot(X_tails, Y_tails)


# ## Mean delay by carrier

# In[110]:


fig, ax = plt.subplots()
df[['ARR_DELAY', 'UNIQUE_CARRIER']].groupby('UNIQUE_CARRIER').mean().plot.bar(figsize=(16,10), title='Mean delay by carrier', ax=ax)
ax.legend(["Mean delay in minutes"])


# ## Mean delay by day of week

# In[117]:


df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].astype(str)


# In[121]:


fig, ax = plt.subplots()
df[['ARR_DELAY', 'DAY_OF_WEEK']].groupby('DAY_OF_WEEK').mean().plot.bar(figsize=(16,10), title='Mean delay by day of week', ax=ax)
ax.legend(["Mean delay in minutes"])


# ## Mean delay by month

# In[123]:


fig, ax = plt.subplots()
df[['ARR_DELAY', 'MONTH']].groupby('MONTH').mean().plot.bar(figsize=(16,10), title='Mean delay by month', ax=ax)
ax.legend(["Mean delay in minutes"])


# # Feature engineering

# ## Identification of features to keep for the model

# We will keep following features :  
#   
# QUARTER                  1.000000  
# ORIGIN                   1.000000 => Origin airport  
# DEST_AIRPORT_ID          1.000000  
# CRS_DEP_TIME             1.000000 => we'll keep only the hour.  Maybe cut it into bins.  
# MONTH                    1.000000  
# DAY_OF_MONTH             1.000000  
# DAY_OF_WEEK              1.000000    
# UNIQUE_CARRIER           1.000000  
# DEST                     1.000000 => Destination airport  
# CANCELLED                0.999999 => to keep to construct a delay label  
# CRS_ARR_TIME             0.999999  
# DIVERTED                 0.999999 => use this to construct delay label  
# DISTANCE                 0.999999  
# FLIGHTS                  0.999999 => Number of flights  
# CRS_ELAPSED_TIME         0.999998 => carrier scheduled elapsed time  
# DEP_TIME                 0.988726  
# DEP_DELAY                0.988726  
# DEP_DEL15                0.988726  
# DEP_DELAY_GROUP          0.988726  
# ARR_TIME                 0.987937  
# ARR_DELAY_GROUP          0.985844  
# ARR_DEL15                0.985844  
# ARR_DELAY                0.985844  
# ARR_DELAY_NEW            0.985844  
# ACTUAL_ELAPSED_TIME      0.985844  
# AIR_TIME                 0.985844  => Difference between ACTUAL_ELAPSE_TIME ??  
# 
# 
# 
# Columns that we will not use :  
# ORIGIN_CITY_MARKET_ID    1.000000  
# 
# Too close from origin airport :  
# ORIGIN_WAC               1.000000  
# ORIGIN_CITY_NAME         1.000000  
# ORIGIN_STATE_ABR         1.000000  
# ORIGIN_STATE_FIPS        1.000000  
# ORIGIN_STATE_NM          1.000000  
# 
# Too close from destination airport :  
# DEST_WAC                 1.000000  
# DEST_CITY_NAME           1.000000  
# DEST_STATE_ABR           1.000000  
# DEST_STATE_FIPS          1.000000  
# DEST_STATE_NM            1.000000  
# 
#   
# => But we may try later to use those instead of origin airport  
# 
# 
# ORIGIN_AIRPORT_ID        1.000000 => Origin airport ID  
# => Redundant with ORIGIN, and better formatted field  
#   
# DEST_AIRPORT_SEQ_ID      1.000000  
# => Redundant with DEST_AIRPORT_ID and DEST  
# 
# DEST_CITY_MARKET_ID      1.000000  
# 
# 
# ORIGIN_AIRPORT_SEQ_ID    1.000000  
# AIRLINE_ID               1.000000  
# => redundant with DEST  
# 
# CARRIER                  1.000000  
# => redundant with UNIQUE_CARRIER  
# 
# DEP_TIME_BLK             1.000000  
# => not useful for modeling. would create data leak.  
# 
# ARR_TIME_BLK             0.999999  
# => not useful for our model  
# 
# FL_NUM                   1.000000   
# => flight number. Identifier, not useful  
#   
# DISTANCE_GROUP           0.999999  
# => redundante with DISTANCE  
# 
# TAIL_NUM                 0.997738 => aircraft ID number printed on the tail  
#     => This feature would be very interesting.  Unfortunately, as a customer we do not know it until the last moment.  
#     => and as a carrier company, I guess it may be defined pretty late. So, including this information would be data leak.  
#     => But it would be interesting to know if certain planes have more delays than others  
#       
# TAXI_IN                  0.987938  
# For arriving flights: the Actual taXi-In Time  is the  
# period between the Actual Landing Time and the Actual In-Block Time (  
#   
# => Not included (data leak / we don't know the information until the last moment)  
# 
# TAXI_OUT                 0.988374  
# For departing flights: the Actual taXi-Out Time is  
# the period between the Actual Off-Block Time and the Actual Take Off Time .  
# => Not included (data leak / we don't know the information until the last moment)  
# 
# WHEELS_OFF               0.988374    
# Wheels Off Time (local time: hhmm)  
# => Not included (data leak / we don't know the information until the last moment)  
# 
# WHEELS_ON                0.987938  
# Wheels On Time (local time: hhmm)  
# => Not included (data leak / we don't know the information until the last moment)  
# 
# 
# CARRIER_DELAY            0.171832  
# WEATHER_DELAY            0.171832  
# NAS_DELAY                0.171832  
# SECURITY_DELAY           0.171832    
# LATE_AIRCRAFT_DELAY      0.171832  
# CANCELLATION_CODE        0.011706  
# 
# => Delay causes : not relevant  
# 
# TOTAL_ADD_GTIME          0.006127  
# FIRST_DEP_TIME           0.006127  
# LONGEST_ADD_GTIME        0.006127  
# 
# => Very specific information (gate return or cancelled return)  
# => know at the last moment and not useful to predict delays  
# 
# FL_DATE
# => Redundant with 'MONTH' and 'DAY_OF_MONTH
# 
# DEP_DELAY_NEW            0.988726  
#  DEP_DELAY_NEW is the same as DEP_DELAY when DEP_DELAY >=0,  and that DEP_DELAY_NEW is 0 when DEP_DELAY is < 0

# In[42]:


df[['ARR_DEL15','TAIL_NUM']].groupby(['ARR_DEL15','TAIL_NUM'])


# ## Identification of quantitative and qualitative features

# In[43]:


df.columns[1]


# In[44]:


# Below are feature from dataset that we decided to keep: 
all_features = ['QUARTER','ORIGIN','DEST_AIRPORT_ID','CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','DEST','CANCELLED','CRS_ARR_TIME','DIVERTED','DISTANCE','FLIGHTS','CRS_ELAPSED_TIME','DEP_TIME','DEP_DELAY_NEW','DEP_DELAY','DEP_DEL15','DEP_DELAY_GROUP','ARR_TIME','ARR_DELAY_GROUP','ARR_DEL15','ARR_DELAY','ARR_DELAY_NEW','ACTUAL_ELAPSED_TIME','AIR_TIME']

quantitative_features = []
qualitative_features = []
features_todrop = []

for feature_name in all_features:
    if (df[feature_name].dtype == 'object'):
        qualitative_features.append(feature_name)
        
    else:
        quantitative_features.append(feature_name)

for df_column in df.columns:
    if df_column not in all_features:
        features_todrop.append(df_column)
        
print(f'Quantitative features : {quantitative_features} \n')
print(f'Qualitative features : {qualitative_features} \n')

print(f'Features to drop : {features_todrop} \n')


# In[45]:


for feature_name in qualitative_features:
    print_column_information(df, feature_name)


# In[46]:


df['QUARTER'].astype(str).unique()


# In[47]:


df['QUARTER'].unique()


# In[48]:


df[quantitative_features].head(5)


# In[49]:


df.hist(bins=50, figsize=(20,15))


# In[50]:


corr_matrix = df.corr()


# In[51]:


corr_matrix[quantitative_features].loc[quantitative_features]


# In[52]:


plt.figure(figsize=(16, 10))
plt.title('Corrélation entre les valeurs numériques')
sns.heatmap(corr_matrix[quantitative_features].loc[quantitative_features], 
        xticklabels=corr_matrix[quantitative_features].loc[quantitative_features].columns,
        yticklabels=corr_matrix[quantitative_features].loc[quantitative_features].columns, cmap='coolwarm', center=0.20)


# # Cercle des corrélations et première réduction de dimensionalité des variables numériques

# In[53]:


#common_functions.display_projections(df.sample(10000), quantitative_features)

