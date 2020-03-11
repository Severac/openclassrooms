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
        proxy = urllib.request.ProxyHandler({'https': 'https://user:pass@ncproxy:8080'})
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

# In[5]:


DATA_PATH_FILE = os.path.join(DATA_PATH, "*.csv")
ALL_FILES_LIST = glob.glob(DATA_PATH_FILE)


# ## Inspection de quelques lignes du fichier pour avoir un aperçu visuel du texte brut :

# In[6]:


def read_raw_file(nblines, data_path = DATA_PATH):
    csv_path = ALL_FILES_LIST[0]
    
    fp = open(csv_path)
    
    line = ""
    
    for cnt_lines in range(nblines+1):
        line = fp.readline()
        
    print(">>>>>> Line %d" % (cnt_lines))
    print(line)
    
    


# In[7]:


read_raw_file(0)
read_raw_file(1)
read_raw_file(2)


# ## Data load

# In[8]:


import pandas as pd

pd.set_option('display.max_columns', None)

# Time features by chronological order :
time_feats = ['CRS_DEP_TIME','DEP_DELAY','DEP_TIME', 'TAXI_OUT', 'WHEELS_OFF', 'AIR_TIME', 'CRS_ARR_TIME', 'WHEELS_ON','TAXI_IN','ARR_DELAY','ARR_TIME', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'DEP_TIME_BLK', 'ARR_TIME_BLK']

# hhmm timed features formatted
feats_hhmm = ['CRS_DEP_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON' , 'CRS_ARR_TIME', 'ARR_TIME']


def load_data(data_path=DATA_PATH):
    csv_path = DATA_PATH_FILE
    df_list = []
    
    for f in ALL_FILES_LIST:
        print(f'Loading file {f}')
        #df_list.append(pd.read_csv(f, sep=',', header=0, encoding='utf-8', error_bad_lines=False, low_memory=False))
        
        df_list.append(pd.read_csv(f, sep=',', header=0, encoding='utf-8', error_bad_lines=False, low_memory=False,
                                   parse_dates=feats_hhmm)
        )

        
    return pd.concat(df_list)


# In[9]:


df = load_data()


# In[10]:


df.reset_index(drop=True, inplace=True)


# ###  On vérifie que le nombre de lignes intégrées dans le Dataframe correspond au nombre de lignes du fichier

# In[11]:


num_lines = 0

for f in ALL_FILES_LIST:
    num_lines += sum(1 for line in open(f, encoding='utf-8'))
    
message = (
f'Nombre de lignes total (en comptant les entêtes): {num_lines} \n'
f"Nombre d'instances dans le dataframe: {df.shape[0]}"
)
print(message)


# ### Puis on affiche quelques instances de données :

# In[12]:


df.head(10)


# In[13]:


df['Unnamed: 64'].value_counts()


# => No value for last column "Unnamed: 64":  we drop it

# In[14]:


df.drop(labels='Unnamed: 64', axis=1, inplace=True)


# In[15]:


df['YEAR'].value_counts()


# => Une seule valeur pour la colonne YEAR (2016) sauf une ligne au 16/03/04 : l'information n'apportera donc rien pour les prédictions

# In[16]:


#df.drop(labels='YEAR', axis=1, inplace=True) # Drop will be done later in the notebook


# ### Liste des colonnes

# In[17]:


df.info()


# In[18]:


df.describe()


# ### Vérification s'il y a des doublons

# In[19]:


#df[df.duplicated()] # Code commented out because we have already executed it, and we know there are not duplicates


# ### Pas de suppression de doublons nécessaire

# In[20]:


#df.drop_duplicates(inplace=True) # Code commented out because we have already executed it, and we know there are not duplicates


# # Flight lifecycle information

# In[21]:


df_lifecycle = pd.read_csv('Flight_lifecycle.csv')


# ![image](Flight_lifecycle.png)

# In[22]:


df_lifecycle


# ACTUAL_ELAPSED_TIME = TAXI_OUT + AIR_TIME + TAXI_IN

# In[23]:


df[time_feats].head(15)


# # Quality of data analysis and first removals of useless data

# ## Display column names with their percentage of filled values (non NA)
# L'objectif est de voir quelles sont les features qui seront les plus fiables en terme de qualité de donnée, et quelles sont celles pour lesquelles on devra faire des choix

# In[24]:


pd.set_option('display.max_rows', 100)
(df.count()/df.shape[0]).sort_values(axis=0, ascending=False)


# ## Display of different possible values for qualitative features

# In[25]:


def print_column_information(df, column_name):
    column_type = df.dtypes[column_name]
    print(f'Column {column_name}, type {column_type}\n')
    print('--------------------------')

    print(df[[column_name]].groupby(column_name).size().sort_values(ascending=False))
    print(df[column_name].unique())    
    print('\n')

    
for column_name in df.select_dtypes(include=['object']).columns:
    print_column_information(df, column_name)


# ## Identifier of air company : columns analysis and determine which feature to keep

# In[26]:


df[['AIRLINE_ID']].groupby('AIRLINE_ID').size().sort_values(ascending=False)


# In[27]:


df[['UNIQUE_CARRIER']].groupby('UNIQUE_CARRIER').size().sort_values(ascending=False)


# => Les deux champs sont équivalents. On conservera UNIQUE_CARRIER, et on enlèvera la ligne qui contient la valeur 10397 (outlier)

# In[28]:


df.drop(index=df[df['UNIQUE_CARRIER'] == '10397'].index, axis=0, inplace=True)


# In[29]:


df[df['UNIQUE_CARRIER'] == '10397']


# ## Identifier of airport : columns analysis and determine which feature to keep

# In[30]:


df['DEST_AIRPORT_ID'] = df['DEST_AIRPORT_ID'].astype('str')  # Data clean (we have a mixed type of int and str on original data)
df['DEST'] = df['DEST'].astype('str')  # Data clean (we have a mixed type of int and str on original data)


# In[31]:


df[['DEST_AIRPORT_ID']].groupby('DEST_AIRPORT_ID').size().sort_values(ascending=False).head(5)


# In[32]:


df[['DEST']].groupby('DEST').size().sort_values(ascending=False).head(5)


# => We see that DEST is equivalent to DEST_AIRPORT_ID
# => ORIGIN will also be equivalent to ORIGIN_AIRPORT_ID
# 
# => We'll keep ORIGIN and DEST features

# ## We see that MONTH and DAY_OF_MONTH are equivalent to FL_DATE (without the year)
# We can keep MONTH and DAY_OF_MONTH instead of FL_DATE

# In[33]:


df[['FL_DATE', 'MONTH', 'DAY_OF_MONTH']].sample(10)


# ## Analysis of DELAY_NEW to see what this variable means and if we need it

# In[34]:


df['DEP_DELAY_NEW'].unique()


# In[35]:


df['DEP_DELAY_NEW'].hist(bins=50)


# In[36]:


df[df['DEP_DELAY_NEW'] < 100]['DEP_DELAY_NEW'].hist(bins=50)


# In[37]:


df[df['DEP_DELAY_NEW'] == 0]['DEP_DELAY_NEW'].count()


# In[38]:


df[df['DEP_DELAY_NEW'] > 0][['DEP_DELAY_NEW', 'DEP_DELAY']].sample(10)


# In[39]:


df[['DEP_DELAY_NEW', 'DEP_DELAY']].sample(10)


# In[40]:


(df[df['DEP_DELAY'] > 0]['DEP_DELAY'] - df[df['DEP_DELAY'] > 0]['DEP_DELAY_NEW']).unique()


# In[41]:


s_delay = (df[df['DEP_DELAY'] > 0]['DEP_DELAY'] - df[df['DEP_DELAY'] > 0]['DEP_DELAY_NEW']) != 0


# In[42]:


s_delay[s_delay == True]


# => Only one row has different value for DEP_DELAY and DEP_DELAY_NEW when DEP_DELAY > 0

# In[43]:


df[df['DEP_DELAY'] > 0].loc[[3376972]]


# => We see that DEP_DELAY_NEW is the same as DEP_DELAY when DEP_DELAY >=0,  and that DEP_DELAY_NEW is 0 when DEP_DELAY is < 0
# => We'll not keep DEP_DELAY_NEW since we're not interested in predicting negative delays  (= planes that arrive before schedule)

# In[44]:


df[df['ARR_DEL15'] == 1][['ARR_DEL15','TAIL_NUM']].groupby(['ARR_DEL15','TAIL_NUM']).size().sort_values(ascending=False).hist(bins=50)


# ## Analysis of FLIGHTS variable 

# In[45]:


df['FLIGHTS'].unique()


# In[46]:


df[df['FLIGHTS'].notnull() == False]


# => All values are 1 except 3 that are nan ! => We'll not use FLIGHTS as a feature

# # Some information about delays

# ## Display of delays grouped by tail number (plane identifier)

# In[47]:


pd.set_option('display.max_rows', 50)
df_delays_groupby_tails = df[df['ARR_DEL15'] == 1][['ARR_DEL15','TAIL_NUM']].groupby(['ARR_DEL15','TAIL_NUM']).size().sort_values(ascending=False)
df_delays_groupby_tails


# In[48]:


X_tails = range(df_delays_groupby_tails.shape[0])
Y_tails = df_delays_groupby_tails.to_numpy()


# In[49]:


X_tails


# In[50]:


plt.title('Plane delays')
plt.ylabel("Number of delays")
plt.xlabel("Tails ID of plane")
plt.plot(X_tails, Y_tails)


# ## Mean delay by carrier

# In[51]:


fig, ax = plt.subplots()
df[['ARR_DELAY', 'UNIQUE_CARRIER']].groupby('UNIQUE_CARRIER').mean().plot.bar(figsize=(16,10), title='Mean delay by carrier', ax=ax)
ax.legend(["Mean delay in minutes"])


# ## Mean delay by day of week

# In[52]:


df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].astype(str)


# In[53]:


fig, ax = plt.subplots()
df[['ARR_DELAY', 'DAY_OF_WEEK']].groupby('DAY_OF_WEEK').mean().plot.bar(figsize=(16,10), title='Mean delay by day of week', ax=ax)
ax.legend(["Mean delay in minutes"])


# ## Mean delay by month

# In[54]:


fig, ax = plt.subplots()
df[['ARR_DELAY', 'MONTH']].groupby('MONTH').mean().plot.bar(figsize=(16,10), title='Mean delay by month', ax=ax)
ax.legend(["Mean delay in minutes"])


# In[98]:


#pd.to_datetime(df['CRS_DEP_TIME'], format="%I%M")


# # Feature engineering

# ## Identification of features to keep for the model

# We will keep following features :  
#   
# ORIGIN                   1.000000 => Origin airport  
# CRS_DEP_TIME             1.000000 => we'll keep only the hour.  Maybe cut it into bins.  (compare result with and without binning)
# MONTH                    1.000000  
# DAY_OF_MONTH             1.000000  
# DAY_OF_WEEK              1.000000    
# UNIQUE_CARRIER           1.000000 => Flight company   
# DEST                     1.000000 => Destination airport  
# CANCELLED                0.999999 => to keep to construct a delay label , for later  
# CRS_ARR_TIME             0.999999  
# DIVERTED                 0.999999 => use this to construct delay label, for later
# DISTANCE                 0.999999   
# CRS_ELAPSED_TIME         0.999998 => carrier scheduled elapsed time  => redundant
# ARR_DELAY                0.985844  
# 
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
# DEST_AIRPORT_ID          1.000000    
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
#  
#  FLIGHTS                  0.999999 => Number of flights 
#  All values are 1 except three of them  => useless feature
#  
#  
#  DEP_DELAY                0.988726  => we'll only predict arrival delays and not departure delays
# DEP_DEL15                0.988726  => we'll only predict arrival delays and not departure delays
# DEP_DELAY_GROUP          0.988726  => we'll only predict arrival delays and not departure delays
# 
# ARR_DELAY_GROUP          0.985844  => redundant with ARR_DELAY
# ARR_DEL15                0.985844  => redundant with ARR_DELAY
# ARR_DELAY_NEW            0.985844  => redundant with ARR_DELAY 
# 
# ARR_TIME                 0.987937  => Not kept : arrival time  (we already have arrival delay information)
# ACTUAL_ELAPSED_TIME      0.985844  => Not kept : actual information, know only at the last moment (but we keep scheduled information CRS_ELAPSED_TIME)
# AIR_TIME                 0.985844  => Not kept : actual information, know only at the last moment. Would be date leak. Difference between ACTUAL_ELAPSE_TIME ??  
# DEP_TIME                 0.988726 => Not kept : actual information, know only at the last moment. Would be date leak.
# 
# QUARTER                  1.000000 => redundant with MONTH

# In[55]:


df[['ARR_DEL15','TAIL_NUM']].groupby(['ARR_DEL15','TAIL_NUM'])


# ## Identification of quantitative and qualitative features

# In[56]:


df.columns[1]


# In[57]:


# Below are feature from dataset that we decided to keep: 
all_features = ['ORIGIN','CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','DEST','CANCELLED','CRS_ARR_TIME','DIVERTED','DISTANCE','CRS_ELAPSED_TIME','ARR_DELAY']

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


# ## Data cleaning of our features

# ### Quality of data

# In[58]:


(df[all_features].count()/df[all_features].shape[0]).sort_values(axis=0, ascending=False)


# In[59]:


df[df['DEP_TIME'].notnull() == False].sample(20)


# => We see that when flight is cancelled (value 1), we don't have actual delay values which is normal  
# => We may want to keep these values later, to be able to predict cancellations.  But for now, our model will not consider cancellations as delay.

# In[60]:


df[df['CANCELLED'] == 1].shape


# => Only 65973 cancelled flights on 5M total. Data seems very thin to make predictions.  
# => If we want to make cancellation predictions, we'll use another model dedicated to this task

# We also have nan values that correspond to DIVERTED flights :

# In[61]:


df[df['DEP_TIME'].notnull() == False].shape


# In[62]:


df[df['DIVERTED'] == 1].shape


# In[63]:


df[df['DIVERTED'] == 1]


# Let's check flights that have arrival delay null, but not cancelled  (cancelled flights do have null arrival delay : in that case it's normal)

# In[64]:


df[(df['ARR_DELAY'].notnull() == False) & (df['CANCELLED'] == 0)].shape


# In[65]:


df[(df['ARR_DELAY'].notnull() == False) & (df['CANCELLED'] == 0)][all_features].sample(10)


# => We see that DIVERTED == 1 for those lines : that's why we don't have delay information

# ### Display of qualitative features values :

# In[66]:


for feature_name in qualitative_features:
    print_column_information(df, feature_name)


# ### Display of quantiative features values :

# In[67]:


pd.set_option('display.max_rows', 10000)
for column_name in quantitative_features:
    #print(df[column_name].value_counts)
    print_column_information(df, column_name)


# ### Conversion of qualitative features into clean str format

# In[68]:


for feature_name in qualitative_features:
    df[feature_name] = df[feature_name].astype(str)


# ### Display qualitative features again

# In[69]:


pd.set_option('display.max_rows', 1000)
for feature_name in qualitative_features:
    print_column_information(df, feature_name)


# ### Cleaning outliers on qualitative features

# => What we can see from above :  
# Origin : 5 airports appear only 1 time: too few informations to make predictions   
# 
# SPN          1  
# BFF          1  
# MHK          1  
# ENV          1  
# EFD          1  
# 4.00
# 
# DEST_AIRPORT_ID :  
# 
# 4 outilers :  
# 
# 7.00          1  
# 13290         1  
# 14955         1  
# -1            1  
# 
# DEST :   
# 
# 4 outliers :  
# 
# MHK               1  
# SPN               1  
# 1800-1859         1  

# Creating new dataframe with only the features we keep, to gain memory :

# In[70]:


df_new = df[all_features].copy(deep = True)


# In[71]:


del df


# In[72]:


df = df_new


# #### Clean ORIGIN

# In[73]:


df.drop(index=df[df['ORIGIN'].isin(['SPN', 'BFF', 'MHK', 'ENV', 'EFD', '4.00'])].index, axis=0, inplace=True)


# #### Clean DEST

# In[74]:


df.drop(index=df[df['DEST'].isin(['MHK', 'SPN', '1800-1859'])].index, axis=0, inplace=True)


# ### Display quantitative features distributions

# In[75]:


df.hist(bins=50, figsize=(20,15), log=True)


# ## Correlation matrix of quantitative features

# In[76]:


corr_matrix = df.corr()


# In[77]:


corr_matrix[quantitative_features].loc[quantitative_features]


# In[78]:


plt.figure(figsize=(16, 10))
plt.title('Corrélation entre les valeurs numériques')
sns.heatmap(corr_matrix[quantitative_features].loc[quantitative_features], 
        xticklabels=corr_matrix[quantitative_features].loc[quantitative_features].columns,
        yticklabels=corr_matrix[quantitative_features].loc[quantitative_features].columns, cmap='coolwarm', center=0.20)


# # Cercle des corrélations et première réduction de dimensionalité des variables numériques

# In[79]:


#common_functions.display_projections(df.sample(10000), quantitative_features)


# # Annexe : ancien code inutile

# #### Clean DEST_AIRPORT_ID

# df[df['DEST_AIRPORT_ID'].isin(['7.00', '13290', '14955', '-1'])]

# df.drop(index=df[df['DEST_AIRPORT_ID'].isin(['7.00', '13290', '14955', '-1'])].index, axis=0, inplace=True)
