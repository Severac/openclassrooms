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
DATA_PATH_OUT = os.path.join(DATA_PATH, "out")

DATA_URL = DOWNLOAD_ROOT + "Dataset+Projet+4.zip"

ARCHIVE_PATH_FILE = os.path.join(DATA_PATH, "Dataset+Projet+4.zip")

DATA_PATH_FILE_OUTPUT = os.path.join(DATA_PATH_OUT, "transats_metadata_transformed.csv")

DOWNLOAD_DATA = False  # A la première exécution du notebook, ou pour rafraîchir les données, mettre cette variable à True

plt.rcParams["figure.figsize"] = [16,9] # Taille par défaut des figures de matplotlib

import seaborn as sns
sns.set()

#import common_functions


### For progress bar :
from tqdm import tqdm_notebook as tqdm
                                        


# In[2]:


def qgrid_show(df):
    display(qgrid.show_grid(df, grid_options={'forceFitColumns': False, 'defaultColumnWidth': 170}))


# In[3]:


def display_percent_complete(df):
    not_na = 100 - (df.isnull().sum() * 100 / len(df))
    not_na_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_complete': not_na}).sort_values(by='percent_complete', ascending=False)
    display(not_na_df)


# # Download and decompression of data

# In[4]:


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


# In[5]:


if (DOWNLOAD_DATA == True):
    fetch_dataset()


# # Import of CSV file

# In[6]:


DATA_PATH_FILE = os.path.join(DATA_PATH, "*.csv")
ALL_FILES_LIST = glob.glob(DATA_PATH_FILE)


# ## Raw data display of some lines of the file :

# In[7]:


def read_raw_file(nblines, data_path = DATA_PATH):
    csv_path = ALL_FILES_LIST[0]
    
    fp = open(csv_path)
    
    line = ""
    
    for cnt_lines in range(nblines+1):
        line = fp.readline()
        
    print(">>>>>> Line %d" % (cnt_lines))
    print(line)
    
    


# In[8]:


read_raw_file(0)
read_raw_file(1)
read_raw_file(2)


# ## Data load

# In[9]:


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


# In[10]:


df = load_data()


# In[11]:


df.reset_index(drop=True, inplace=True)


# ###  On vérifie que le nombre de lignes intégrées dans le Dataframe correspond au nombre de lignes du fichier

# In[12]:


num_lines = 0

for f in ALL_FILES_LIST:
    num_lines += sum(1 for line in open(f, encoding='utf-8'))
    
message = (
f'Nombre de lignes total (en comptant les entêtes): {num_lines} \n'
f"Nombre d'instances dans le dataframe: {df.shape[0]}"
)
print(message)


# ### Puis on affiche quelques instances de données :

# In[13]:


df.head(10)


# In[14]:


df['Unnamed: 64'].value_counts()


# => No value for last column "Unnamed: 64":  we drop it

# In[15]:


df.drop(labels='Unnamed: 64', axis=1, inplace=True)


# In[16]:


df['YEAR'].value_counts()


# => Une seule valeur pour la colonne YEAR (2016) sauf une ligne au 16/03/04 : l'information n'apportera donc rien pour les prédictions

# In[17]:


#df.drop(labels='YEAR', axis=1, inplace=True) # Drop will be done later in the notebook


# ### Liste des colonnes

# In[18]:


df.info()


# In[19]:


df.describe()


# ### Vérification s'il y a des doublons

# In[20]:


#df[df.duplicated()] # Code commented out because we have already executed it, and we know there are not duplicates


# ### Pas de suppression de doublons nécessaire

# In[21]:


#df.drop_duplicates(inplace=True) # Code commented out because we have already executed it, and we know there are not duplicates


# # Flight lifecycle information

# In[22]:


df_lifecycle = pd.read_csv('Flight_lifecycle.csv')


# ![image](Flight_lifecycle.png)

# In[23]:


df_lifecycle


# ACTUAL_ELAPSED_TIME = TAXI_OUT + AIR_TIME + TAXI_IN

# In[24]:


df[time_feats].head(15)


# # Quality of data analysis and first removals of useless data

# ## Display column names with their percentage of filled values (non NA)
# L'objectif est de voir quelles sont les features qui seront les plus fiables en terme de qualité de donnée, et quelles sont celles pour lesquelles on devra faire des choix

# In[25]:


not_na = 100 - (df.isnull().sum() * 100 / len(df))
not_na_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_complete': not_na}).sort_values(by='percent_complete', ascending=False)
not_na_df


# ## Display of different possible values for qualitative features

# In[26]:


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

# In[27]:


df[['AIRLINE_ID']].groupby('AIRLINE_ID').size().sort_values(ascending=False)


# In[28]:


df[['UNIQUE_CARRIER']].groupby('UNIQUE_CARRIER').size().sort_values(ascending=False)


# => Les deux champs sont équivalents. On conservera UNIQUE_CARRIER, et on enlèvera la ligne qui contient la valeur 10397 (outlier)

# In[29]:


df.drop(index=df[df['UNIQUE_CARRIER'] == '10397'].index, axis=0, inplace=True)


# In[30]:


df[df['UNIQUE_CARRIER'] == '10397']


# ## Identifier of airport : columns analysis and determine which feature to keep

# In[31]:


df['DEST_AIRPORT_ID'] = df['DEST_AIRPORT_ID'].astype('str')  # Data clean (we have a mixed type of int and str on original data)
df['DEST'] = df['DEST'].astype('str')  # Data clean (we have a mixed type of int and str on original data)


# In[32]:


df[['DEST_AIRPORT_ID']].groupby('DEST_AIRPORT_ID').size().sort_values(ascending=False).head(5)


# In[33]:


df[['DEST']].groupby('DEST').size().sort_values(ascending=False).head(5)


# => We see that DEST is equivalent to DEST_AIRPORT_ID
# => ORIGIN will also be equivalent to ORIGIN_AIRPORT_ID
# 
# => We'll keep ORIGIN and DEST features

# ## We see that MONTH and DAY_OF_MONTH are equivalent to FL_DATE (without the year)
# We can keep MONTH and DAY_OF_MONTH instead of FL_DATE

# In[34]:


df[['FL_DATE', 'MONTH', 'DAY_OF_MONTH']].sample(10)


# ## Analysis of DELAY_NEW to see what this variable means and if we need it

# In[35]:


df['DEP_DELAY_NEW'].unique()


# In[36]:


df['DEP_DELAY_NEW'].hist(bins=50)


# In[37]:


df[df['DEP_DELAY_NEW'] < 100]['DEP_DELAY_NEW'].hist(bins=50)


# In[38]:


df[df['DEP_DELAY_NEW'] == 0]['DEP_DELAY_NEW'].count()


# In[39]:


df[df['DEP_DELAY_NEW'] > 0][['DEP_DELAY_NEW', 'DEP_DELAY']].sample(10)


# In[40]:


df[['DEP_DELAY_NEW', 'DEP_DELAY']].sample(10)


# In[41]:


(df[df['DEP_DELAY'] > 0]['DEP_DELAY'] - df[df['DEP_DELAY'] > 0]['DEP_DELAY_NEW']).unique()


# In[42]:


s_delay = (df[df['DEP_DELAY'] > 0]['DEP_DELAY'] - df[df['DEP_DELAY'] > 0]['DEP_DELAY_NEW']) != 0


# In[43]:


s_delay[s_delay == True]


# => Only one row has different value for DEP_DELAY and DEP_DELAY_NEW when DEP_DELAY > 0

# In[44]:


df[df['DEP_DELAY'] > 0].loc[[3376972]]


# => We see that DEP_DELAY_NEW is the same as DEP_DELAY when DEP_DELAY >=0,  and that DEP_DELAY_NEW is 0 when DEP_DELAY is < 0
# => We'll not keep DEP_DELAY_NEW since we're also interested in predicting negative delays  (= planes that arrive before schedule)

# In[45]:


df[df['ARR_DEL15'] == 1][['ARR_DEL15','TAIL_NUM']].groupby(['ARR_DEL15','TAIL_NUM']).size().sort_values(ascending=False).hist(bins=50)


# ## Analysis of FLIGHTS variable 

# In[46]:


df['FLIGHTS'].unique()


# In[47]:


df[df['FLIGHTS'].notnull() == False]


# => All values are 1 except 3 that are nan ! => We'll not use FLIGHTS as a feature

# # Some information about delays

# ## Display of delays grouped by tail number (plane identifier) compared to delays not grouped

# In[48]:


pd.set_option('display.max_rows', 50)
df_delays_groupby_tails = df[df['ARR_DELAY'] > 0][['TAIL_NUM', 'ARR_DELAY']].groupby(['TAIL_NUM']).mean().sort_values(by='ARR_DELAY', ascending=False)
df_delays_groupby_tails


# In[49]:


X_tails = range(df_delays_groupby_tails.shape[0])
Y_tails = df_delays_groupby_tails.to_numpy()


# In[50]:


X_tails


# In[51]:


plt.title('Plane delays')
plt.ylabel("Mean delay")
plt.xlabel("Tails ID of plane")
plt.plot(X_tails, Y_tails)


# In[52]:


pd.set_option('display.max_rows', 50)
df_delays = df['ARR_DELAY'].sort_values(ascending=False)
df_delays


# In[53]:


X_delay = range(df_delays.shape[0])
Y_delay = df_delays.to_numpy()

plt.title('Plane delays')
plt.ylabel("delay in minutes")
plt.xlabel("Flights")
plt.plot(X_delay, Y_delay)


# We see that plane model seems not to make any difference for delays.  
# We also see an elbow in the curve

# ## Mean delay by carrier

# In[54]:


fig, ax = plt.subplots()
df[['ARR_DELAY', 'UNIQUE_CARRIER']].groupby('UNIQUE_CARRIER').mean().plot.bar(figsize=(16,10), title='Mean delay by carrier', ax=ax)
ax.legend(["Mean delay in minutes"])


# ## Mean delay by origin airport

# In[55]:


df[['ARR_DELAY', 'ORIGIN']].groupby('ORIGIN').mean().std()


# In[56]:


pd.set_option('display.max_rows', 500)
df[['ARR_DELAY', 'ORIGIN']].groupby('ORIGIN').mean().sort_values(by='ARR_DELAY', ascending=False)


# In[57]:


fig, ax = plt.subplots()
df[['ARR_DELAY', 'ORIGIN']].groupby('ORIGIN').mean().plot.bar(figsize=(16,10), title='Mean delay by origin airport', ax=ax)
ax.legend(["Mean delay in minutes"])


# In[58]:


fig, ax = plt.subplots()
df[['ARR_DELAY', 'ORIGIN']].groupby('ORIGIN').mean().sort_values(by='ARR_DELAY', ascending=False).head(10).plot.bar(figsize=(16,10), title='Mean delay by origin airport', ax=ax)
ax.legend(["Mean delay in minutes"])


# ## Mean delay by destination airport

# In[59]:


df[['ARR_DELAY', 'DEST']].groupby('DEST').mean().std()


# In[60]:


df[['ARR_DELAY', 'DEST']].groupby('DEST').mean().sort_values(by='ARR_DELAY', ascending=False)


# In[61]:


df[['ARR_DELAY', 'ORIGIN', 'DAY_OF_WEEK']].groupby(['ORIGIN', 'DAY_OF_WEEK']).mean().sort_values(by='ARR_DELAY', ascending=False).head(10)


# ## Mean delay by day of week

# In[62]:


df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].astype(str)


# In[63]:


fig, ax = plt.subplots()
df[['ARR_DELAY', 'DAY_OF_WEEK']].groupby('DAY_OF_WEEK').mean().plot.bar(figsize=(16,10), title='Mean delay by day of week', ax=ax)
ax.legend(["Mean delay in minutes"])
plt.show()


# In[64]:


fig, ax = plt.subplots()
df[['DAY_OF_WEEK']].groupby('DAY_OF_WEEK').size().plot.bar(figsize=(16,10), title='Number of flights by day of week', ax=ax)
ax.legend(["Number of flights"])
plt.show()


# In[65]:


fig, ax = plt.subplots()
df[['ARR_DELAY', 'DAY_OF_WEEK']].groupby('DAY_OF_WEEK').std().plot.bar(figsize=(16,10), title='Standard deviation delay by day of week', ax=ax)
ax.legend(["Standard deviation delay in minutes"])


# ## Mean delay by month

# In[66]:


fig, ax = plt.subplots()
df[['ARR_DELAY', 'MONTH']].groupby('MONTH').mean().plot.bar(figsize=(16,10), title='Mean delay by month', ax=ax)
ax.legend(["Mean delay in minutes"])


# # Feature engineering

# ## Identification of features to keep for the model

# We will keep following features :  
#   
# ORIGIN                   1.000000 => Origin airport  
# CRS_DEP_TIME             1.000000 => we may only keep only the hour.  Maybe cut it into bins.  (compare result with and without binning)
# MONTH                    1.000000  
# DAY_OF_MONTH             1.000000  
# DAY_OF_WEEK              1.000000    
# UNIQUE_CARRIER           1.000000 => Flight company   
# DEST                     1.000000 => Destination airport  
# CRS_ARR_TIME             0.999999  
# DISTANCE                 0.999999   
# CRS_ELAPSED_TIME         0.999998 => carrier scheduled elapsed time  => redundant
# ARR_DELAY                0.985844  
# DEP_DELAY                0.988726  => we may also try to predict DEP_DELAY
# TAXI_OUT                 0.988374  
# For departing flights: the Actual taXi-Out Time is  
# the period between the Actual Off-Block Time and the Actual Take Off Time .  
# => Will be needed for second approach : prediction of arrival delay when we know of departure delay
# 
# 
# TAIL_NUM                 0.997738 => aircraft ID number printed on the tail  
#     => This feature would be very interesting.  Unfortunately, as a customer we do not know it until the last moment.  
#     => and as a carrier company, I guess it may be defined pretty late. So, including this information would be data leak.  
#     => However, we will keep this feature to predict arrival delays after plane take off
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
#       
# TAXI_IN                  0.987938  
# For arriving flights: the Actual taXi-In Time  is the  
# period between the Actual Landing Time and the Actual In-Block Time (  
#   
# => Not included (data leak / we don't know the information until the last moment)  
# 
# 
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
#  
# DEP_DEL15                0.988726  => redundant with DEP_DELAY
# DEP_DELAY_GROUP          0.988726  => redundant with DEP_DELAY
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
# 
# CANCELLED                0.999999 => Not kept for now.  May be useful to predict cancellations, but we predict delays.
# DIVERTED                 0.999999 => Not kept for now.  May be useful to predict cancellations, but we predict delays.

# In[67]:


df[['ARR_DEL15','TAIL_NUM']].groupby(['ARR_DEL15','TAIL_NUM'])


# ## Identification of features to add for the model

# ### Number of flights per origin airport per day

# In[68]:


# Create dictionary containing sum of flights for each tuple (date, origin airport)
dict_date_airport_nbflights = df[['FL_DATE', 'ORIGIN']].groupby(['FL_DATE', 'ORIGIN']).size().to_dict()


# In[69]:


def map_nbflights(date_val, airport_val):
    progbar.update(1)
    return(dict_date_airport_nbflights[(date_val, airport_val)])
    


# In[70]:


# Then for each row, we apply the dictionary and create a new column with its vales
progbar = tqdm(range(len(df)))
df['NBFLIGHTS_FORDAY_FORAIRPORT'] = df[['FL_DATE', 'ORIGIN']].apply(lambda row : map_nbflights(row.FL_DATE, row.ORIGIN), axis=1)


# ## Mean delay airport / day compared to mean flights airport / day

# In[71]:


fig, ax = plt.subplots()
df[df['ORIGIN'].isin(['ATL', 'ORD', 'DEN', 'LAX', 'DFW'])][['ORIGIN', 'NBFLIGHTS_FORDAY_FORAIRPORT']].groupby(['ORIGIN']).mean().sort_values(by='ORIGIN', ascending=False).plot.bar(figsize=(16,10), title='Mean number of flights per airport per day', ax=ax)
ax.legend(["Mean number of flights"])


# In[72]:


fig, ax = plt.subplots()
df[df['ORIGIN'].isin(['ATL', 'ORD', 'DEN', 'LAX', 'DFW'])][['ORIGIN', 'ARR_DELAY']].groupby(['ORIGIN']).mean().sort_values(by='ORIGIN', ascending=False).plot.bar(figsize=(16,10), title='Mean delay per origin airport', ax=ax)
ax.legend(["Mean delay per origin airport"])


# => No correlation between delay airport / day compared to mean flights airport / day

# ### Number of flights per origin airport per day and hour

# In[73]:


# Extract hour from scheduled departure time
df['HOUR_SCHEDULED'] = df['CRS_DEP_TIME'].str.slice(start=0,stop=2, step=1)


# In[74]:


df['HOUR_SCHEDULED'] = df['HOUR_SCHEDULED'].astype(str)


# In[75]:


# Create dictionary containing sum of flights for each tuple (date, origin airport)
dict_date_hour_airport_nbflights = df[['FL_DATE', 'HOUR_SCHEDULED', 'ORIGIN']].groupby(['FL_DATE', 'HOUR_SCHEDULED', 'ORIGIN']).size().to_dict()


# In[76]:


len(dict_date_hour_airport_nbflights)


# In[77]:


def map_date_hour_airport_to_nbflights(date_val, hour_val, airport_val):
    progbar.update(1)
    return(dict_date_hour_airport_nbflights[(date_val, hour_val, airport_val)])
    


# In[78]:


# Then for each row, we apply the dictionary and create a new column with its vales
progbar = tqdm(range(len(df)))
df['NBFLIGHTS_FORDAYHOUR_FORAIRPORT'] = df[['FL_DATE', 'HOUR_SCHEDULED', 'ORIGIN']].apply(lambda row : map_date_hour_airport_to_nbflights(row.FL_DATE, row.HOUR_SCHEDULED, row.ORIGIN), axis=1)


# In[79]:


fig, ax = plt.subplots()
df[df['ORIGIN'].isin(['ATL', 'ORD', 'DEN', 'LAX', 'DFW'])][['ORIGIN', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT']].groupby(['ORIGIN']).mean().sort_values(by='NBFLIGHTS_FORDAYHOUR_FORAIRPORT', ascending=False).plot.bar(figsize=(16,10), title='Mean number of flights per airport per day hour', ax=ax)
ax.legend(["Mean number of flights"])


# => No correlation between mear delay per origin airport and mean number of flights per airport per day hour

# In[123]:


fig, ax = plt.subplots(figsize=(16, 10))

plt.title('Delay by number of flights / hour in the airport')
plt.ylabel("Delay in minutes")
plt.xlabel("Number of flights / hour in the airport")

mean_delay = df['ARR_DELAY'].mean()

nbflights_80p = df.shape[0]*0.8
min_flights_per_hour_80p = df['NBFLIGHTS_FORDAYHOUR_FORAIRPORT'].sort_values(ascending=False).reset_index().loc[int(np.floor(nbflights_80p))]['NBFLIGHTS_FORDAYHOUR_FORAIRPORT']

plt.axhline(mean_delay, color='red', linestyle='--', label=f'Mean delay : ({mean_delay:.2f} minutes)')
plt.axvline(min_flights_per_hour_80p, color='green', linestyle='--', label=f"80% of flights have flights/hour > ({min_flights_per_hour_80p:.0f})")

ax.scatter(df['NBFLIGHTS_FORDAYHOUR_FORAIRPORT'], df['ARR_DELAY'], alpha=0.1)
plt.legend()


# ## Identification of quantitative and qualitative features

# In[80]:


df.columns[1]


# In[81]:


# Below are feature from dataset that we decided to keep: 
all_features = ['ORIGIN','CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','DEST','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME','ARR_DELAY','DEP_DELAY', 'TAXI_OUT', 'TAIL_NUM', 'NBFLIGHTS_FORDAY_FORAIRPORT', 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT']

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

# In[82]:


pd.set_option('display.max_rows', 100)
not_na = 100 - (df.isnull().sum() * 100 / len(df))
not_na_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_complete': not_na}).sort_values(by='percent_complete', ascending=False)
not_na_df


# In[83]:


df[df['DEP_TIME'].notnull() == False].sample(20)


# => We see that when flight is cancelled (value 1), we don't have actual delay values which is normal  
# => We may want to keep these values later, to be able to predict cancellations.  But for now, our model will not consider cancellations as delay.

# In[84]:


df[df['CANCELLED'] == 1].shape


# => Only 65973 cancelled flights on 5M total. Data seems very thin to make predictions.  
# => If we want to make cancellation predictions, we'll use another model dedicated to this task

# We also have nan values that correspond to DIVERTED flights :

# In[85]:


df[df['DEP_TIME'].notnull() == False].shape


# In[86]:


df[df['DIVERTED'] == 1].shape


# In[87]:


df[df['DIVERTED'] == 1]


# Let's check flights that have arrival delay null, but not cancelled  (cancelled flights do have null arrival delay : in that case it's normal)

# In[88]:


df[(df['ARR_DELAY'].notnull() == False) & (df['CANCELLED'] == 0)].shape


# In[89]:


df[(df['ARR_DELAY'].notnull() == False) & (df['CANCELLED'] == 0)][all_features].sample(10)


# => We see that DIVERTED == 1 for those lines : that's why we don't have delay information

# ### Display of qualitative features values :

# In[90]:


for feature_name in qualitative_features:
    print_column_information(df, feature_name)


# ### Display of quantiative features values :

# In[91]:


pd.set_option('display.max_rows', 10000)
for column_name in quantitative_features:
    #print(df[column_name].value_counts)
    print_column_information(df, column_name)


# ### Conversion of qualitative features into clean str format

# In[92]:


for feature_name in qualitative_features:
    df[feature_name] = df[feature_name].astype(str)


# ### Display qualitative features again

# In[93]:


pd.set_option('display.max_rows', 1000)
for feature_name in qualitative_features:
    print_column_information(df, feature_name)


# ### Remove delays caused by a delay from a previous flight 

# In[94]:


df.drop(index=df[df['LATE_AIRCRAFT_DELAY'] == 1].index, axis=0, inplace=True)


# ### Clean of DISTANCE

# In[95]:


df[df['DISTANCE'].notnull() == False][['DIVERTED', 'CANCELLED']]


# => Inconsistant values (NaN or 313.0)

# In[96]:


df.drop(index=df[df['DISTANCE'].notnull() == False].index, axis=0, inplace=True)


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

# ### Remove cancellations, divertions

# In[97]:


df.drop(index=df[ (df['CANCELLED'] == 1) | (df['DIVERTED'] == 1)].index, axis=0, inplace=True)


# In[98]:


df.reset_index(drop=True, inplace=True)


# #### Creating new dataframe with only the features we keep, to gain memory :

# In[99]:


df_new = df[all_features].copy(deep = True)


# In[100]:


del df


# In[101]:


df = df_new


# #### Clean ORIGIN

# In[102]:


df.drop(index=df[df['ORIGIN'].isin(['SPN', 'BFF', 'MHK', 'ENV', 'EFD', '4.00'])].index, axis=0, inplace=True)


# #### Clean DEST

# In[103]:


df.drop(index=df[df['DEST'].isin(['MHK', 'SPN', '1800-1859'])].index, axis=0, inplace=True)


# ### Clean TAXI OUT value

# In[104]:


df.loc[1701904]['TAXI_OUT']


# In[105]:


df.at[1701904, 'TAXI_OUT'] = '11.00'


# In[106]:


df.loc[1701904]['TAXI_OUT']


# In[107]:


df['TAXI_OUT'] = df['TAXI_OUT'].astype(float)


# In[108]:


df.loc[1701904]['TAXI_OUT']


# ### Display final data quality

# In[109]:


display_percent_complete(df)


# ### Display quantitative features distributions

# In[110]:


df.hist(bins=50, figsize=(20,15), log=True)


# In[111]:


df['ARR_DELAY'].hist(bins=[-100, -50, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, +100], figsize=(20,15))


# In[112]:


df[df['ARR_DELAY'] < 0].count()


# => A huge amount of flights arrive before schedule

# ## Correlation matrix of quantitative features

# In[113]:


corr_matrix = df.corr()


# In[114]:


corr_matrix[quantitative_features].loc[quantitative_features]


# In[115]:


plt.figure(figsize=(16, 10))
plt.title('Corrélation entre les valeurs numériques')
sns.heatmap(corr_matrix[quantitative_features].loc[quantitative_features], 
        xticklabels=corr_matrix[quantitative_features].loc[quantitative_features].columns,
        yticklabels=corr_matrix[quantitative_features].loc[quantitative_features].columns, cmap='coolwarm', center=0.20)


# In[117]:


df


# # Export cleaned data to CSV

# In[118]:


if not os.path.isdir(DATA_PATH_OUT):
    os.makedirs(DATA_PATH_OUT)

df.to_csv(DATA_PATH_FILE_OUTPUT, index=False)


# # Annexe : ancien code inutile

# #### Clean DEST_AIRPORT_ID

# df[df['DEST_AIRPORT_ID'].isin(['7.00', '13290', '14955', '-1'])]

# df.drop(index=df[df['DEST_AIRPORT_ID'].isin(['7.00', '13290', '14955', '-1'])].index, axis=0, inplace=True)

# pd.set_option('display.max_rows', 50)
# df_delays_groupby_tails = df[df['ARR_DEL15'] == 1][['ARR_DEL15','TAIL_NUM']].groupby(['ARR_DEL15','TAIL_NUM']).size().sort_values(ascending=False)
# df_delays_groupby_tails
