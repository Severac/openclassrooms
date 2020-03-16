#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ4 : transats dataset : modelisation notebook

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import zipfile
import urllib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import qgrid

import glob

from pandas.plotting import scatter_matrix

SAMPLED_DATA = False  # If True : data is sampled (1000 instances only) for faster testing purposes

DATA_PATH = os.path.join("datasets", "transats")
DATA_PATH = os.path.join(DATA_PATH, "out")

DATA_PATH_FILE_INPUT = os.path.join(DATA_PATH, "transats_metadata_transformed.csv")

plt.rcParams["figure.figsize"] = [16,9] # Taille par défaut des figures de matplotlib

import seaborn as sns
sns.set()

#import common_functions


# In[2]:


def qgrid_show(df):
    display(qgrid.show_grid(df, grid_options={'forceFitColumns': False, 'defaultColumnWidth': 170}))


# In[3]:


def print_column_information(df, column_name):
    column_type = df.dtypes[column_name]
    print(f'Column {column_name}, type {column_type}\n')
    print('--------------------------')

    print(df[[column_name]].groupby(column_name).size().sort_values(ascending=False))
    print(df[column_name].unique())    
    print('\n')


# In[4]:


def display_percent_complete(df):
    not_na = 100 - (df.isnull().sum() * 100 / len(df))
    not_na_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_complete': not_na}).sort_values(by='percent_complete', ascending=False)
    display(not_na_df)


# In[5]:


def identify_features(df, all_features):
    quantitative_features = []
    qualitative_features = []
    features_todrop = []

    for feature_name in all_features:
        if (df[feature_name].dtype == 'object'):
            qualitative_features.append(feature_name)

        else:
            quantitative_features.append(feature_name)

    print(f'Quantitative features : {quantitative_features} \n')
    print(f'Qualitative features : {qualitative_features} \n')  
    
    return quantitative_features, qualitative_features


# # Data load

# In[6]:


# hhmm timed features formatted
feats_hhmm = ['CRS_DEP_TIME',  'CRS_ARR_TIME']

df = pd.read_csv(DATA_PATH_FILE_INPUT, sep=',', header=0, encoding='utf-8', low_memory=False, parse_dates=feats_hhmm)


# In[7]:


df.shape


# In[8]:


display_percent_complete(df)


# In[9]:


'''
for column_name in df.columns:
    print_column_information(df, column_name)
    
'''


# # Identification of features

# In[10]:


# Below are feature from dataset that we decided to keep: 
all_features = ['ORIGIN','CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','DEST','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME','ARR_DELAY','DEP_DELAY', 'TAXI_OUT', 'TAIL_NUM']

model1_features = ['ORIGIN','CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','DEST','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME']
model1_label = 'ARR_DELAY'

quantitative_features = []
qualitative_features = []
features_todrop = []

for feature_name in all_features:
    if (df[feature_name].dtype == 'object'):
        qualitative_features.append(feature_name)
        
    else:
        quantitative_features.append(feature_name)

print(f'Quantitative features : {quantitative_features} \n')
print(f'Qualitative features : {qualitative_features} \n')        
        

#Commented out : no drop of features
#for df_column in df.columns:
#    if df_column not in all_features:
#        features_todrop.append(df_column)
#        
#print(f'Features to drop : {features_todrop} \n')


# # Split train set, test set

# In[11]:


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train = df_train.copy()
df_test = df_test.copy()


# In[12]:


if (SAMPLED_DATA == True):
    df_train = df_train.sample(1000).copy(deep=True)
    df = df.loc[df_train.index]


# In[13]:


df_train


# In[14]:


#df = df.loc[df_train.index]


# # Features encoding

# In[15]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn import decomposition
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import statistics


'''
Cette fonction fait un 1 hot encoding des features qui sont des catégories
Old function not used anymore
'''
    
def add_categorical_features_1hot(df, categorical_features_totransform):
    #df.drop(labels=categorical_features_totransform, axis=1, inplace=True)
    
    
    #df_encoded = pd.get_dummies(df, columns=categorical_features_totransform, sparse=True)
    
    for feature_totransform in categorical_features_totransform:
        print(f'Adding 1hot Feature : {feature_totransform}')
        
        print('First')
        df_transformed = df[feature_totransform].str.get_dummies().add_prefix(feature_totransform +'_')   
        
        #df_new = pd.get_dummies(df, columns=['ORIGIN'])
        
        
        
        
        #df.drop(labels=feature_totransform, axis=1, inplace=True)
        print('Second')
        del df[feature_totransform]
        
        print('Third')
        df = pd.concat([df, df_transformed], axis=1)
        
    return(df)


class HHMM_to_Minutes(BaseEstimator, TransformerMixin):
    def __init__(self, features_toconvert = ['CRS_DEP_TIME', 'CRS_ARR_TIME']):
        self.features_toconvert = features_toconvert
        return None
    
    def fit(self, df):      
        return self
    
    def transform(self, df):       
        for feature_toconvert in self.features_toconvert:
            print(f'Converting feature {feature_toconvert}\n')
            #print('1\n')
            df_concat = pd.concat([df[feature_toconvert].str.slice(start=0,stop=2, step=1),df[feature_toconvert].str.slice(start=2,stop=4, step=1)], axis=1).astype(int).copy(deep=True)
                    
            #print('2\n')
            df[feature_toconvert] = (df_concat.iloc[:, [0]] * 60 + df_concat.iloc[:, [1]])[feature_toconvert]
            
            #print('3\n')
        
        return(df)

        
class CategoricalFeatures1HotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features_totransform=['ORIGIN', 'UNIQUE_CARRIER', 'DEST']):
        self.categorical_features_totransform = categorical_features_totransform
    
    def fit(self, df, labels=None):      
        return self
    
    def transform(self, df):
        df_encoded = pd.get_dummies(df, columns=self.categorical_features_totransform, sparse=True)  # Sparse allows to gain memory. But then, standardscale must be with_mean=False
        #df_encoded = pd.get_dummies(df, columns=self.categorical_features_totransform, sparse=False)
        print('type of df : ' + str(type(df_encoded)))
        return(df_encoded)

class FeaturesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features_toselect = None):  # If None : every column is kept, nothing is done
        self.features_toselect = features_toselect
    
    def fit(self, df, labels=None):      
        return self
    
    def transform(self, df):       
        if (self.features_toselect != None):
            filter_cols = [col for col in df if (col.startswith(tuple(self.features_toselect)))]
            return(df[filter_cols])    

        else:
            return(df)
  
conversion_pipeline = Pipeline([
    ('data_converter', HHMM_to_Minutes()),
    #('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()),
    #('standardscaler', preprocessing.StandardScaler()),
])

preparation_pipeline = Pipeline([
    ('data_converter', HHMM_to_Minutes()),
    ('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()),
    #('standardscaler', preprocessing.StandardScaler()),
])

preparation_pipeline2 = Pipeline([
    ('data_converter', HHMM_to_Minutes()),
    ('multiple_encoder', ColumnTransformer([
        ('categoricalfeatures_1hotencoder', OneHotEncoder(), ['ORIGIN', 'UNIQUE_CARRIER', 'DEST'])
    ], remainder='passthrough')),
    #('standardscaler', preprocessing.StandardScaler()),
])

# If matrix is sparse, with_mean=False must be passed to StandardScaler
prediction_pipeline = Pipeline([
    ('features_selector', FeaturesSelector(features_toselect=['ORIGIN','CRS_DEP_TIME','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','UNIQUE_CARRIER','DEST','CRS_ARR_TIME','DISTANCE','CRS_ELAPSED_TIME'])),
    #('standardscaler', ColumnTransformer([
    #    ('standardscaler_specific', StandardScaler(), ['CRS_DEP_TIME','MONTH','DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_ARR_TIME', 'DISTANCE', 'CRS_ELAPSED_TIME'])
    #], remainder='passthrough')),
    #('predictor', To_Complete(predictor_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
])


'''
ColumnTransformer([
        ('standardscaler_specific', StandardScaler(), ['MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DISTANCE', 'CRS_ELAPSED_TIME', 'ARR_DELAY', 'DEP_DELAY', 'TAXI_OUT'])
    ], remainder='passthrough')
'''


# # For later : select qualitative airport features that represent 90% of data

# In[37]:


((((df[['ORIGIN']].groupby('ORIGIN').size() / len(df)).sort_values(ascending=False)) * 100).cumsum() < 95)


# In[57]:


# Get ORIGIN values that represent 90% of data
ORIGIN_high_percentile = ((((df[['ORIGIN']].groupby('ORIGIN').size() / len(df)).sort_values(ascending=False)) * 100).cumsum() < 90).where(lambda x : x == True).dropna().index.values.tolist()
ORIGIN_low_percentile = ((((df[['ORIGIN']].groupby('ORIGIN').size() / len(df)).sort_values(ascending=False)) * 100).cumsum() >= 90).where(lambda x : x == True).dropna().index.values.tolist()


# In[66]:


ORIGIN_total = len(df['ORIGIN'].unique())
ORIGIN_high_percentile_sum = len(ORIGIN_high_percentile)
ORIGIN_low_percentile_sum = len(ORIGIN_low_percentile)
high_low_sum = ORIGIN_high_percentile_sum + ORIGIN_low_percentile_sum

print(f'Total number of ORIGIN values : {ORIGIN_total}')
print(f'Number of ORIGIN high percentile values : {ORIGIN_high_percentile_sum}')
print(f'Number of ORIGIN low percentile values : {ORIGIN_low_percentile_sum}')
print(f'Sum of high percentile + low percentile values : {high_low_sum}')


# In[16]:


df_transformed = preparation_pipeline.fit_transform(df_train)


# In[17]:


df_transformed.shape


# In[18]:


type(df_transformed)


# In[19]:


df_transformed.info()


# In[20]:


df_transformed = prediction_pipeline.fit_transform(df_transformed)


# In[21]:


df_transformed.shape


# In[22]:


df_transformed


# In[23]:


df_transformed.info()


# In[ ]:


pd.set_option('display.max_columns', 400)


# In[ ]:


quantitative_features, qualitative_features = identify_features(df, all_features)


# # Basic linear regression

# In[22]:


df_transformed.shape


# In[25]:


df_train[model1_label].shape


# In[23]:


df.shape


# In[26]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(df_transformed, df_train[model1_label])


# In[29]:


df_transformed


# In[27]:


from sklearn import linear_model

regressor = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
regressor.fit(df_transformed, df_train[model1_label])


# In[ ]:


from sklearn.metrics import mean_squared_error

df_predictions = lin_reg.predict(df_transformed)
lin_mse = mean_squared_error(df[model1_label], df_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:





# In[ ]:




