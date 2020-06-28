#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:45:19 2020

@author: francois
"""
DEBUG_LEVEL = 0  # 1 = Main steps

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn import decomposition
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

from sklearn.neighbors import NearestNeighbors
#from sklearn.neighbors import KDTree

import statistics

from scipy import sparse

import pandas as pd

#import qgrid

import numpy as np

import pickle


RECOMPUTE_GRIDSEARCH = True  # CAUTION : computation is several hours long
SAVE_GRID_RESULTS = True # If True : grid results object will be saved to pickle files that have GRIDSEARCH_FILE_PREFIX
LOAD_GRID_RESULTS = False # If True : grid results object will be loaded from pickle files that have GRIDSEARCH_FILE_PREFIX


#GRIDSEARCH_CSV_FILE = 'grid_search_results.csv'

GRIDSEARCH_FILE_PREFIX = 'grid_search_results_'

### Doc2vec settings

#DOC2VEC_TRAINING_SAVE_FILE = 'doc2vec_model'

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.parsing.preprocessing import remove_stopwords

import time

from gensim.test.utils import get_tmpfile

import gensim


#model.save(fname)
#model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

from tqdm import tqdm_notebook as tqdm

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


def qgrid_show(df):
    display(qgrid.show_grid(df, grid_options={'forceFitColumns': False, 'defaultColumnWidth': 170}))

def display_percent_complete(df):
    not_na = 100 - (df.isnull().sum() * 100 / len(df))
    not_na_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_complete': not_na}).sort_values(by='percent_complete', ascending=False)
    display(not_na_df)
    

def print_column_information(df, column_name):
    column_type = df.dtypes[column_name]
    print(f'Column {column_name}, type {column_type}\n')
    print('--------------------------')

    print(df[[column_name]].groupby(column_name).size().sort_values(ascending=False))
    print(df[column_name].unique())    
    print('\n')

    
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
    
    '''
    def transform(self, df):
        return(df)
    '''
    
    def transform(self, df):      
        for feature_toconvert in self.features_toconvert:
            print(f'Converting feature {feature_toconvert}\n')
            #print('1\n')
            df_concat = pd.concat([df[feature_toconvert].str.slice(start=0,stop=2, step=1),df[feature_toconvert].str.slice(start=2,stop=4, step=1)], axis=1).astype(int).copy(deep=True)
                    
            #print('2\n')
            df[feature_toconvert] = (df_concat.iloc[:, [0]] * 60 + df_concat.iloc[:, [1]])[feature_toconvert]
            del df_concat
            
            #print('3\n')
        
        return(df)
        
    
class HHMM_to_HH(BaseEstimator, TransformerMixin):
    def __init__(self, features_toconvert = ['CRS_DEP_TIME', 'CRS_ARR_TIME']):
        self.features_toconvert = features_toconvert
        return None
    
    def fit(self, df):      
        return self
    
    def transform(self, df):       
        for feature_toconvert in self.features_toconvert:
            print(f'Converting feature {feature_toconvert}\n')
            #print('1\n')

            df.loc[:, feature_toconvert] = df[feature_toconvert].str.slice(start=0,stop=2, step=1)
        
        return(df)
    
'''
class CategoricalFeatures1HotEncoder_old(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features_totransform=['ORIGIN', 'UNIQUE_CARRIER', 'DEST']):
        self.categorical_features_totransform = categorical_features_totransform
    
    def fit(self, df, labels=None):      
        return self
    
    def transform(self, df):
        # /!\ Array will not have the same shape if we fit an ensemble of samples that have less values than total dataset
        df_encoded = pd.get_dummies(df, columns=self.categorical_features_totransform, sparse=True)  # Sparse allows to gain memory. But then, standardscale must be with_mean=False
        #df_encoded = pd.get_dummies(df, columns=self.categorical_features_totransform, sparse=False)
        print('type of df : ' + str(type(df_encoded)))
        return(df_encoded)
'''

class CategoricalFeatures1HotEncoder(BaseEstimator, TransformerMixin):
    #def __init__(self, categorical_features_totransform=['ORIGIN', 'UNIQUE_CARRIER', 'DEST']):
    def __init__(self):
        #self.categorical_features_totransform = categorical_features_totransform
        self.fitted = False
        self.all_feature_values = {}
        #self.df_encoded = None
    
    #def fit(self, df, labels=None):      
    def fit(self, df, categorical_features_totransform=['ORIGIN', 'UNIQUE_CARRIER', 'DEST']):      
        print('Fit data')
        self.categorical_features_totransform = categorical_features_totransform
        #print('!! categorical_features_totransform' + str(self.categorical_features_totransform))

        if (self.categorical_features_totransform != None):
            for feature_name in self.categorical_features_totransform:
                df[feature_name] = df[feature_name].astype(str) # Convert features to str in case they are not already     
                self.all_feature_values[feature_name] = feature_name + '_' + df[feature_name].unique()
        
        self.fitted = True
        
        return self
    
    def transform(self, df):
        if (self.fitted == False):
            self.fit(df)
        
        if (self.categorical_features_totransform != None):
            print('Transform data')
            
            #print('1hot encode categorical features...')
            #df_encoded = pd.get_dummies(df, columns=self.categorical_features_totransform, sparse=True)  # Sparse allows to gain memory. But then, standardscale must be with_mean=False
            df_encoded = pd.get_dummies(df, columns=self.categorical_features_totransform, sparse=False)

            # Get category values that were in fitted data, but that are not in data to transform 
            for feature_name, feature_values in self.all_feature_values.items():
                diff_columns = list(set(feature_values) - set(df_encoded.columns.tolist()))
                #print(f'Column values that were in fitted data but not in current data: {diff_columns}')

                if (len(diff_columns) > 0):
                    #print('Adding those column with 0 values to the DataFrme...')
                    # Create columns with 0 for the above categories, in order to preserve same matrix shape between train et test set
                    zeros_dict = dict.fromkeys(diff_columns, 0)
                    df_encoded = df_encoded.assign(**zeros_dict)

            #print('type of df : ' + str(type(df_encoded)))
            return(df_encoded)

        else:
            return(df)

class Aggregate_then_GroupByMean_then_Sort_numericalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features_totransform=['ORIGIN', 'UNIQUE_CARRIER', 'DEST']):
        self.categorical_features_totransform = categorical_features_totransform
        self.fitted = False
        self.all_feature_values = {}
    
    def fit(self, df, labels=None):      
        #print('Fit data')
        
        self.feature_maps = {}
        
        for feature_name in self.categorical_features_totransform:
            #print(f'Fitting feature {feature_name}')
            # List all feature values ordered by mean delay
            list_feature_mean_ordered = df[['ARR_DELAY', feature_name]].groupby(feature_name).mean().sort_values(by='ARR_DELAY', ascending=True).index.tolist()
            
            # Generate a dictionary of feature values as keys and index as values
            self.feature_maps[feature_name] = {}
            self.feature_maps[feature_name]['list_feature_mean_ordered_dict'] = {list_feature_mean_ordered[i] : i for i in range(len(list_feature_mean_ordered))  }
            
            #print('Dictionnary : ' + str(self.feature_maps[feature_name]['list_feature_mean_ordered_dict']))
            
            # BUG : we had to do that line of code in transform instead, not in fit (result is the same, no difference)
            #self.feature_maps[feature_name]['list_feature_mean_ordered_mapper'] = lambda k : self.feature_maps[feature_name]['list_feature_mean_ordered_dict'][k]
                  
        self.fitted = True
        
        return self
    
    def transform(self, df):
        if (self.fitted == False):
            print('Launching fit first (if you see this message : ensure that you have passed training set as input, not test set)')
            self.fit(df)
        
        #print('Encode categorical features...')
        
        for feature_name in self.categorical_features_totransform:
            #print(f'Encoding feature {feature_name} ...')
            #print('Dictionnary : ' + str(self.feature_maps[feature_name]['list_feature_mean_ordered_dict']))
            
            # Replace each feature value by its index (the lowest the index, the lowest the mean delay is for this feature)
            list_feature_mean_ordered_mapper = lambda k : self.feature_maps[feature_name]['list_feature_mean_ordered_dict'][k]
                            
            #df[feature_name] = df.loc[:, feature_name].apply(self.feature_maps[feature_name]['list_feature_mean_ordered_mapper'])  # BUG (we had to use line below instead)
            df[feature_name] = df.loc[:, feature_name].apply(list_feature_mean_ordered_mapper)

        return(df)
    
    
class FeaturesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features_toselect = None):  # If None : every column is kept, nothing is done
        self.features_toselect = features_toselect
    
    def fit(self, df, labels=None):      
        if (DEBUG_LEVEL >= 1) :
            print('Fit FeaturesSelector')
        return self
    
    def transform(self, df):       
        if (DEBUG_LEVEL >= 1) :
            print('Transform FeaturesSelector')
        
        if (self.features_toselect != None):
            filter_cols = [col for col in df if (col.startswith(tuple(self.features_toselect)))]
            
            filter_cols.sort()
            
            #print("Features selected (in order): " + str(df.loc[:, filter_cols].columns))
            
            #df = df.loc[:, filter_cols]
            return(df.loc[:, filter_cols])

        else:
            return(df)

'''
In order have less features globally: we Keep only features_tofilter that represent percent_tokeep% of total values
Features which values represent less than percent_tokeep% will be set "OTHERS" value instead of their real value
'''

class Filter_High_Percentile(BaseEstimator, TransformerMixin):
    def __init__(self, features_tofilter = ['ORIGIN', 'DEST'], percent_tokeep = 80):
        self.features_tofilter = features_tofilter
        self.percent_tokeep = percent_tokeep
        self.high_percentile = None
        self.low_percentile = None
    
    def fit(self, df, labels=None): 
        if (DEBUG_LEVEL >= 1) :
            print('Fit high percentile filter...')
            
        for feature_tofilter in self.features_tofilter:
            # Get feature_tofilter values that represent 80% of data
            self.high_percentile = ((((df[[feature_tofilter]].groupby(feature_tofilter).size() / len(df)).sort_values(ascending=False)) * 100).cumsum() < self.percent_tokeep).where(lambda x : x == True).dropna().index.values.tolist()
            self.low_percentile = ((((df[[feature_tofilter]].groupby(feature_tofilter).size() / len(df)).sort_values(ascending=False)) * 100).cumsum() >= self.percent_tokeep).where(lambda x : x == True).dropna().index.values.tolist()

            total = len(df[feature_tofilter].unique())
            high_percentile_sum = len(self.high_percentile)
            low_percentile_sum = len(self.low_percentile)
            high_low_sum = high_percentile_sum + low_percentile_sum

            print(f'Total number of {feature_tofilter} values : {total}')
            print(f'Number of {feature_tofilter} high percentile (> {self.percent_tokeep}%) values : {high_percentile_sum}')
            print(f'Number of {feature_tofilter} low percentile values : {low_percentile_sum}')
            print(f'Sum of high percentile + low percentile values : {high_low_sum}')
        
        print('End of high percentile filter fit')
        return self
    
    def transform(self, df):       
        if (self.features_tofilter != None):
            print('Apply high percentile filter...')
            
            for feature_tofilter in self.features_tofilter:
                print(f'Apply filter on feature {feature_tofilter}')
                # To do for later : apply low_percentile specific to the feature, and not only last calculated low_percentile  (in our case it's the same percentile for ORIGIN and DEST so this is not a problem)
                df.loc[df[feature_tofilter].isin(self.low_percentile), feature_tofilter] = 'OTHERS'   
            
            return(df)    

        else:
            return(df)
        

    
class DenseToSparseConverter(BaseEstimator, TransformerMixin):
    def __init__(self):  # If None : every column is kept, nothing is done
        return None
    
    def fit(self, matrix, labels=None):      
        return self
    
    def transform(self, matrix):   
        return(sparse.csr_matrix(matrix))

    
'''
This class adds polynomial features in univariate way  (if feature X and n_degree 3 :  then it will add X², X³, and an intercept at the end)

Requires ndarray as input
'''    
class PolynomialFeaturesUnivariateAdder(BaseEstimator, TransformerMixin):
    def __init__(self, n_degrees=2):
        self.n_degrees = n_degrees
        self.fitted = False
    
    def fit(self, df, labels=None):
        self.fitted = True
        return self
    
    def transform(self, df):
        if (self.fitted == False):
            self.fit(df)

        nb_instances, n_features = df.shape
        df_poly = np.empty((nb_instances, 0)) # Create empty array of nb_instances line and 0 features yet (we'll concatenate polynomial features to it)

        progbar = tqdm(range(n_features))
        print('Adding polynomial features')

        for feature_index in range(n_features):    
            df_1feature = df[:,feature_index]  # Reshape 

            for n_degree in range(self.n_degrees):
                df_poly = np.c_[df_poly, np.power(df_1feature, n_degree + 1)]

            progbar.update(1)

        # Add bias (intercept)
        df_poly = np.c_[df_poly, np.ones((len(df_poly), 1))]  # add x0 = 1 feature        
        
        return(df_poly)

'''
This class adds polynomial features in univariate way  (if feature X and n_degree 3 :  then it will add X², X³, and an intercept at the end)

Requires DataFrame as input
'''        
    
class PolynomialFeaturesUnivariateAdder_DataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, n_degrees=2):
        self.n_degrees = n_degrees
        self.fitted = False
    
    def fit(self, df, features_toadd=None):  
        print('fit')
        self.features_toadd = features_toadd
        print('Features to add :')
        print(self.features_toadd)
        self.fitted = True
        return self
    
    def transform(self, df):
        print('transform')
        if (self.fitted == False):
            self.fit(df)

        nb_instances, n_features = df.shape
        #df_poly = np.empty((nb_instances, 0)) # Create empty array of nb_instances line and 0 features yet (we'll concatenate polynomial features to it)
        #df_poly = pd.DataFrame(index=df.index,columns=None)
        
        progbar = tqdm(range(len(self.features_toadd)))
        print('Adding polynomial features')

        for column_name in self.features_toadd:    
            #df_1feature = df.loc[:, column_name]

            for n_degree in range(1, self.n_degrees):
                #df = pd.concat([df, np.power(df.loc[:, column_name], n_degree + 1)], axis=1)
                df = pd.concat([df, np.power(df[[column_name]], n_degree + 1).rename(columns={column_name : column_name+'_DEG'+str(n_degree+1)})], axis=1)

            progbar.update(1)

        # Add bias (intercept)
        #df_poly = pd.concat([df_poly, np.ones((len(df_poly), 1))])  # add x0 = 1 feature        
        #del df
        return(df)    
    
    
class StandardScalerMultiple_old(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fitted = False
    
    def fit(self, df, columns=None):              
        self.columns = columns
        self.scaler = StandardScaler()
  
        if (self.columns == None):
            return(df)
        else:
            self.scaler.fit(df[self.columns].to_numpy())            
        
        return self
    
    def transform(self, df):
        if (self.fitted == False):
            self.fit(df)
        
        if (self.columns == None):
            return(df)
        
        else:
            df[self.columns] = self.scaler.transform(df[self.columns].to_numpy())

        return(df)
  
'''
This class does StandardScale scaling 
It is "Multiple" because you can select which features to scale.
Other columns are kept untouched.

In and out data : pandas DataFrame
'''      
class StandardScalerMultiple(BaseEstimator, TransformerMixin):
    def __init__(self, features_toscale=None):
        self.fitted = False
        self.columns = features_toscale
    
    def fit(self, df):              
        print('Fit Std scale multiple')
        self.scaler = StandardScaler()
  
        if (self.columns == None):
            self.fitted = True
            return(df)
        else:
            self.scaler.fit(df[self.columns].to_numpy())            
            self.fitted = True
        
        return self
    
    def transform(self, df):
        print('Transform Std scale multiple')
        if (self.fitted == False):
            self.fit(df)
        
        if (self.columns == None):
            return(df)
        
        else:
            df.loc[:, self.columns] = self.scaler.transform(df.loc[:, self.columns].to_numpy())

        return(df)


'''
This class does MinMax scaling 
It is "Multiple" because you can select which features to scale.
Other columns are kept untouched.

In and out data : pandas DataFrame
'''
class MinMaxScalerMultiple(BaseEstimator, TransformerMixin):
    #def __init__(self, features_toscale='ALL_FEATURES'):        
    def __init__(self, features_toscale):        
        if (DEBUG_LEVEL >= 1) :
            print('Init MinMaxScalerMultiple')
        #print(f'At init : features_toscale == {features_toscale}')
        self.fitted = False
        self.columns = features_toscale
        # This line is mandatory for gridsearch to take features_toscale into account :
        self.features_toscale = features_toscale 
    
    def fit(self, df, labels=None):              
        if (DEBUG_LEVEL >= 1) :
            print('Fit Min max scaler multiple')
            
        self.scaler = MinMaxScaler()
  
        if (self.columns == None):
            #print('self.columns == None')
            self.fitted = True
            return(df)
            
        else:
            #print('self.columns not None')
            if (self.columns == 'ALL_FEATURES'):
                self.columns = df.columns.tolist()
                #print(f'MinMaxScalerMultiple : self.columns  = {self.columns}')
            
            #print('1')
            self.scaler.fit(df[self.columns].to_numpy())            
            self.fitted = True
            #print('2')
        
        return self
    
    def transform(self, df):
        if (DEBUG_LEVEL >= 1) :
            print('Transform Min max scaler multiple')
        if (self.fitted == False):
            self.fit(df)
        
        if (self.columns == None):
            return(df)
        
        else:
            df.loc[:, self.columns] = self.scaler.transform(df.loc[:, self.columns].to_numpy())

        return(df)        

'''
This class does Logarithm scaling 
It is "Multiple" because you can select which features to scale.
Other columns are kept untouched.

In and out data : pandas DataFrame
'''
        
class LogScalerMultiple(BaseEstimator, TransformerMixin):
    def __init__(self, features_toscale=None):
        self.fitted = False
        self.columns = features_toscale
    
    def fit(self, df):              
        if (DEBUG_LEVEL >= 1) :
            print('Fit log scaler multiple')
        self.scaler = FunctionTransformer(np.log1p, validate=True)
  
        if (self.columns == None):
            self.fitted = True
            return(df)
            
        else:
            self.scaler.fit(df[self.columns].to_numpy())            
            self.fitted = True
        
        return self
    
    def transform(self, df):
        if (DEBUG_LEVEL >= 1) :
            print('Transform log scaler scaler multiple')
        if (self.fitted == False):
            self.fit(df)
        
        if (self.columns == None):
            return(df)
        
        else:
            df.loc[:, self.columns] = self.scaler.transform(df.loc[:, self.columns].to_numpy())

        return(df)    

'''
This class is a wrapper for dimensionality reduction.
Different dimensionality reduction algorithms can be selected, along with hyperparameters

In and out data : pandas DataFrame
'''
class DimensionalityReductor(BaseEstimator, TransformerMixin):
    def __init__(self, features_totransform=None, algorithm_to_use='PCA', n_dim=20, labels_featurename=None, n_neighbors=10):
        # Passing labels_featurename here for NCA is a mistake :(  it should be passed as a label to fit function, for
        # gridsearch to correctly split labels associated with folds.
        # So, I kept labels_featurenamesfor backwards compatibility with the rest of the notebook.
        # But with GridSearchCV, labels_featurename won't be used :  labels passed to fit will be used instead
        
        # labels_featurename can be a feature name and also a list of discrete labels               
        self.fitted = False
        self.features_totransform = features_totransform
        self.algorithm_to_use = algorithm_to_use
        self.n_dim = n_dim
        self.labels_featurename = labels_featurename # For NCA
        self.n_neighbors = n_neighbors # For LLE
    
    def fit(self, df, labels=None):  # Labels=None added to attempt to remove weird error fit() takes 2 positional arguments but 3 were given
        if (DEBUG_LEVEL >= 1) :
            print('Fit Dimensionality Reductor')
              
        if (self.features_totransform == None):
            self.fitted = True
            return(self)
            
        else:            
            if (self.algorithm_to_use == 'PCA'):
                self.reductor = PCA(n_components=self.n_dim, random_state=42)                
            
            if (self.algorithm_to_use == 'NCA'):
                '''
                if (self.labels == None):
                    self.labels = df['TotalPricePerMonth']
                '''
                
                if (self.labels_featurename != None):
                    if isinstance(self.labels_featurename, str):
                        self.labels_discrete = pd.cut(df[self.labels_featurename], bins=10, right=True).astype(str).tolist()
                        
                    else:
                        #self.labels_discrete = self.labels_featurename
                        self.labels_discrete = pd.cut(self.labels_featurename, bins=10, right=True).astype(str).tolist()
                
                else:  # For use with GridSearchCV : in this case, labels variable has been passed to fit()                    
                    #print('df indexes :')
                    #print(df.index)
                        
                    labels = labels[df.index]  # Select only labels in df (which is a fold of the training set)
                    #print('labels sliced successfuly')
                    #print('Printing labels sliced :')
                    #print(labels)
                    
                    labels_discrete = pd.cut(labels, bins=10, right=True).astype(str).tolist()
                    #print('labels discretized successfuly')
                    #print(labels_discrete)
                    
                
                
                self.reductor = NeighborhoodComponentsAnalysis(random_state=42, n_components=self.n_dim)
                
            
            if (self.algorithm_to_use == 'TSNE'):
                self.reductor = TSNE(n_components=self.n_dim, random_state=42)

            if (self.algorithm_to_use == 'LLE'):
                self.reductor = LocallyLinearEmbedding(n_components=self.n_dim, n_neighbors=self.n_neighbors, random_state=42)

            if (self.algorithm_to_use == 'ISOMAP'):
                self.reductor = Isomap(n_components=self.n_dim)
            
            
            if (self.features_totransform == 'ALL'):
                self.filter_cols = [col for col in df]
            
            else:
                self.filter_cols = [col for col in df if (col.startswith(tuple(self.features_totransform)))]            

            self.filter_cols.sort(key=lambda v: (isinstance(v, str), v))
            
            #print('1')
            #print("Features selected (in order): " + str(df.loc[:, self.filter_cols].columns))            
            #print('2')
            
            if ((self.algorithm_to_use == 'PCA') or (self.algorithm_to_use == 'LLE') or (self.algorithm_to_use == 'ISOMAP')):
                self.reductor.fit(df[self.filter_cols].to_numpy())            
            
            if (self.algorithm_to_use == 'NCA'):
                if (self.labels_featurename != None):  # Backwards compatibility. For use without GridSearchCV
                    self.reductor.fit(df[self.filter_cols].to_numpy(), self.labels_discrete)       
                    
                else:  # For use with GridSearchCV : labels passed to fit
                    #print('self.labels_featurename != None : for gridsearch')
                    
                    #print('len of labels: ' + str(len(labels)))
                    #print('len of labels_discrete: ' + str(len(labels_discrete)))
                    #print('len of input df to transform : ' + str(len(df[self.filter_cols])))    
                                       
                    #print('unique labels : ')
                    #print(np.unique(labels_discrete))
                    
                    self.reductor.fit(df[self.filter_cols].to_numpy(), labels_discrete)       
                    #print('reductor fit called')
            
            if  (self.algorithm_to_use == 'TSNE'):
                print('No fit for TSNE')
            
            self.fitted = True
        
        return self
    
    def transform(self, df):
        if (DEBUG_LEVEL >= 1) :
            print('Transform Dimensionality Reductor')
            
        if (self.fitted == False):
            self.fit(df)
        
        if (self.features_totransform == None):
            return(df)
        
        else:
            remaining_columns = list(set(df.columns.tolist()) - set(self.filter_cols))
            
            #print(f'Remaining columns: {remaining_columns}')
            
            if (self.algorithm_to_use == 'TSNE'):
                # TNSE has not transform function so we have to do fit_transform directly
                np_transformed = self.reductor.fit_transform(df[self.filter_cols].to_numpy())            
            
            else:
                #print('1')
                np_transformed = self.reductor.transform(df[self.filter_cols].to_numpy())
                #print('2')
            
            if (remaining_columns != []):
                #print('3')
                #print('df.index before concatenation: ')
                #print(df.index)
                df_transformed = pd.concat([df[remaining_columns].reset_index(drop=True), pd.DataFrame(np_transformed)], axis=1)
                #print('Print 1 line of Df after concat and before reindex :')
                #print(df_transformed.head(1))
                
                # Added to be able to extract index (customer IDs) for labels after
                # Needs to be tested again with the notebook
                df_transformed.set_index(df.index, inplace=True) 
                
                #print('df_transformed.index after concatenation: ')
                #print(df_transformed.index)
                #print('Print 1 line of Df after concat and after reindex :') # => result : nan values, not good
                #print(df_transformed.head(1))
                #print('4')
                
            else:
                df_transformed = pd.DataFrame(np_transformed)

        #print('5')
        return(df_transformed)    

 

def load_data(in_file):
    df = pd.read_csv(in_file, encoding='utf-8', converters={'InvoiceNo': str, 'StockCode':str, 'Description': str, \
                                       'CustomerID':str, 'Country': str, 'DescriptionNormalized': str})   
        
    return(df)
    
'''
This function splits training and test set with a stratify strategy on 1 feature
'''    
def custom_train_test_split_sample(df, split_feature, SAMPLED_DATA=False):
    from sklearn.model_selection import train_test_split
    
    if (SAMPLED_DATA == True):
        df_labels_discrete = pd.cut(df[split_feature], bins=50)
        df, df2 = train_test_split(df, train_size=NB_SAMPLES, random_state=42, shuffle = True, stratify = df_labels_discrete)

    if (split_feature != None):
        #df_labels_discrete = pd.cut(df[split_feature], bins=50)
        df_labels_discrete = pd.qcut(df[split_feature], 10)

        df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, shuffle = True, stratify = df_labels_discrete)
        #df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

    else:
        df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

    '''
    convert_dict = {'InvoiceNo': str, 'StockCode':str, 'Description': str, \
                                       'CustomerID':str, 'Country': str, 'DescriptionNormalized': str}
  
    df_train = df_train.astype(convert_dict) 
    df_test = df_test.astype(convert_dict)
    '''

    '''
    df_train = df_train.copy()
    df_test = df_test.copy()
    '''
    
    return df, df_train, df_test

'''
This class does a bag of words encoding : which means for each possible word in
the feature to encode, a column called featurename_word is created in the dataframe.
#
You can select which features to encode (others are left untouched)
'''
class BowEncoder(BaseEstimator, TransformerMixin):
    #def __init__(self, categorical_features_totransform=['ORIGIN', 'UNIQUE_CARRIER', 'DEST']):
    def __init__(self, min_df=0.001):
        #self.categorical_features_totransform = categorical_features_totransform
        self.fitted = False
        self.all_feature_values = {}
        self.min_df = min_df
        self.vectorizers = {}
        #self.df_encoded = None
    
    #def fit(self, df, labels=None):      
    def fit(self, df, labels=None, categorical_features_totransform=['DescriptionNormalized']):      
        if (DEBUG_LEVEL >= 1) :
            print('BowEncoder : Fit data')
            
        #print(f'categorical_features_totransform == {categorical_features_totransform}')
        self.categorical_features_totransform = categorical_features_totransform
        #print('!! categorical_features_totransform' + str(self.categorical_features_totransform))

        if (self.categorical_features_totransform != None):
            for feature_name in self.categorical_features_totransform:
                self.vectorizers[feature_name] = CountVectorizer(min_df=self.min_df)
                #print('track1')
                matrix_vectorized = self.vectorizers[feature_name].fit(df[feature_name].astype(str))
                #print('track2')
                                
        self.fitted = True
        
        return self
    
    def transform(self, df):
        if (DEBUG_LEVEL >= 1) :
            print('BowEncoder : transform data')
            
        if (self.fitted == False):
            self.fit(df)
        
        if (self.categorical_features_totransform != None):
            #df = df.copy(deep=True)                                                                 
            
            #print('Transform data')
            for feature_name in self.categorical_features_totransform:
                matrix_vectorized = self.vectorizers[feature_name].transform(df[feature_name])
                
                bow_features = [feature_name + '_' + str(s) for s in self.vectorizers[feature_name].get_feature_names()]
        
                df_vectorized = pd.DataFrame(matrix_vectorized.todense(), columns=bow_features, dtype='int8')
                del matrix_vectorized
                
                df = pd.concat([df.reset_index(), df_vectorized], axis=1)            
        
            return(df)

        else:
            return(df)
            
    
'''
This class wraps clustering algorithms

It also implements a predict method for Ward algorithm (which does not have
a predict implementation in scikit learn) 
'''    
class Clusterer(BaseEstimator, TransformerMixin):
    def __init__(self, algorithm_to_use='KMEANS', n_clusters=10):
        self.clusterer = None
        self.fitted = False
        self.algos_without_transform = ['WARD'] # List of algorithms that do not have a transform function and must be handled differently in predict()
        self.knn_model = None # Will be used for custom transform for algorithms that do not have a transform function
        
        self.algorithm_to_use = algorithm_to_use
        self.n_clusters = n_clusters
    
    def fit(self, df, labels=None):              
        if (DEBUG_LEVEL >= 1) :
            print('Fit method of Clusterer')
        
        if (self.algorithm_to_use == 'KMEANS'):
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
        
        elif (self.algorithm_to_use == 'WARD'):
            self.clusterer = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='euclidean', linkage='ward')
        
        self.clusterer.fit(df)
        
        if (self.algorithm_to_use in self.algos_without_transform):
            self.knn_model = NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric='minkowski')
            self.knn_model.fit(df)
            
        self.fitted = True
    
        return self

    def predict(self, df):
        if (DEBUG_LEVEL >= 1) :
            print('Predict method of Clusterer')
        
        if (self.algorithm_to_use in self.algos_without_transform):
            # First, get closest instances from training set, to the instances we're predicting
            
            # > This will return 2d array of 1 values : [[indice_0], [indice_1], ...]
            df_train_nearest_neighbors_indices = self.knn_model.kneighbors(df, 1, return_distance=False)  
            
            # > Convert to simple array : [indice_0, indice_1, ...]
            array_train_nearest_neighbors_indices = [df_train_nn[0] for df_train_nn in df_train_nearest_neighbors_indices]

            df_predictions = pd.DataFrame(index=df.index, data=self.clusterer.labels_[array_train_nearest_neighbors_indices])

            # Return cluster labels from these instances on training set, as a series with customer id as index
            #return(self.clusterer.labels_[array_train_nearest_neighbors_indices])
            return(df_predictions[0])
            
        else:
            # Code
            df_predictions = pd.DataFrame(index=df.index, data=self.clusterer.predict(df))
            
            #return(self.clusterer.predict(df))
            return(df_predictions[0])

        #return(labels_predicted)
        
    # If there are less than 2 labels : silhouette score will be -1
    def score(self, X, y=None):
        if (DEBUG_LEVEL >= 1) :
            print('Score method of Clusterer')
        
        predicted_labels = self.predict(X)
        
        if (len(np.unique(predicted_labels)) < 2):
            print('Labels for cluster < 2 : we return a silhouette score of -1')
            return(-1)
        
        else:
            return(silhouette_score(X, predicted_labels))
        
'''
This function is abled to either save a grid search result, or load it 
(depending on SAVE_GRID_RESULTS)
'''
def save_or_load_search_params(grid_search, save_file_suffix):
    if (SAVE_GRID_RESULTS == True):
        #df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
        #df_grid_search_results.to_csv(GRIDSEARCH_CSV_FILE)

        df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["mean_test_score"])],axis=1)
        df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["std_test_score"], columns=["std_test_score"])],axis=1)
        df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_fit_time"], columns=["mean_fit_time"])],axis=1)
        df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_score_time"], columns=["mean_score_time"])],axis=1)
        df_grid_search_results.to_csv(GRIDSEARCH_FILE_PREFIX + save_file_suffix + '.csv')

        with open(GRIDSEARCH_FILE_PREFIX + save_file_suffix + '.pickle', 'wb') as f:
            pickle.dump(grid_search, f, pickle.HIGHEST_PROTOCOL)
            
        return(grid_search, df_grid_search_results)

    if (LOAD_GRID_RESULTS == True):
        if ((SAVE_GRID_RESULTS == True) or (RECOMPUTE_GRIDSEARCH == True)):
            print('Error : if want to load grid results, you should not have saved them or recomputed them before, or you will loose all your training data')

        else:
            with open(GRIDSEARCH_FILE_PREFIX + save_file_suffix + '.pickle', 'rb') as f:
                grid_search = pickle.load(f)

            df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["mean_test_score"])],axis=1)
            df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["std_test_score"], columns=["std_test_score"])],axis=1)
            df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_fit_time"], columns=["mean_fit_time"])],axis=1)
            df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_score_time"], columns=["mean_score_time"])],axis=1)
            
            return(grid_search, df_grid_search_results)

'''
This function loads grid results
'''
def load_search_params(grid_search, save_file_suffix):
    with open(GRIDSEARCH_FILE_PREFIX + save_file_suffix + '.pickle', 'rb') as f:
        grid_search = pickle.load(f)
    
    df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["mean_test_score"])],axis=1)
    df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["std_test_score"], columns=["std_test_score"])],axis=1)
    df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_fit_time"], columns=["mean_fit_time"])],axis=1)
    df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_score_time"], columns=["mean_score_time"])],axis=1)
    
    return(grid_search, df_grid_search_results)


'''
This class is a wrapper for doc2vec (gensim)

At init, it can be passed an already trained doc2vec with gensim library (optional)

In and out data : pandas DataFrame
'''
class Doc2Vec_Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_path=None, model_save_path=None, feature_totransform=None, n_dim=200, window=5, min_count=5, remove_stopwords=True):
        # Passing labels_featurename here for NCA is a mistake :(  it should be passed as a label to fit function, for
        # gridsearch to correctly split labels associated with folds.
        # So, I kept labels_featurenamesfor backwards compatibility with the rest of the notebook.
        # But with GridSearchCV, labels_featurename won't be used :  labels passed to fit will be used instead

        # labels_featurename can be a feature name and also a list of discrete labels
        self.fitted = False
        self.feature_totransform = 'all_text'
        self.n_dim = n_dim
        self.window = window
        self.min_count = min_count
        self.model_path = model_path
        self.model = None

    def fit(self, df,
            labels=None):  # Labels=None added to attempt to remove weird error fit() takes 2 positional arguments but 3 were given
        if (DEBUG_LEVEL >= 1):
            print('Fit Doc2vec_Vectorizer')

        if (self.feature_totransform == None):
            self.fitted = True
            return (self)
        
        if (self.model_path == None): # If no model file to load : we train the model from scratch
            # Constructing tokens
            cnt_label = 0
            InputDocs = []
    
            for document in df[self.feature_totransform ]:            
                doc_transformed = remove_stopwords(document)
                doc_toappend = gensim.utils.simple_preprocess(doc_transformed)
                
                InputDocs.append(TaggedDocument(doc_toappend,[cnt_label]))    
                cnt_label += 1    
    
            # Training model
            start = time.time()
            self.model = Doc2Vec(InputDocs, vector_size=self.n_dim, window=self.window, min_count=self.min_count, workers=4)  # All input docs loaded in memory
            end = time.time()
            
            print('Duration of doc2vec training: ' + str(end - start) + ' seconds')       
            
            # Saving model to file
            if (self.model_save_path != None):
                print('Saving model to file...')
                model.save(self.model_save_path)
            
        else: # Loading model from file
            self.model = Doc2Vec.load(self.model_path)

        self.fitted = True

        return self

    def transform(self, df):
        if (DEBUG_LEVEL >= 1):
            print('Transform Doc2vec_Vectorizer')

        if (self.fitted == False):
            self.fit(df)

        if (self.feature_totransform == None):
            return (df)

        else:
            #return([self.model.infer_vector(gensim.utils.simple_preprocess(text)) for text in df[self.feature_totransform]])
            df_out = pd.DataFrame()
            cnt = 0
            progbar = tqdm(range(df.shape[0]))
            
            for text in df[self.feature_totransform]:
                #df_out.loc[cnt] = self.model.infer_vector(gensim.utils.simple_preprocess(text))
                #print(self.model.infer_vector(gensim.utils.simple_preprocess(text)).tolist())
                #df_out.append(self.model.infer_vector(gensim.utils.simple_preprocess(text)).tolist())
                
                df_out = df_out.append(pd.Series(self.model.infer_vector(gensim.utils.simple_preprocess(text))), ignore_index=True)
                
                progbar.update(1)
                cnt += 1
            
            return(df_out)
    