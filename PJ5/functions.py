#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:45:19 2020

@author: francois
"""


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

import statistics

from scipy import sparse

import pandas as pd

import qgrid

import numpy as np

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
        print('!! categorical_features_totransform' + str(self.categorical_features_totransform))

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
            
            print('1hot encode categorical features...')
            #df_encoded = pd.get_dummies(df, columns=self.categorical_features_totransform, sparse=True)  # Sparse allows to gain memory. But then, standardscale must be with_mean=False
            df_encoded = pd.get_dummies(df, columns=self.categorical_features_totransform, sparse=False)

            # Get category values that were in fitted data, but that are not in data to transform 
            for feature_name, feature_values in self.all_feature_values.items():
                diff_columns = list(set(feature_values) - set(df_encoded.columns.tolist()))
                print(f'Column values that were in fitted data but not in current data: {diff_columns}')

                if (len(diff_columns) > 0):
                    print('Adding those column with 0 values to the DataFrme...')
                    # Create columns with 0 for the above categories, in order to preserve same matrix shape between train et test set
                    zeros_dict = dict.fromkeys(diff_columns, 0)
                    df_encoded = df_encoded.assign(**zeros_dict)

            print('type of df : ' + str(type(df_encoded)))
            return(df_encoded)

        else:
            return(df)

class Aggregate_then_GroupByMean_then_Sort_numericalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features_totransform=['ORIGIN', 'UNIQUE_CARRIER', 'DEST']):
        self.categorical_features_totransform = categorical_features_totransform
        self.fitted = False
        self.all_feature_values = {}
    
    def fit(self, df, labels=None):      
        print('Fit data')
        
        self.feature_maps = {}
        
        for feature_name in self.categorical_features_totransform:
            print(f'Fitting feature {feature_name}')
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
        
        print('Encode categorical features...')
        
        for feature_name in self.categorical_features_totransform:
            print(f'Encoding feature {feature_name} ...')
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
        return self
    
    def transform(self, df):       
        if (self.features_toselect != None):
            filter_cols = [col for col in df if (col.startswith(tuple(self.features_toselect)))]
            
            filter_cols.sort()
            
            print("Features selected (in order): " + str(df[filter_cols].columns))
            
            df = df.loc[:, filter_cols]
            return(df)

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
        

class MinMaxScalerMultiple(BaseEstimator, TransformerMixin):
    def __init__(self, features_toscale=None):
        self.fitted = False
        self.columns = features_toscale
    
    def fit(self, df):              
        print('Fit Min max scaler multiple')
        self.scaler = MinMaxScaler()
  
        if (self.columns == None):
            self.fitted = True
            return(df)
        else:
            if (self.columns == 'ALL_FEATURES'):
                self.columns = df.columns.tolist()
                #print(f'self.columns = {self.columns}')
                
            self.scaler.fit(df[self.columns].to_numpy())            
            self.fitted = True
        
        return self
    
    def transform(self, df):
        print('Transform Min max scaler multiple')
        if (self.fitted == False):
            self.fit(df)
        
        if (self.columns == None):
            return(df)
        
        else:
            df.loc[:, self.columns] = self.scaler.transform(df.loc[:, self.columns].to_numpy())

        return(df)        
        
class LogScalerMultiple(BaseEstimator, TransformerMixin):
    def __init__(self, features_toscale=None):
        self.fitted = False
        self.columns = features_toscale
    
    def fit(self, df):              
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
        print('Transform log scaler scaler multiple')
        if (self.fitted == False):
            self.fit(df)
        
        if (self.columns == None):
            return(df)
        
        else:
            df.loc[:, self.columns] = self.scaler.transform(df.loc[:, self.columns].to_numpy())

        return(df)    


class DimensionalityReductor(BaseEstimator, TransformerMixin):
    def __init__(self, features_totransform=None, algorithm_to_use='PCA', n_dim=20, labels_featurename=None):
        self.fitted = False
        self.features_totransform = features_totransform
        self.algorithm_to_use = algorithm_to_use
        self.n_dim = n_dim
        self.labels_featurename = labels_featurename
    
    def fit(self, df):              
        print('Fit Dimensionality Reductor')
              
        if (self.features_totransform == None):
            self.fitted = True
            return(df)
            
        else:            
            if (self.algorithm_to_use == 'PCA'):
                self.reductor = PCA(n_components=self.n_dim, random_state=42)                
            
            if (self.algorithm_to_use == 'NCA'):
                '''
                if (self.labels == None):
                    self.labels = df['TotalPricePerMonth']
                '''
                
                self.labels_discrete = pd.cut(df[self.labels_featurename], bins=range(1,10), right=True).astype(str).tolist()
                self.reductor = NeighborhoodComponentsAnalysis(random_state=42, n_components=self.n_dim)
                
            
            if (self.algorithm_to_use == 'TSNE'):
                self.reductor = TSNE(n_components=self.n_dim, random_state=42)
            
            self.filter_cols = [col for col in df if (col.startswith(tuple(self.features_totransform)))]            
            self.filter_cols.sort()
            
            print("Features selected (in order): " + str(df[self.filter_cols].columns))            
            
            if (self.algorithm_to_use == 'PCA'):
                self.reductor.fit(df[self.filter_cols].to_numpy())            
            
            if (self.algorithm_to_use == 'NCA'):
                self.reductor.fit(df[self.filter_cols].to_numpy(), self.labels_discrete)       
            
            if  (self.algorithm_to_use == 'TSNE'):
                print('No fit for TSNE')
            
            self.fitted = True
        
        return self
    
    def transform(self, df):
        print('Transform Dimensionality Reductor')
        if (self.fitted == False):
            self.fit(df)
        
        if (self.features_totransform == None):
            return(df)
        
        else:
            remaining_columns = list(set(df.columns.tolist()) - set(self.filter_cols))
            
            print(f'Remaining columns: {remaining_columns}')
            
            if (self.algorithm_to_use == 'TSNE'):
                # TNSE has not transform function so we have to do fit_transform directly
                np_transformed = self.reductor.fit_transform(df[self.filter_cols].to_numpy())            
            
            else:
                np_transformed = self.reductor.transform(df[self.filter_cols].to_numpy())
            
            if (remaining_columns != []):
                df_transformed = pd.concat([df[remaining_columns].reset_index(drop=True), pd.DataFrame(np_transformed)], axis=1)
                
            else:
                df_transformed = pd.DataFrame(np_transformed)

        return(df_transformed)    

 

def load_data(in_file):
    df = pd.read_csv(in_file, encoding='utf-8', converters={'InvoiceNo': str, 'StockCode':str, 'Description': str, \
                                       'CustomerID':str, 'Country': str, 'DescriptionNormalized': str})   
        
    return(df)
    
def custom_train_test_split_sample(df, split_feature, SAMPLED_DATA=False):
    from sklearn.model_selection import train_test_split
    
    if (SAMPLED_DATA == True):
        df_labels_discrete = pd.cut(df[split_feature], bins=50)
        df, df2 = train_test_split(df, train_size=NB_SAMPLES, random_state=42, shuffle = True, stratify = df_labels_discrete)
            
    #df_labels_discrete = pd.cut(df[split_feature], bins=50)
    df_labels_discrete = pd.qcut(df[split_feature], 10)
    
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, shuffle = True, stratify = df_labels_discrete)
    #df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

    convert_dict = {'InvoiceNo': str, 'StockCode':str, 'Description': str, \
                                       'CustomerID':str, 'Country': str, 'DescriptionNormalized': str}
  
    df_train = df_train.astype(convert_dict) 
    df_test = df_test.astype(convert_dict)
    
    '''
    df_train = df_train.copy()
    df_test = df_test.copy()
    '''
    
    return df, df_train, df_test



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
    def fit(self, df, categorical_features_totransform=['DescriptionNormalized']):      
        print('BowEncoder : Fit data')
        print(f'categorical_features_totransform == {categorical_features_totransform}')
        self.categorical_features_totransform = categorical_features_totransform
        print('!! categorical_features_totransform' + str(self.categorical_features_totransform))

        if (self.categorical_features_totransform != None):
            for feature_name in self.categorical_features_totransform:
                self.vectorizers[feature_name] = CountVectorizer(min_df=self.min_df)
                matrix_vectorized = self.vectorizers[feature_name].fit(df[feature_name])
                                
        self.fitted = True
        
        return self
    
    def transform(self, df):
        if (self.fitted == False):
            self.fit(df)
        
        if (self.categorical_features_totransform != None):
            print('Transform data')
            for feature_name in self.categorical_features_totransform:
                matrix_vectorized = self.vectorizers[feature_name].transform(df[feature_name])
                
                bow_features = [feature_name + '_' + str(s) for s in self.vectorizers[feature_name].get_feature_names()]
        
                df_vectorized = pd.DataFrame(matrix_vectorized.todense(), columns=bow_features, dtype='int8')
                del matrix_vectorized
                
                df = pd.concat([df, df_vectorized], axis=1)            
        
            return(df)

        else:
            return(df)
            
'''
This function agregates orders to client level :
    - Drop generic products 
    - Get client ids that have cancelled at least 1 order
    - Get client ids that bought top value products (passed as input to the model)
    - Remove cancellations
    - Calculate number of months for each client max(last month - first month ordered, 1 month)
    
    - Agregate on client level and sum: TotalPrice, BoW features
        > Then divide Total price by number of months of client
        
    - Add features :
        - Client has cancelled at least 1 order
        - Clients has bought top value product
'''
            
class AgregateToClientLevel(BaseEstimator, TransformerMixin):
    def __init__(self, stockcodes_top_value_products):
        self.fitted = False
        self.stockcodes_top_value_products = stockcodes_top_value_products
    
    def fit(self, df):      
        print('AgregateToClientLevel : Fit data')

                                
        self.fitted = True
        
        return self
    
    def transform(self, df):
        if (self.fitted == False):
            self.fit(df)
        
        LAST_ORDER_DATE_IN_DATASET = '2011-12-09 12:50:00'
        
        # NEED TO fix agregation of TotalPrice on these products :
        # Drop generic products (postage fees, manual entries, dot postage, CRUK commission) 
        # But keep discounts (StockCode == 'D')
        # In data modelisation notebook there will be nothing to drop since we cleaned data in exploration notebook
        # But for live data feed, it may needed to drop those lines
        df.drop(index=df[df['StockCode'].isin(['POST', 'M', 'PADS', 'DOT', 'CRUK'])].index, axis=0, inplace=True)
        
        # Get client ids that have cancelled at least 1 order
        custid_cancelled = df[df['InvoiceNo'].str.startswith('C') == True]['CustomerID'].unique()
        
        # Get client ids that bought top value products (passed as input to the model)
        custid_topvalue = df[df['StockCode'].isin(self.stockcodes_top_value_products)]['CustomerID'].unique()

        # Remove cancellations
        df_nocancel = df[df['InvoiceNo'].str.startswith('C') == False]
        df_nocancel.reset_index(inplace=True)
        
        # Calculate number of months for each client max(last month - first month ordered, 1 month)
        series_gbclient_nbmonths = np.maximum(\
            (\
               (\
               pd.to_datetime('2011-12-09 12:50:00')\
               - pd.to_datetime(df_nocancel[['CustomerID', 'InvoiceDate']].groupby('CustomerID').min()['InvoiceDate'])
               )\
                / (np.timedelta64(1, "M"))\
            ), 1)

        # Agregate on client level and sum: TotalPrice, BoW features
        feat_list = ['CustomerID', 'TotalPrice']
        feat_list_bow = [col for col in df_nocancel if col.startswith('DescriptionNormalized_')]
        feat_list.extend(feat_list_bow)
        
        df_gbcust_nocancel = df_nocancel[feat_list].groupby('CustomerID').sum()
        # Set max value of bow features that we just summed to 1 :
        df_gbcust_nocancel[feat_list_bow] = df_gbcust_nocancel[feat_list_bow].clip(upper=1)
        
        # Then divide Total price by number of months of client
        df_gbcust_nocancel['TotalPrice'] = df_gbcust_nocancel['TotalPrice'] / series_gbclient_nbmonths
        df_gbcust_nocancel.rename(columns={'TotalPrice' : 'TotalPricePerMonth'}, inplace=True)

        # Add feature : Client has cancelled at least 1 order
        df_gbcust_nocancel['HasEverCancelled'] = 0
        df_gbcust_nocancel.loc[df_gbcust_nocancel.index.isin(custid_cancelled), 'HasEverCancelled'] = 1

        # Add feature : Clients has bought top value product
        df_gbcust_nocancel['BoughtTopValueProduct'] = 0
        df_gbcust_nocancel.loc[df_gbcust_nocancel.index.isin(custid_topvalue), 'BoughtTopValueProduct'] = 1

        # Now, need to remove POSTAGE products and maybe DISCOUNT, and M

        return(df_gbcust_nocancel)