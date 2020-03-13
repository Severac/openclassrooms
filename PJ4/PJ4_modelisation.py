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


# # Data load

# In[5]:


# hhmm timed features formatted
feats_hhmm = ['CRS_DEP_TIME',  'CRS_ARR_TIME']

df = pd.read_csv(DATA_PATH_FILE_INPUT, sep=',', header=0, encoding='utf-8', low_memory=False, parse_dates=feats_hhmm)


# In[6]:


df.shape


# In[7]:


display_percent_complete(df)


# In[8]:


for column_name in df.columns:
    print_column_information(df, column_name)


# # Identification of features

# In[9]:


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


# # Features encoding

# ## Convert _TIME features (HHMM format) to minutes

# In[10]:


#df convert_HHMM_to_Minutes


# In[11]:


df['CRS_ARR_TIME']


# In[12]:


# Concatenation of first 2 digits (hours)  and last 2 digits (minutes)
df_CRS_ARR_TIME = pd.concat([df['CRS_ARR_TIME'].str.slice(start=0,stop=2, step=1),df['CRS_ARR_TIME'].str.slice(start=2,stop=4, step=1)], axis=1).astype(int)


# In[13]:


# Conversion of hours and minutes into minutes :  HH*60 + MM = minutes
df['CRS_ARR_TIME'] = (df_CRS_ARR_TIME.iloc[:, [0]] * 60 + df_CRS_ARR_TIME.iloc[:, [1]])['CRS_ARR_TIME']


# In[14]:


df


# In[ ]:





# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF

from sklearn import decomposition
from sklearn import preprocessing
#from sklearn.neighbors import KNeighborsTransformer

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import NeighborhoodComponentsAnalysis

import statistics

class DuplicatesRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.df_origin = None
        return None
    
    def fit(self, df, labels=None):      
        return self
    
    def transform(self, df):
        self.df_origin = df
        df = df.copy(deep=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return(df)
    
class NumericalFeaturesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features=['movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes', 'imdb_score' , 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'facenumber_in_poster', 'duration', 'num_user_for_reviews', 'actor_3_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'budget', 'gross','title_year']):
        self.numerical_features = numerical_features
    
    def fit(self, df, labels=None):      
        return self
    
    def transform(self, df):
        numerical_features_columns = df[self.numerical_features].columns
        numerical_features_index = df[self.numerical_features].index

        # Imputation par régression linéaire :
        imp = IterativeImputer(max_iter=10, random_state=0)
        transformed_data = imp.fit_transform(df[self.numerical_features])  

        # Drop des features non imputées sur l'axe des colonnes
        df.drop(labels=self.numerical_features, axis=1, inplace=True)
        
        # Recréation d'un dataframe avec les features imputées
        df_numerical_features_imputed = pd.DataFrame(data=transformed_data, columns=numerical_features_columns, index=numerical_features_index)
        
        df = pd.concat([df, df_numerical_features_imputed], axis=1)
        
        return(df)

'''
Cette fonction fait un 1 hot encoding des features qui sont des catégories
Elle fonctionne pour les 2 cas de figure suivant :
- Les valeurs possibles de la colonne sont une chaîne de caractère (ex : cat1)
- Les valeurs possibles de la colonne sont des chaînes de caractère avec des séparateurs (ex:  cat1|cat2|cat3)
'''
    
def add_categorical_features_1hot(df, categorical_features_totransform):
    #df.drop(labels=categorical_features_totransform, axis=1, inplace=True)
    
    for feature_totransform in categorical_features_totransform:
        print(f'Adding 1hot Feature : {feature_totransform}')
        
        df_transformed = df[feature_totransform].str.get_dummies().add_prefix(feature_totransform +'_')   
        df.drop(labels=feature_totransform, axis=1, inplace=True)
        
        df = pd.concat([df, df_transformed], axis=1)
        
    return(df)


'''
Cette fonction commence par merger les valeurs de toutes les colonnes comprises dans 
categorical_features_tomerge_andtransform  dans une seule colonne temporaire
Puis elle fait un 1 hot encode du résultat en appelant la fonction add_categorical_features_1hot

df :  dataframe source, sur lequel on va droper les features avant transformation, et ajouter les features après transformation
categorical_features_tomerge_andtransform : la liste des catégories à merger et à 1 hot encode,
             par exemple: ['actor_1_name', 'actor_2_name', 'actor_3_name']
merged_feature_name : le nom de la feature qui sera mergée
    exemple si le nom est 'actors_names'
    On pourra avoir les colonnes suivantes de créées:  'actors_names_Le nom du 1er acteur', 'actors_names_Le nom du 2eme acteur'
          
'''
def add_categorical_features_merge_and_1hot(df, categorical_features_tomerge_andtransform, merged_feature_name):
    #df.drop(labels=categorical_features_tomerge_andtransform, axis=1, inplace=True)
    
    cnt = 0
    for feature_totransform in categorical_features_tomerge_andtransform:                            
        if (cnt == 0):
            df[merged_feature_name] = df[feature_totransform]
        
        else:
            df[merged_feature_name] = df[merged_feature_name] + '|' + df[feature_totransform]
            
        cnt += 1
    
    df.drop(labels=categorical_features_tomerge_andtransform, axis=1, inplace=True)
    
    return(add_categorical_features_1hot(df, [merged_feature_name]))
    

        
class CategoricalFeatures1HotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features_totransform =['country', 'language', 'director_name', 'genres', 'plot_keywords', 'color', 'content_rating']):
        self.categorical_features_totransform = categorical_features_totransform
    
    def fit(self, df, labels=None):      
        return self
    
    def transform(self, df):       
        return(add_categorical_features_1hot(df, self.categorical_features_totransform))

class CategoricalFeaturesMergerAnd1HotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features_tomerge_andtransform = ['actor_1_name', 'actor_2_name', 'actor_3_name'], merged_feature_name = 'actors_names'):
        self.categorical_features_tomerge_andtransform = categorical_features_tomerge_andtransform
        self.merged_feature_name = merged_feature_name
    
    def fit(self, df, labels=None):      
        return self
    
    def transform(self, df):       
        return(add_categorical_features_merge_and_1hot(df, self.categorical_features_tomerge_andtransform, self.merged_feature_name))   

class CategoricalFeaturesBowEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features_tobow = ['movie_title']):
        self.categorical_features_tobow = categorical_features_tobow
    
    def fit(self, df, labels=None):      
        return self
    
    def transform(self, df):       
        for feature_tobow in self.categorical_features_tobow:
            print(f'Adding bow Feature. : {feature_tobow}')

            df_transformed = df[feature_tobow].str.lower().str.replace(r'[^\w\s]', '').str.replace(u'\xa0', u'').str.get_dummies(sep=' ').add_prefix(feature_tobow +'_')
            df.drop(labels=feature_tobow, axis=1, inplace=True)
            df = pd.concat([df, df_transformed], axis=1)
        
        return(df)

class FeaturesDroper(BaseEstimator, TransformerMixin):
    def __init__(self, features_todrop = ['aspect_ratio', 'movie_imdb_link']):
        self.features_todrop = features_todrop
    
    def fit(self, df, labels=None):      
        return self
    
    def transform(self, df):       
        #df.drop(labels=self.features_todrop, axis=1, inplace=True)  
        if (self.features_todrop != None):
            for feature_to_drop in self.features_todrop:
                df = df.loc[:,~df.columns.str.startswith(feature_to_drop)]
            print('Features drop done')
            
        return(df)
    
'''
This function predicts imdb_score using KNN technique



'''
class KNNTransform(BaseEstimator, TransformerMixin):
    def __init__(self, knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'}):
        self.knn_params = knn_params
        self.nbrs = None
        self.labels = None
        self.metric = None
    
    def fit(self, X, labels=None, df_encoded=None):  
        # df_encoded sert pour la classe KNNTransform_predict_similarity
        # df_encoded est le df 1hot encoded numérique après les étapes de transformation
        print('KNN fit')
        self.df_encoded = df_encoded
        self.labels = labels
        self.nbrs = NearestNeighbors(n_neighbors=self.knn_params['n_neighbors'], algorithm=self.knn_params['algorithm'], metric=self.knn_params['metric']).fit(X)
        return self

    
    '''
    # This function returns similarity score for each film instance in X  (summed for all X instances)
    # The more the score is, the more recommended films (knn_matrix[film_instance, 1...5]) 
    # are close to input film (knn_matrix[film_instance, 0]) 

    /!\ To calculate similarity,  this method requires to set global variable df_encoded 
    (df_encoded as being the output of preparation_pipeline)
    
    Previous version of the method used to accept df_encoded pass as a parameter to KNNTransform fit function
    Parameter was passed like this :
    recommendation_pipeline_PCA_KNN.fit(df_encoded, labels, KNN__df_encoded = df_encoded)
    > But this did not work with GridSearchCV (that does not support third customer parameter to fit function)
    '''
    def score(self, X, y=None):
        print('KNN score')

        distances_matrix, knn_matrix = self.nbrs.kneighbors(X)

        scorings = []
        
        scaler = MinMaxScaler() 
        array_scaled = scaler.fit_transform(df_encoded)
        df_scaled  = pd.DataFrame(data=array_scaled , columns=df_encoded.columns, index=df_encoded.index)
        
    
        # Drop features that have nothing to do with the film items themselves, and are not to be taken into account for similarity
        df_scaled.drop(labels=['movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes', 'imdb_score', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'num_user_for_reviews', 'actor_3_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes'], axis=1, inplace=True)
        
        
        for i in range(0, X.shape[0]):
            scoring_1 = get_similarity_df_scaled_input(df_scaled, knn_matrix[i, 0], knn_matrix[i, 1]).sum()
            scoring_2 = get_similarity_df_scaled_input(df_scaled, knn_matrix[i, 0], knn_matrix[i, 2]).sum()
            scoring_3 = get_similarity_df_scaled_input(df_scaled, knn_matrix[i, 0], knn_matrix[i, 3]).sum()
            scoring_4 = get_similarity_df_scaled_input(df_scaled, knn_matrix[i, 0], knn_matrix[i, 4]).sum()
            scoring_5 = get_similarity_df_scaled_input(df_scaled, knn_matrix[i, 0], knn_matrix[i, 5]).sum()
                
            scorings.append((scoring_1 + scoring_2 + scoring_3 + scoring_4 + scoring_5) / 5)
        
        return(statistics.mean(scorings))
    
    
    def predict(self, X, y=None): # Quand on appelle predict, transform est appelé avant automatiquement
        print('KNN predict')

        distances_matrix, knn_matrix = self.nbrs.kneighbors(X)

        # Pour chaque film (chaque ligne comprise dans X), on calcule la prédiction du score ci-dessous
        # On fait la moyenne des scores  (compris dans labels) de chaques films renvoyés par le KNN
        scoring_predictions = (self.labels[knn_matrix[:,1]] + self.labels[knn_matrix[:,2]] + self.labels[knn_matrix[:,3]] + self.labels[knn_matrix[:,4]] + self.labels[knn_matrix[:,5]])/5
        
        return(scoring_predictions)
    
    def transform(self, X):   
        print('KNN transform')
        #distances_matrix, reco_matrix = nbrs.kneighbors(X)
        return(self.nbrs.kneighbors(X))    

    
'''
Cette fonction permet de fournir un score de similarité moyen entre chaque prédiction et le film d'origine

On peut calculer cette similarité en faisant un apply sur 2 variables:  
en appelant la fonction get_similarity_df(df_encoded, index1, index2).  
Pour ça, on passe df_encoded à la classe KNNTransform

Le paramètre devra être passé comme ceci:
recommendation_pipeline_PCA_KNN.fit(df_encoded, labels, KNN__df_encoded = df_encoded)
'''
    
class KNNTransform_predict_similarity(KNNTransform):
    def score(self, X, y=None):
        distances_matrix, knn_matrix = self.nbrs.kneighbors(X)

        scaler = MinMaxScaler() 
        array_scaled = scaler.fit_transform(self.df_encoded)        
        
        scoring_1 = ((array_scaled[:, 0] + array_scaled[:, 1]) / 2)
        # A compléter
    
    def predict(self, X, y=None): # Quand on appelle predict, transform est appelé avant automatiquement
        print('KNN predict')

        distances_matrix, knn_matrix = self.nbrs.kneighbors(X)

        scoring_predictions = []
        
        scaler = MinMaxScaler() 
        array_scaled = scaler.fit_transform(self.df_encoded)
        df_scaled  = pd.DataFrame(data=array_scaled , columns=self.df_encoded.columns, index=self.df_encoded.index)
        
        print('X.shape[0] : ' + str(X.shape[0]))
        for i in range(0, X.shape[0]):
            scoring_1 = get_similarity_df_scaled_input(df_scaled, knn_matrix[i, 0], knn_matrix[i, 1]).sum()
            scoring_2 = get_similarity_df_scaled_input(df_scaled, knn_matrix[i, 0], knn_matrix[i, 2]).sum()
            scoring_3 = get_similarity_df_scaled_input(df_scaled, knn_matrix[i, 0], knn_matrix[i, 3]).sum()
            scoring_4 = get_similarity_df_scaled_input(df_scaled, knn_matrix[i, 0], knn_matrix[i, 4]).sum()
            scoring_5 = get_similarity_df_scaled_input(df_scaled, knn_matrix[i, 0], knn_matrix[i, 5]).sum()
                
            scoring_predictions.append((scoring_1 + scoring_2 + scoring_3 + scoring_4 + scoring_5) / 5)
            
        '''
        df_knn_matrix = pd.DataFrame(data=knn_matrix)
        
        df_knn_matrix['similarity_1_score'] = df[[0, 1]].apply(segmentMatch, axis=1)
        
        # Pour chaque film (chaque ligne comprise dans X), on calcule la prédiction du score ci-dessous
        # On fait la moyenne des scores  (compris dans labels) de chaques films renvoyés par le KNN
        scoring_predictions = (self.labels[knn_matrix[:,1]] + self.labels[knn_matrix[:,2]] + self.labels[knn_matrix[:,3]] + self.labels[knn_matrix[:,4]] + self.labels[knn_matrix[:,5]])/5
        '''
        
        return(scoring_predictions)
    
'''
This function wraps NCA transformer. What it does more, is that it generates discretized categorical labels
from numerical score labels that are passed as input.
Labels are mandatory in order to use NCA transformer.
'''    
    
class NCATransform(BaseEstimator, TransformerMixin):
    def __init__(self, nca_params =  {'random_state':42, 'n_components':200 }):
        self.nca_params = nca_params
        self.nca = None

    def fit(self, X, labels=None):      
        print('NCA fit')
        self.labels = labels

        # Discretize labels for the need of NCA algorithm :
        df_labels =  pd.DataFrame(data=labels)
        self.labels_discrete = pd.cut(df_labels[0], bins=range(1,10), right=True).astype(str).tolist()
        
        self.nca = NeighborhoodComponentsAnalysis(random_state=self.nca_params['random_state'], n_components=self.nca_params['n_components'])
        self.nca.fit(X, self.labels_discrete)
            
        return self
 
    def transform(self, X):   
        print('NCA transform')
        return(self.nca.transform(X))    
    
'''
This class wraps either NCA transformer, or PCA transformer (to be used with grid search)
'''

class DimensionalityReduction_Transform(BaseEstimator, TransformerMixin):
    def __init__(self, reduction_params =  {'reduction_type': 'PCA', 'n_components':200 }):
        self.reduction_params = reduction_params
        self.model = None
        #self.pca = None

    def fit(self, X, labels=None):      
        if (self.reduction_params['reduction_type'] == 'NCA'):        
            print('NCA fit')
            self.labels = labels

            # Discretize labels for the need of NCA algorithm :
            df_labels =  pd.DataFrame(data=labels)
            self.labels_discrete = pd.cut(df_labels[0], bins=range(1,10), right=True).astype(str).tolist()

            self.model = NeighborhoodComponentsAnalysis(random_state=42, n_components=self.reduction_params['n_components'])
            self.model.fit(X, self.labels_discrete)

            return self
 
        if (self.reduction_params['reduction_type'] == 'PCA'):
            self.model = decomposition.PCA(n_components=self.reduction_params['n_components'])
            self.model.fit(X)
            
            return self
            

    def transform(self, X):   
        if (self.reduction_params['reduction_type'] == 'NCA'):
            print('NCA transform')
        
        if (self.reduction_params['reduction_type'] == 'PCA'):
            print('PCA transform')
        
        return(self.model.transform(X))

class PipelineFinal(BaseEstimator, TransformerMixin):
    def __init__(self, params=None):
        print('init')
        self.params = params
    
    def fit(self, df, labels=None):   
        print('fit')
        return self
    
    def predict(self, df, y=None):
        print('predict')
        return([i for i in range(df.shape[0])])
    
    def fit_predict(self, df, labels=None):
        self.fit(df)
        return self.predict(df)
    
    def transform(self, df):
        print('transform')
        return(df)
        #return(df.to_numpy())


preparation_pipeline = Pipeline([
    ('duplicates_remover', DuplicatesRemover()),
    ('numerical_features_imputer', NumericalFeaturesImputer()),
    ('categoricalfeatures_1hotencoder', CategoricalFeatures1HotEncoder()),
    ('categoricalfeatures_merger_1hotencoder', CategoricalFeaturesMergerAnd1HotEncoder()),
    ('categoricalfeatures_bow_1hotencoder', CategoricalFeaturesBowEncoder()),
    ('features_droper', FeaturesDroper(features_todrop=['aspect_ratio', 'movie_imdb_link'])),

])


recommendation_pipeline_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('reduction', DimensionalityReduction_Transform(reduction_params = {'reduction_type' : 'PCA', 'n_components' : 200 })),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
])

recommendation_pipeline_PCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('pca', decomposition.PCA(n_components=200)),
    ('KNN', KNNTransform()),
    #('pipeline_final', PipelineFinal()),

])


recommendation_pipeline_NCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('NCA', NCATransform(nca_params =  {'random_state':42, 'n_components':200 })),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    #('pipeline_final', PipelineFinal()),
])


recommendation_pipeline_NMF = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('NMF', NMF(n_components=200, init='random', random_state=0)),
    ('KNN', KNNTransform()),
    #('pipeline_final', PipelineFinal()),

])

reducer_pipeline = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('NCA', NCATransform(nca_params =  {'random_state':42, 'n_components':200 })),
])

reducer_pipeline10 = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('NCA', NCATransform(nca_params =  {'random_state':42, 'n_components':10 })),
])

'''
kmeans_transformer_pipeline = Pipeline([
    ('kmeans', KMeans(n_clusters=10)),
    #('pipeline_final', PipelineFinal()),

])
'''

