#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ3 : IMDB dataset :  data clean and modelisation notebook 

# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import zipfile
import urllib

import matplotlib.pyplot as plt

import numpy as np

import qgrid

from pandas.plotting import scatter_matrix

DOWNLOAD_ROOT = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Moteur+de+recommandation+de+films/"
DATA_PATH = os.path.join("datasets", "imdb")

DATA_PATH_FILE = os.path.join(DATA_PATH, "movie_metadata.csv")
DATA_URL = DOWNLOAD_ROOT + "imdb-5000-movie-dataset.zip"

DATA_PATH_FILE_OUTPUT = os.path.join(DATA_PATH, "movie_metadata_transformed.csv")

DOWNLOAD_DATA = False  # A la première exécution du notebook, ou pour rafraîchir les données, mettre cette variable à True

plt.rcParams["figure.figsize"] = [16,9] # Taille par défaut des figures de matplotlib

import seaborn as sns
sns.set()

####### Paramètres pour sauver et restaurer les modèles :
import pickle

####### Paramètres à changer par l'utilisateur selon son besoin :
RECOMPUTE_GRIDSEARCH = True  # CAUTION : computation is several hours long
SAVE_GRID_RESULTS = True # If True : grid results object will be saved to GRIDSEARCH_PICKLE_FILE, and accuracy results to SEARCH_CSV_FILE
LOAD_GRID_RESULTS = False # If True : grid results object will be loaded from GRIDSEARCH_PICKLE_FILE

GRIDSEARCH_PICKLE_FILE = 'grid_search_results.pickle'
GRIDSEARCH_CSV_FILE = 'grid_search_results.csv'

GRIDSEARCH_FILE_PREFIX = 'grid_search_results_'

SAVE_API_MODEL = True # If True : API model containing recommendation matrix and DataFrame (with duplicates removed) will be saved
API_MODEL_PICKLE_FILE = 'API_model_PJ3.pickle'

EXECUTE_INTERMEDIATE_MODELS = True # If True: every intermediate model (which results are manually analyzed in the notebook) will be executed


# In[2]:


def qgrid_show(df):
    display(qgrid.show_grid(df, grid_options={'forceFitColumns': False, 'defaultColumnWidth': 170}))


# # Téléchargement et décompression des données

# In[3]:


PROXY_DEF = 'BNP'
#PROXY_DEF = None

def fetch_dataset(data_url=DATA_URL, data_path=DATA_PATH):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    archive_path = os.path.join(data_path, "imdb-5000-movie-dataset.zip")
    
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


# # Import du fichier CSV

# ## Chargement des données

# In[5]:


import pandas as pd
# Commented, because notebook would be too big and slow :
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)

def load_data(data_path=DATA_PATH):
    csv_path = DATA_PATH_FILE
    return pd.read_csv(csv_path, sep=',', header=0, encoding='utf-8')


# In[6]:


df = load_data()


# ###  On vérifie que le nombre de lignes intégrées dans le Dataframe correspond au nombre de lignes du fichier

# In[7]:


num_lines = sum(1 for line in open(DATA_PATH_FILE, encoding='utf-8'))
message = (
f"Nombre de lignes dans le fichier (en comptant l'entête): {num_lines}\n"
f"Nombre d'instances dans le dataframe: {df.shape[0]}"
)
print(message)


# ### Puis on affiche quelques instances de données :

# In[8]:


df.head()


# ### Vérification s'il y a des doublons

# In[9]:


df[df.duplicated()]


# ### Suppression des doublons

# In[10]:


df.drop_duplicates(inplace=True)
#df = df.reset_index(drop=True)
df.reset_index(drop=True, inplace=True)


# In[11]:


df.info()


# Imputation des variables manquantes :
#     https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
# https://scikit-learn.org/stable/modules/impute.html  => à lire en premier
# 
# ACP et 1 hot encoding :  y a-t-il une autre possibilité ?
# 
# Valeurs de variables très peu représentées :   => à voir dans un second temps
# https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
# 

# ## Affichage des champs renseignés (non NA) avec leur pourcentage de complétude
# L'objectif est de voir quelles sont les features qui seront les plus fiables en terme de qualité de donnée, et quelles sont celles pour lesquelles on devra faire des choix

# In[12]:


(df.count()/df.shape[0]).sort_values(axis=0, ascending=False)


# ## Identification des typologies de features à traiter 

# In[13]:


numerical_features = ['movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes', 'imdb_score' , 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'facenumber_in_poster', 'duration', 'num_user_for_reviews', 'actor_3_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'budget', 'gross','title_year']

# à 1 hot encoder, et à splitter avant si nécessaire  ('genres' et 'plot_keywords' doivent être splittées)
categorical_features = ['country', 'language', 'director_name', 'genres', 'plot_keywords', 'color', 'content_rating']
#categorical_features = ['language', 'director_name', 'genres', 'plot_keywords', 'color', 'content_rating']

## => Rajouter language (pour l'instant il n'est pas pris en compte)

# à transformer en bag of words
categorical_features_tobow = ['movie_title']  

# à fusioner en 1 seule variable
categorical_features_tomerge = ['actor_1_name', 'actor_2_name', 'actor_3_name']  

# features qui ne seront pas conservées :
features_notkept = ['aspect_ratio', 'movie_imdb_link']




# ## Affichage des features qui seront splittées avant le 1hot encode :

# In[14]:


df[['genres', 'plot_keywords']].sample(10)


# # Encodage des features

# In[18]:


# KNN imputer pas encore supporté par la version de sklearn que j'utilise :

#from sklearn.impute import KNNImputer

#imputer = KNNImputer(n_neighbors=2, weights="uniform")  
#imputer.fit_transform(df[numerical_features])


# In[18]:


numerical_features_columns = df[numerical_features].columns


# In[19]:


numerical_features_index = df[numerical_features].index


# In[20]:


numerical_features_columns.shape


# ## Imputation des données numériques par régression linéaire

# In[22]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)

transformed_data = imp.fit_transform(df[numerical_features])  


# In[23]:


df_numerical_features_imputed = pd.DataFrame(data=transformed_data, columns=numerical_features_columns, index=numerical_features_index)


# ### Visualisation de quelques résultats par comparaison avant/après :

# In[24]:


qgrid_show(df[numerical_features])


# In[25]:


qgrid_show(df_numerical_features_imputed)


# ### Constat que toutes les valeurs sont maintenant renseignées :

# In[26]:


(df_numerical_features_imputed.count()/df_numerical_features_imputed.shape[0]).sort_values(axis=0, ascending=False)


# ## Transformation des features de catégorie
# ### Voir le §  Identification des typologies de features à traiter  ci-dessus pour une explication des différents cas de figure

# In[51]:


'''
Cette fonction fait un 1 hot encoding des features qui sont des catégories
Elle fonctionne pour les 2 cas de figure suivant :
- Les valeurs possibles de la colonne sont une chaîne de caractère (ex : cat1)
- Les valeurs possibles de la colonne sont des chaînes de caractère avec des séparateurs (ex:  cat1|cat2|cat3)
'''

def add_categorical_features_1hot(df, df_target, categorical_features_totransform):
    for feature_totransform in categorical_features_totransform:
        print(f'Adding 1hot Feature : {feature_totransform}')
        df_transformed = df[feature_totransform].str.get_dummies().add_prefix(feature_totransform +'_')
        df_target = pd.concat([df_target, df_transformed], axis=1)
        
    return(df_target)

'''
Cette fonction commence par merger les valeurs de toutes les colonnes comprises dans 
categorical_features_tomerge_andtransform  dans une colonne temporaire
Puis elle fait un 1 hot encode du résultat en appelant la fonction add_categorical_features_1hot

df :  dataframe source
df_target : dataframe cible où seront concaténées les nouvelles features créées
categorical_features_tomerge_andtransform : la liste des catégories à merger et à 1 hot encode,
             par exemple: ['actor_1_name', 'actor_2_name', 'actor_3_name']
merged_feature_name : le nom de la feature qui sera mergée
    exemple si le nom est 'actors_names'
    On pourra avoir les colonnes suivantes de créées:  'actors_names_Le nom du 1er acteur', 'actors_names_Le nom du 2eme acteur'
          
'''
def add_categorical_features_merge_and_1hot(df, df_target, categorical_features_tomerge_andtransform, merged_feature_name):
    cnt = 0
    for feature_totransform in categorical_features_tomerge_andtransform:                            
        if (cnt == 0):
            df[merged_feature_name] = df[feature_totransform]
        
        else:
            df[merged_feature_name] = df[merged_feature_name] + '|' + df[feature_totransform]
            
        cnt += 1
        
    return(add_categorical_features_1hot(df, df_target, [merged_feature_name]))

def add_categorical_features_bow_and_1hot(df, df_target, categorical_features_totransform):
    for feature_totransform in categorical_features_totransform:
        print(f'Adding 1hot Feature : {feature_totransform}')
        df_transformed = df[feature_totransform].str.lower().str.replace(r'[^\w\s]', '').str.replace(u'\xa0', u'').str.get_dummies(sep=' ').add_prefix(feature_totransform +'_')
        # \xa0  character present at the end of film titles prevented last character to be catched by dummies
        
        df_target = pd.concat([df_target, df_transformed], axis=1)
        
    return(df_target)            


# In[28]:


df_imputed = add_categorical_features_1hot(df, df_numerical_features_imputed, categorical_features)
df_imputed = add_categorical_features_merge_and_1hot(df, df_imputed, categorical_features_tomerge, 'actors_names' )


# In[29]:


df_imputed.shape


# In[30]:


df_imputed = add_categorical_features_bow_and_1hot(df, df_imputed, categorical_features_tobow)


# In[93]:


pd.set_option('display.max_columns', 100)
df_imputed.head(10)


# ### Comparaison avant/après de quelques valeurs 1hot encoded :

# In[34]:


df[['actor_1_name', 'actor_2_name', 'actor_3_name', 'actors_names', 'country', 'genres']].head(10)


# In[35]:


df_imputed[['actors_names_Johnny Depp', 'actors_names_Orlando Bloom', 'actors_names_Jack Davenport', 'actors_names_Joel David Moore', 'country_USA', 'country_UK', 'genres_Action', 'genres_Adventure']].head(10)


# In[ ]:


df_imputed.loc[0]


# # Construction d'un premier modèle de recommendation

# ## Scaling et réduction de dimensionalité

# In[ ]:


from sklearn import decomposition
from sklearn import preprocessing

n_comp = 200 # Nombre de dimension cible

X = df_imputed.values

features = df_imputed.columns

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca = decomposition.PCA(n_components=n_comp)
pca.fit(X_scaled)


# In[ ]:


X_reduced = pca.transform(X_scaled)


# In[ ]:


X_reduced.shape


# ## Algorithme KNN

# In[ ]:


from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X_reduced)
distances_matrix, reco_matrix = nbrs.kneighbors(X_reduced)


# In[ ]:


distances_matrix.shape


# In[ ]:


reco_matrix.shape


# In[ ]:


reco_matrix


# In[ ]:


print(f"{(df.iloc[0]['movie_title'])}")


# In[ ]:


df[df['movie_title'].str.contains('Nixon')]


# In[51]:


pd.options.display.max_colwidth = 100
df.loc[[3820]]


# In[52]:


df.loc[[1116]]


# ## Affichage d'échantillon de recommandations

# In[15]:


from sklearn.preprocessing import MinMaxScaler

'''
This function returns a similarity score between 0 and 1, for each relevant column 
between df_encoded.loc[index1] and df_encoded.loc[index2]

Relevant column meaning that at least df_encoded.loc[index1][column] or df_encoded.loc[index2][column] is not 0
Which means either film 1 or film 2 has the attribute
'''
def get_similarity_df(df_encoded, index1, index2):
    # Transforming data so that values are between 0 and 1, positive
    scaler = MinMaxScaler() 
    array_scaled = scaler.fit_transform(df_encoded)
    df_scaled  = pd.DataFrame(data=array_scaled , columns=df_encoded.columns, index=df_encoded.index)
    
    # This line of code allows not to keep 1hot columns that are both 0  (for example, both films NOT having word "the" is not relevant :  1hot features are to sparse to keep 0 values like that)
    df_relevant_items = df_scaled[df_scaled.columns[(df_scaled.loc[index1] + df_scaled.loc[index2]) > 0]]
    
    # We substract from 1 because the higher the score, the higher the similarity
    # 1 hot columns that have 0 value as a result mean that 1 and only 1 of the 2 films has the attribute
    # (Those are differenciating attributes, as opposed to attributes that are both 0))
    return(1 - ((df_relevant_items.loc[index1] - df_relevant_items.loc[index2]).abs())).sort_values(ascending=False)

'''
get_similarity_df_scaled_input  does the same as above, except that it does not scale input and does not drop irrelevant features for similarity
It does not do that to optimize performance, so that it can be called from a scikit learn Estimator's score() function

'''
def get_similarity_df_scaled_input(df_scaled, index1, index2):
    # Transforming data so that values are between 0 and 1, positive
    # This function assumes that below code must be run before call
    '''
    scaler = MinMaxScaler() 
    array_scaled = scaler.fit_transform(df_encoded)
    df_scaled  = pd.DataFrame(data=array_scaled , columns=df_encoded.columns, index=df_encoded.index)
    '''
    
    # This function also assumes that you drop all non-relevant items for similarity score before call, like this :
    '''
    df_scaled.drop(labels=['movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes', 'imdb_score', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'num_user_for_reviews', 'actor_3_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes'], axis=1, inplace=True)
    '''
    
    # This line of code allows not to keep 1hot columns that are both 0  (for example, both films NOT having word "the" is not relevant :  1hot features are to sparse to keep 0 values like that)
    df_relevant_items = df_scaled[df_scaled.columns[(df_scaled.loc[index1] + df_scaled.loc[index2]) > 0]]
    
    # We substract from 1 because the higher the score, the higher the similarity
    # 1 hot columns that have 0 value as a result mean that 1 and only 1 of the 2 films has the attribute
    # (Those are differenciating attributes, as opposed to attributes that are both 0))
    return(1 - ((df_relevant_items.loc[index1] - df_relevant_items.loc[index2]).abs())).sort_values(ascending=False)


# In[16]:


def afficher_recos(film_index, reco_matrix, df_encoded, with_similarity_display=False):
    print(f"> Film choisi : {(df.loc[film_index]['movie_title'])} - imdb score : {df.loc[film_index]['imdb_score']} - {df.loc[film_index]['movie_imdb_link']}")
          
    print(f"\nFilms recommandés : ")
    for nb_film in range(5):
        print(f"{df.loc[reco_matrix[film_index, nb_film+1]]['movie_title']} - imdb score : {df.loc[reco_matrix[film_index, nb_film+1]]['imdb_score']} - {df.loc[reco_matrix[film_index, nb_film+1]]['movie_imdb_link']}")
        if (with_similarity_display == True):
            print(f'ID film recommandé = {reco_matrix[film_index, nb_film + 1]}')
            print('\nTableau de similarités (1 = point commun catégoriel.  0 = élément catégoriel différenciant)')
            print(get_similarity_df(df_encoded, film_index, reco_matrix[film_index, nb_film + 1] ))
            print('\n')
              
    print("\n")


# In[17]:


def afficher_recos_films(reco_matrix, df_encoded, with_similarity_display=False, films_temoins_indexes=[2703, 0, 3, 4820, 647, 124, 931, 1172, 3820]):
    for film_temoin_index in films_temoins_indexes:
        afficher_recos(film_temoin_index, reco_matrix, df_encoded, with_similarity_display=with_similarity_display)


# In[58]:


df.loc[2703]['country']


# In[59]:


afficher_recos_films(reco_matrix, df_imputed)


# In[60]:


afficher_recos_films(reco_matrix, df_imputed, with_similarity_display=True)


# In[46]:


df[df['movie_title'].str.contains('night')]['movie_title']


# In[62]:


df.shape[0]


# In[63]:


df[['movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes', 'imdb_score' , 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'facenumber_in_poster', 'duration', 'num_user_for_reviews', 'actor_3_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'budget', 'gross','title_year']]


# In[64]:


df[['movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes', 'imdb_score' , 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'facenumber_in_poster', 'duration', 'num_user_for_reviews', 'actor_3_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'budget', 'gross','title_year']].loc[4]


# # Industralisation du modèle avec Pipeline, et métriques scoring + prédiction imdb

# ## Définition des Pipelines et métriques

# In[18]:


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

'''
kmeans_transformer_pipeline = Pipeline([
    ('kmeans', KMeans(n_clusters=10)),
    #('pipeline_final', PipelineFinal()),

])
'''


# ## Lancement du pipeline preparation

# In[19]:


# Récupération des étiquettes de scoring :

# D'abord, dropper les duplicates pour que les index de df soient alignés avec ceux de df_encoded (qui a déjà fait l'objet d'un drop duplicates dans le pipeline)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

labels = df['imdb_score'].to_numpy()


# In[20]:


df_encoded = preparation_pipeline.fit_transform(df)


# ## Recherche multiple de paramètres avec métrique similarité

# In[21]:


if (RECOMPUTE_GRIDSEARCH == True):
    from sklearn.model_selection import GridSearchCV

    recommendation_pipeline_KNN = Pipeline([
        ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
        ('standardscaler', preprocessing.StandardScaler()),
        ('reduction', DimensionalityReduction_Transform(reduction_params = {'reduction_type' : 'PCA', 'n_components' : 200 })),
        ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    ])


    # Erreur avec la distance mahalanobis : ValueError: Must provide either V or VI for Mahalanobis distance

    param_grid = {
            'features_droper__features_todrop':  [#None,
                                                  #['imdb_score'],
            
                ## Drop de tout ce qui est en haut à droite du plan factoriel :
                #['title_year', 'cast_total_facebook_likes', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes'],

                # Drop de tout ce qui est en bas à droite sur le plan factoriel :
                ['movie_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'num_user_for_reviews', 'num_voted_users', 'duration', 'imdb_score'],
                
                # Conservation de cast_total_facebook_likes (en haut à droite) + imdb_score (en bas à droite) + les features OHE
                ['title_year', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'movie_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'num_user_for_reviews', 'num_voted_users', 'duration'],
                
            ],

            'reduction__reduction_params': [{'reduction_type' : 'NCA', 'n_components' : 200 },
                                            #{'reduction_type' : 'NCA', 'n_components' : 150 },
                                            {'reduction_type' : 'NCA', 'n_components' : 100 },
                                            {'reduction_type' : 'NCA', 'n_components' : 10 },
                                            {'reduction_type' : 'PCA', 'n_components' : 200 },
                                            #{'reduction_type' : 'PCA', 'n_components' : 150 },
                                            {'reduction_type' : 'PCA', 'n_components' : 100 },
                                            {'reduction_type' : 'PCA', 'n_components' : 10 },
                                            

            ],


            'KNN__knn_params': [{'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'}, 
                                {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'manhattan'}, 
                                {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'euclidean'}, 
            ],
        }

    grid_search = GridSearchCV(recommendation_pipeline_KNN, param_grid, cv=5, verbose=2, error_score=np.nan)
    grid_search.fit(df_encoded, labels)

    


# ### Sauvegarde et restauration des résultats

# In[35]:


if (SAVE_GRID_RESULTS == True):
    #df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
    #df_grid_search_results.to_csv(GRIDSEARCH_CSV_FILE)

    df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["mean_test_score"])],axis=1)
    df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["std_test_score"], columns=["std_test_score"])],axis=1)
    df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_fit_time"], columns=["mean_fit_time"])],axis=1)
    df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_score_time"], columns=["mean_score_time"])],axis=1)
    df_grid_search_results.to_csv(GRIDSEARCH_CSV_FILE)
    
    with open(GRIDSEARCH_PICKLE_FILE, 'wb') as f:
        pickle.dump(grid_search, f, pickle.HIGHEST_PROTOCOL)
        
if (LOAD_GRID_RESULTS == True):
    if ((SAVE_GRID_RESULTS == True) or (RECOMPUTE_GRIDSEARCH == True)):
        print('Error : if want to load grid results, you should not have saved them or recomputed them before, or you will loose all your training data')
        
    else:
        with open(GRIDSEARCH_PICKLE_FILE, 'rb') as f:
            grid_search = pickle.load(f)
           
        df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["mean_test_score"])],axis=1)
        df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["std_test_score"], columns=["std_test_score"])],axis=1)
        df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_fit_time"], columns=["mean_fit_time"])],axis=1)
        df_grid_search_results = pd.concat([df_grid_search_results,pd.DataFrame(grid_search.cv_results_["mean_score_time"], columns=["mean_score_time"])],axis=1)


# In[23]:


if ((LOAD_GRID_RESULTS == True) or (RECOMPUTE_GRIDSEARCH == True)):
    display(grid_search.cv_results_)


# In[34]:


if ((LOAD_GRID_RESULTS == True) or (RECOMPUTE_GRIDSEARCH == True)):
    display(df_grid_search_results.sort_values(by=['mean_test_score'], ascending=False))


# ### Définition d'une fonction de sauvegarde / chargement résultats pour les prochains tests

# In[40]:


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
            
        return(df_grid_search_results)

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
            
            return(df_grid_search_results)


# ### On complète la recherche multiple précédente avec des combinatoires de features drop supplémentaires

# In[39]:


if (RECOMPUTE_GRIDSEARCH == True):
    from sklearn.model_selection import GridSearchCV

    recommendation_pipeline_KNN = Pipeline([
        ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
        ('standardscaler', preprocessing.StandardScaler()),
        ('reduction', DimensionalityReduction_Transform(reduction_params = {'reduction_type' : 'PCA', 'n_components' : 200 })),
        ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    ])


    # Erreur avec la distance mahalanobis : ValueError: Must provide either V or VI for Mahalanobis distance

    param_grid = {
            'features_droper__features_todrop':  [None,
                                                  ['imdb_score'],
            
                ## Drop de tout ce qui est en haut à droite du plan factoriel :
                #['title_year', 'cast_total_facebook_likes', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes'],

                # Drop de tout ce qui est en bas à droite sur le plan factoriel :
                #['movie_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'num_user_for_reviews', 'num_voted_users', 'duration', 'imdb_score'],
                
                # Conservation de cast_total_facebook_likes (en haut à droite) + imdb_score (en bas à droite) + les features OHE
                #['title_year', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'movie_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'num_user_for_reviews', 'num_voted_users', 'duration'],
                
            ],

            'reduction__reduction_params': [{'reduction_type' : 'NCA', 'n_components' : 200 },
                                            #{'reduction_type' : 'NCA', 'n_components' : 150 },
                                            {'reduction_type' : 'NCA', 'n_components' : 100 },
                                            {'reduction_type' : 'NCA', 'n_components' : 10 },
                                            {'reduction_type' : 'PCA', 'n_components' : 200 },
                                            #{'reduction_type' : 'PCA', 'n_components' : 150 },
                                            {'reduction_type' : 'PCA', 'n_components' : 100 },
                                            {'reduction_type' : 'PCA', 'n_components' : 10 },
                                            

            ],


            'KNN__knn_params': [{'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'}, 
                                {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'manhattan'}, 
                                {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'euclidean'}, 
            ],
        }

    grid_search = GridSearchCV(recommendation_pipeline_KNN, param_grid, cv=5, verbose=2, error_score=np.nan)
    grid_search.fit(df_encoded, labels)

    


# In[43]:


df_grid_search_results_new = save_or_load_search_params(grid_search, 'otherfeats_20200229')


# => En combinant les résultats obtenus par les 2 recherches ci-dessus (sauvegardées dans les fichiers grid_search_results.csv et grid_search_results_otherfeats_20200229.csv, que l'on a mergées ensemble dans grid_search_results_similarity_metric.csv), on obtient que le meilleur estimateur pour la métrique similarité des films est :  
# knn_params : {'n_neighbors': 6, 'algorithm': 'ball_tree', 'metric': 'manhattan'}  
# features_todrop : ['imdb_score']  
# reduction_params : {'reduction_type': 'NCA', 'n_components': 100}  
# 

# ## Recherche multiple de paramètres avec métrique score IMDB

# In[ ]:


if (RECOMPUTE_GRIDSEARCH == True):
    from sklearn.model_selection import GridSearchCV

    recommendation_pipeline_KNN = Pipeline([
        ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
        ('standardscaler', preprocessing.StandardScaler()),
        ('reduction', DimensionalityReduction_Transform(reduction_params = {'reduction_type' : 'PCA', 'n_components' : 200 })),
        ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    ])


    # Erreur avec la distance mahalanobis : ValueError: Must provide either V or VI for Mahalanobis distance
    
    param_grid = {
            'features_droper__features_todrop':  [None,
                                                  ['imdb_score'],
            
                ## Drop de tout ce qui est en haut à droite du plan factoriel :
                #['title_year', 'cast_total_facebook_likes', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes'],

                # Drop de tout ce qui est en bas à droite sur le plan factoriel :
                ['movie_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'num_user_for_reviews', 'num_voted_users', 'duration', 'imdb_score'],
                
                # Conservation de cast_total_facebook_likes (en haut à droite) + imdb_score (en bas à droite) + les features OHE
                ['title_year', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'movie_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'num_user_for_reviews', 'num_voted_users', 'duration'],
                
            ],

            'reduction__reduction_params': [{'reduction_type' : 'NCA', 'n_components' : 200 },
                                            #{'reduction_type' : 'NCA', 'n_components' : 150 },
                                            {'reduction_type' : 'NCA', 'n_components' : 100 },
                                            {'reduction_type' : 'NCA', 'n_components' : 10 },
                                            {'reduction_type' : 'PCA', 'n_components' : 200 },
                                            #{'reduction_type' : 'PCA', 'n_components' : 150 },
                                            {'reduction_type' : 'PCA', 'n_components' : 100 },
                                            {'reduction_type' : 'PCA', 'n_components' : 10 },
                                            

            ],


            'KNN__knn_params': [{'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'}, 
                                {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'manhattan'}, 
                                {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'euclidean'}, 
            ],
        }

    grid_search = GridSearchCV(recommendation_pipeline_KNN, param_grid, cv=5, verbose=2, error_score=np.nan, scoring='neg_mean_squared_error')
    grid_search.fit(df_encoded, labels)

    


# ### Sauvegarde et restauration des résultats

# In[ ]:


df_grid_search_results_new = save_or_load_search_params(grid_search, 'metric_imdb_20200229')


# => En combinant les résultats obtenus par les 2 recherches ci-dessus (sauvegardées dans les fichiers grid_search_results.csv et grid_search_results_otherfeats_20200229.csv, que l'on a mergées ensemble dans grid_search_results_similarity_metric.csv), on obtient que le meilleur estimateur pour la métrique similarité des films est :  
# knn_params : {'n_neighbors': 6, 'algorithm': 'ball_tree', 'metric': 'manhattan'}  
# features_todrop : ['imdb_score']  
# reduction_params : {'reduction_type': 'NCA', 'n_components': 100}  
# 

# ## Affichage de recommendations avec NCA_KNN n_components 100 (Meilleur modèle trouvé ci-dessus) :

# In[44]:


recommendation_pipeline_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('reduction', DimensionalityReduction_Transform(reduction_params = {'reduction_type': 'NCA', 'n_components': 100} )),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'manhattan'})),
])

recommendation_pipeline_KNN.fit(df_encoded, labels)


# In[45]:


distances_matrix, reco_matrix = recommendation_pipeline_KNN.transform(df_encoded)


# In[46]:


afficher_recos_films(reco_matrix, df_encoded)


# In[47]:


afficher_recos_films(reco_matrix, df_encoded, with_similarity_display=True)


# ## Affichage de recommendations avec NCA_KNN n_components 200 (Meilleur modèle trouvé avec examen humain des résultats) :

# Ce modèle correspond au 7ème meilleur avec la métrique similarité

# In[48]:


recommendation_pipeline_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('reduction', DimensionalityReduction_Transform(reduction_params = {'reduction_type': 'NCA', 'n_components': 200} )),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
])

recommendation_pipeline_KNN.fit(df_encoded, labels)


# In[49]:


distances_matrix, reco_matrix = recommendation_pipeline_KNN.transform(df_encoded)


# In[50]:


afficher_recos_films(reco_matrix, df_encoded)


# ### => Avec un examen humain, les résultats semblent meilleurs avec NCA qu'avec PCA,  et également meilleurs avec NCA 200 qu'avec NCA 100 :
# ### Par exemple avec le 6ème sens on obtient plus de films avec le thème de la mort et les enfants

# ### Affichage des similarités des recos

# In[51]:


afficher_recos_films(reco_matrix, df_encoded, with_similarity_display=True)


# ## Affichage de recommendations pour NCA_KNN avec n_components 5 (le pire modèle trouvé jusqu'à présent) : 

# In[79]:


recommendation_pipeline_NCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('NCA', NCATransform(nca_params =  {'random_state':42, 'n_components':5 })),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    #('pipeline_final', PipelineFinal()),
])

distances_matrix, reco_matrix = recommendation_pipeline_NCA_KNN.fit_transform(df_encoded, labels)


# In[80]:


afficher_recos_films(reco_matrix, df_encoded)


# ### => Les résultats sont clairement moins bons avec NCA components à 5  suivi de KNN : les imdb scores sont meilleurs mais les catégories sont moins bien capturées
# ### => Pourtant, l'erreur de prédiction de l'imdb score n'est que de 0.33 avec NBCA components = 5,  soit beaucoup moins qu'avec components = 200  :  cela montre que la métrique de prédiction de l'imdb score n'est pas parfaite:  avec cette métrique, il ne faut pas que l'erreur soit trop faible (sinon, on ne capture plus que l'imdb score, et pas grand chose d'autre) ni trop élevée (sinon le modèle n'est plus pertinent)

# In[81]:


afficher_recos_films(reco_matrix, df_encoded, with_similarity_display=True)


# ## Affichage de recommendations avec NCA KNN , n_components à 150

# In[107]:


recommendation_pipeline_NCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('NCA', NCATransform(nca_params =  {'random_state':42, 'n_components':150 })),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    #('pipeline_final', PipelineFinal()),
])


# In[109]:


distances_matrix, reco_matrix = recommendation_pipeline_NCA_KNN.fit_transform(df_encoded, labels)


# In[110]:


afficher_recos_films(reco_matrix, df_encoded)


# ### => Malgré un meilleur score de prediction de ce modèle par rapport à celui à 200 composants, je suis moins convaincu par ce modèle. Par exemple : pour Matrix, il ne parvient pas à capturer un autre film de la trilogie.   Pour le 6ème sens, les recommendations sont moins dans le thème du film.

# In[112]:


afficher_recos_films(reco_matrix, df_encoded, with_similarity_display=True)


# ## Affichage de recommandations avec PCA (5 components) + KNN

# In[89]:


recommendation_pipeline_PCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('pca', decomposition.PCA(n_components=5)),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    #('pipeline_final', PipelineFinal()),
])

distances_matrix, reco_matrix = recommendation_pipeline_PCA_KNN.fit_transform(df_encoded, labels)


# In[90]:


afficher_recos_films(reco_matrix, df_encoded)


# In[91]:


afficher_recos_films(reco_matrix, df_imputed, with_similarity_display=True)


# ## Affichage de recommandations avec PCA (200 components) + KNN

# In[94]:


recommendation_pipeline_PCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('pca', decomposition.PCA(n_components=200)),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    #('pipeline_final', PipelineFinal()),
])

distances_matrix, reco_matrix = recommendation_pipeline_PCA_KNN.fit_transform(df_encoded, labels)


# In[95]:


afficher_recos_films(reco_matrix, df_encoded)


# # Visualisation des films avec KMeans

# In[96]:


df_reduced = reducer_pipeline.fit_transform(df_encoded, labels)


# ## Tentative de KMeans sur les données non réduites (22000 dimensions)

# In[97]:


X = df_encoded.values

features = df_encoded.columns

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)


# Affichage des coefs. de silhouette pour différentes valeurs de k :

# In[99]:


from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_scaled)
                for k in range(1, 10)]
#inertias = [model.inertia_ for model in kmeans_per_k]


# In[100]:


silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[101]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# => Mauvais résultat (scores négatifs)

# ## Tentative de KMeans sur données réduites à 200 dimensions avec NCA

# In[102]:


kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(df_reduced)
                for k in range(1, 50)]


# In[109]:


silhouette_scores = [silhouette_score(df_reduced, model.labels_)
                     for model in kmeans_per_k[1:]]


# In[110]:


plt.figure(figsize=(8, 3))
plt.plot(range(2, 50), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
#plt.axis([1.8, 8.5, 0.55, 0.7]) # [xmin, xmax, ymin, ymax]
#save_fig("silhouette_score_vs_k_plot")
plt.show()


# => Résultat : instable

# ## Tentative de KMeans distance cosine, k=10

# In[84]:


from nltk.cluster import KMeansClusterer
import nltk

k = 200 
model_kmeans = KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance, avoid_empty_clusters=True, repeats=10)      
get_ipython().run_line_magic('time', 'cluster = model_kmeans.cluster(X_scaled, assign_clusters = True)')


# In[87]:


get_ipython().run_line_magic('time', 'cluster_labels = np.array([ model_kmeans.classify(X_scaled_instance) for X_scaled_instance in X_scaled ], np.short)')
get_ipython().run_line_magic('time', "sklearn_silhouette_avg = silhouette_score(X_scaled, cluster_labels, metric='cosine')")


# In[90]:


sklearn_silhouette_avg


# **Visualisation avec TSNE**

# In[145]:


import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from yellowbrick.text import TSNEVisualizer # FUTURE WARNING : Due to yellowbrick imports we currently have a "FutureWarning" : he sklearn.metrics.classification module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.metrics. Anything that cannot be imported from sklearn.metrics is now part of the private API.

my_cmap = plt.matplotlib.colors.ListedColormap(['#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#9A6324','#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff','#000000', '#fffac8'])                                                        
tsne = TSNEVisualizer(metric='cosine', decompose='pca', colormap=my_cmap)

tsne.fit(X_scaled, ["cluster{}".format(c) for c in cluster_labels])
tsne.poof()


# In[91]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_reduced_n2[:, 0], y = X_reduced_n2[:, 1],
                    name = 'Films',
                    mode = 'markers',
                    marker=dict(color=cluster_labels),
                    text = df['movie_title']
                    )


layout = go.Layout(title = 'Films clustered',
                   hovermode = 'closest')

fig = go.Figure(data = [trace_1], layout = layout)

py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot.html') 


# ## Lancement d'un clustering, et visualisation

# In[127]:


from sklearn.cluster import KMeans

kmeans_clusterer = KMeans(n_clusters=19, random_state=42)

clusters = kmeans_clusterer.fit_transform(df_reduced)


# In[128]:


clusters


# ## Réduction de dimensionalité à 2 et 3 pour la visualisation

# In[130]:


X = df_encoded.values

features = df_encoded.columns

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca = decomposition.PCA(n_components=2)
X_reduced_n2 = pca.fit_transform(X_scaled)

# Calcul des composantes principales
pca = decomposition.PCA(n_components=3)
X_reduced_n3 = pca.fit_transform(X_scaled)


# In[114]:


X_reduced_n2[:, 1].shape


# In[115]:


clusters.shape


# In[116]:


kmeans_clusterer.labels_[800]


# In[131]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_reduced_n2[:, 0], y = X_reduced_n2[:, 1],
                    name = 'Films',
                    mode = 'markers',
                    marker=dict(color=kmeans_clusterer.labels_),
                    text = df['movie_title']
                    )


layout = go.Layout(title = 'Films clustered',
                   hovermode = 'closest')

fig = go.Figure(data = [trace_1], layout = layout)

py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot.html') 


# In[57]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)

'''
trace_1 = go.Scatter(x = X_reduced_n2[:, 0], y = X_reduced_n2[:, 1],
                    name = 'Films',
                    mode = 'markers',
                    marker=dict(color=kmeans_clusterer.labels_),
                    text = df['movie_title']
                    )
'''

trace_1 = go.Scatter3d(x = X_reduced_n3[:, 0], y = X_reduced_n3[:, 1], z = X_reduced_n3[:, 2],
                    name = 'Films',
                    mode = 'markers',
                    marker=dict(color=kmeans_clusterer.labels_),
                    text = df['movie_title']
                    )

layout = go.Layout(title = 'Films clustered',
                   hovermode = 'closest')

fig = go.Figure(data = [trace_1], layout = layout)

py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot.html') 


# # Visualisation des fims colorés selon leur imdb_score

# In[95]:





# In[102]:


imdb_score_cat.to_list()


# In[123]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_reduced_n2[:, 0], y = X_reduced_n2[:, 1],
                    name = 'Films',
                    mode = 'markers',
                    marker=dict(color=imdb_score_cat.to_list()),
                    text = df['movie_title']
                    )


layout = go.Layout(title = 'Films clustered',
                   hovermode = 'closest')

fig = go.Figure(data = [trace_1], layout = layout)

py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_imdb.html') 


# # Visualisation avec réduction dimensionnelle à partir du DataFrame d'origine (features numériques seulement)

# In[24]:


X_numerical = df_numerical_features_imputed.values

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X_numerical)
X_numerical_scaled = std_scale.transform(X_numerical)

# Calcul des composantes principales
pca = decomposition.PCA(n_components=2)
X_numerical_reduced = pca.fit_transform(X_numerical_scaled)


# In[25]:


imdb_score_cat = pd.cut(df_numerical_features_imputed['imdb_score'], bins=[-np.inf,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# In[26]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_numerical_reduced[:, 0], y = X_numerical_reduced[:, 1],
                    name = 'Films',
                    mode = 'markers',
                    marker=dict(color=cluster_labels),
                    text = df['movie_title']
                    )


layout = go.Layout(title = 'Films clustered',
                   hovermode = 'closest')

fig = go.Figure(data = [trace_1], layout = layout)

py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_imdb.html') 


# In[26]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)


trace_1 = go.Scatter(x = X_numerical_reduced[:, 0], y = X_numerical_reduced[:, 1],
                    name = 'Films',
                    mode = 'markers',
                    marker=dict(color=cluster_labels),
                    text = df['movie_title']
                    )


layout = go.Layout(title = 'Films clustered',
                   hovermode = 'closest')

fig = go.Figure(data = [trace_1], layout = layout)

py.offline.iplot(fig) # Display in the notebook works with jupyter notebook, but not with jupyter lab

py.offline.plot(fig, filename='clusters_plot_imdb.html') 


# # Sauvegarde du modèle retenu pour l'API

# In[52]:


if (SAVE_API_MODEL == True):
    # Normalement les doublons ont déjà été droppés, mais on refait ce code au cas où :
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    API_model = {'reco_matrix' : reco_matrix, 'movie_names' : df['movie_title'].tolist()}
    
    with open(API_MODEL_PICKLE_FILE, 'wb') as f:
        pickle.dump(API_model, f, pickle.HIGHEST_PROTOCOL)    


# ## Tentative de KNN avec cosine distance

# In[ ]:


recommendation_pipeline = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('pca', decomposition.PCA(n_components=200)),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'cosine'})),
    #('pipeline_final', PipelineFinal()),
])

recommendation_pipeline.fit(df_encoded, labels)
predictions = recommendation_pipeline.predict(df_encoded)


# In[ ]:


from sklearn import neighbors

sorted(neighbors.VALID_METRICS['brute'])


# In[ ]:


recommendation_pipeline_NMF.fit(df_encoded, labels)


# In[ ]:







>>> model = NMF(n_components=2, init='random', random_state=0)
>>> W = model.fit_transform(X)
>>> H = model.components_


# In[ ]:





# In[ ]:





# In[ ]:


# Pour afficher uniquement les recos pour certains films  (à utiliser pour l'API)
#(distances_matrix, reco_matrix) = recommendation_pipeline.transform(df_encoded.loc[np.r_[0:5]])

# Pour calculer les recos pour tous les films :
(distances_matrix, reco_matrix) = recommendation_pipeline.transform(df_encoded)


# In[ ]:





# In[ ]:


reco_matrix


# In[ ]:


df[df['movie_title'].str.contains('Vampire')]


# movie_title                  1.000000   => La distance entre chaque valeur devra être une distance de chaîne de caractère.   Mais comment faire un algo de clustering qui ne calcule pas la distance de la même façon pour cet attribut là que pour les autres ?  Faire un vecteur one hot avec le nombre distinct de titres de films dedans, et réduire sa dimensionalité  ?

# # Annexe : inutile, à effacer plus tard

# In[ ]:


'''
df_transformed = df.copy(deep=True)

for feature_totransform in categorical_features_tosplit_andtransform:
    all_features = []

    for i, row in df.iterrows():
        if (type(row[feature_totransform]) == str):        
            features_list = row[feature_totransform].split(sep='|')
            for feature_name in features_list:
                all_features.append(feature_totransform+'_'+feature_name)

    all_features = list(set(all_features))

    for feature_name in all_features:
        df_transformed[feature_name] = 0


    for i, row in df.iterrows():
        if (type(row[feature_totransform]) == str):        
            features_list = row[feature_totransform].split(sep='|')
            for feature_name in features_list:
                df_transformed.at[i, feature_totransform+'_'+feature_name] = 1
'''


# In[ ]:


'''
df_transformed = df.copy(deep=True)

for feature_totransform in categorical_features_tosplit_andtransform:
    for i, row in df.iterrows():
        if (type(row[feature_totransform]) == str):        
            features_list = row[feature_totransform].split(sep='|')
            for feature_name in features_list:
                df_transformed.at[i, feature_totransform+'_'+feature_name] = 1
'''


# ## Ancien code avant pipelines qui a permis d'obtenir qq résultats

# In[ ]:


#pd.cut(df_labels[0], bins=[1, 2, 3, 4,5,6,7,8,9,10], right=True)
df_labels =  pd.DataFrame(data=labels)
labels_discrete = pd.cut(df_labels[0], bins=range(1,10), right=True).astype(str).tolist()


# In[ ]:


features_droper = FeaturesDroper(features_todrop=['imdb_score'])
standardscaler = preprocessing.StandardScaler()

X_res = features_droper.fit_transform(df_encoded)
X_res = standardscaler.fit_transform(X_res)

#hparam_n_components = [10, 50, 100, 200, 300, 500]
hparam_n_components = [2, 5, 8, 9]

for hparam in hparam_n_components:
    nca = NeighborhoodComponentsAnalysis(random_state=42, n_components=hparam)
    X_reduced = nca.fit_transform(X_res, labels_discrete)
    KNN = KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})
    KNN.fit(X_reduced, labels)
    predictions = KNN.predict(X_reduced)
    print('Avec n_components = ' + str(hparam))
    print_rmse(labels, predictions)


# In[ ]:


features_droper = FeaturesDroper(features_todrop=['imdb_score'])
standardscaler = preprocessing.StandardScaler()

X_res = features_droper.fit_transform(df_encoded)
X_res = standardscaler.fit_transform(X_res)

#hparam_n_components = [10, 50, 100, 200, 300, 500]
hparam_n_components = [10, 50, 100, 200, 300, 500]

for hparam in hparam_n_components:
    nca = NeighborhoodComponentsAnalysis(random_state=42, n_components=hparam)
    X_reduced = nca.fit_transform(X_res, labels_discrete)
    KNN = KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})
    KNN.fit(X_reduced, labels)
    predictions = KNN.predict(X_reduced)
    print('Avec n_components = ' + str(hparam))
    print_rmse(labels, predictions)


# # Annexe 2 : ancien code de pipelines

# ## Tests avec PCA_KNN et la métrique similarité :

# In[25]:


recommendation_pipeline_PCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('pca', decomposition.PCA(n_components=200)),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    #('pipeline_final', PipelineFinal()),
])

recommendation_pipeline_PCA_KNN.fit(df_encoded, labels, KNN__df_encoded = df_encoded)
scores = recommendation_pipeline_PCA_KNN.score(df_encoded)


# In[26]:


scores


# In[58]:


if (EXECUTE_INTERMEDIATE_MODELS == True) :
    recommendation_pipeline_PCA_KNN = Pipeline([
        ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
        ('standardscaler', preprocessing.StandardScaler()),
        ('pca', decomposition.PCA(n_components=200)),
        ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
        #('pipeline_final', PipelineFinal()),
    ])

    recommendation_pipeline_PCA_KNN.fit(df_encoded, labels)
    predictions = recommendation_pipeline_PCA_KNN.predict(df_encoded)


# In[31]:


if (EXECUTE_INTERMEDIATE_MODELS == True):
    print_rmse(labels, predictions)


# Résultat : Erreur moyenne de prédiction de l'IMDB score: 1.0531753089075517

# ## Tests avec PCA_KNN et la métrique scoring imdb

# In[12]:


recommendation_pipeline_PCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('pca', decomposition.PCA(n_components=200)),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    #('pipeline_final', PipelineFinal()),
])

recommendation_pipeline_PCA_KNN.fit(df_encoded, labels)
predictions = recommendation_pipeline_PCA_KNN.predict(df_encoded)


# ### Résultat de tests effectués avec recommendation_pipeline_PCA_KNN, et n_components à 200 :
#  #### Erreur moyenne de prédiction de l'IMDB score avec la variable de scoring dans le JDD : 1.0095843224821195 
#  #### Erreur moyenne de prédiction de l'IMDB score sans la variable de scoring dans le JDD : 1.0486754159658844
# 

# ## Recherche de paramètres PCA (5, 150, 200) + KNN, métrique similarité

# In[71]:


#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

recommendation_pipeline_PCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('pca', decomposition.PCA(n_components=200)),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
])


# Erreur avec la distance mahalanobis : ValueError: Must provide either V or VI for Mahalanobis distance

param_grid = {
        'features_droper__features_todrop':  [#None,
                                              ['imdb_score'],
                                    
        ],

        'pca__n_components': [5, 150, 200],

        'KNN__knn_params': [{'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'}, 
        ],
    }

grid_search = GridSearchCV(recommendation_pipeline_PCA_KNN, param_grid, cv=5, verbose=2, error_score=np.nan)
grid_search.fit(df_encoded, labels)


# In[72]:


df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)


# In[73]:


#df_grid_search_results.to_csv('gridresult_temp.csv')  # Uncomment to save in CSV file


# In[74]:


grid_search.best_estimator_


# In[76]:


df_grid_search_results.to_csv('gridsearch_PCA_results.csv')


# In[75]:


qgrid_show(df_grid_search_results)


# Résultats :  
#     KNN__knn_params	features_droper__features_todrop	pca__n_components	Accuracy  
# {'n_neighbors': 6, 'algorithm': 'ball_tree', 'metric': 'minkowski'}	['imdb_score']	5	8.492163788152574  
# {'n_neighbors': 6, 'algorithm': 'ball_tree', 'metric': 'minkowski'}	['imdb_score']	150	8.214512311933097  
# {'n_neighbors': 6, 'algorithm': 'ball_tree', 'metric': 'minkowski'}	['imdb_score']	200	8.04523852877884  
# 
# => Ces résultats restent contradictoires avec l'examen humain qui montre que nb_components = 5 n'est pas adapté (alors qu'ici, le score de similarité moyen est meilleur)  
# 
# Après avoir modifié la fonction de comparaison des similarités pour enlever de nombreuses features qui ne concernent pas la similarité des items car ce ne sont pas des propriétés des films en eux mêmes (nb facebook likes,  nb votes, ....), on obtient toujours le même classement

# In[43]:


from sklearn.metrics import mean_squared_error

def print_rmse(labels, predictions):
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    print(f"Erreur moyenne de prédiction de l'IMDB score: {rmse}")


# ## Recherche de paramètres NCA (5, 150, 200) + KNN, métrique imdb

# In[78]:


#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

recommendation_pipeline_NCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('NCA', NCATransform(nca_params =  {'random_state':42, 'n_components':200 })),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
])

# Erreur avec la distance mahalanobis : ValueError: Must provide either V or VI for Mahalanobis distance

param_grid = {
        'features_droper__features_todrop':  [#None,
                                              ['imdb_score'],
                                    
        ],

        'NCA__nca_params': [{'random_state':42, 'n_components':5 }, {'random_state':42, 'n_components':150 }, {'random_state':42, 'n_components':200 }],

        'KNN__knn_params': [{'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'}, 
        ],
    }

grid_search = GridSearchCV(recommendation_pipeline_NCA_KNN, param_grid, cv=5, verbose=2, error_score=np.nan, scoring='neg_mean_squared_error')
grid_search.fit(df_encoded, labels)


# In[80]:


grid_search.best_estimator_


# In[82]:


df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)


# In[92]:


df_grid_search_results


# ## Recherche de paramètres NCA (5, 150, 200) + KNN, métrique similarité

# In[79]:


#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

recommendation_pipeline_NCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('NCA', NCATransform(nca_params =  {'random_state':42, 'n_components':200 })),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
])

# Erreur avec la distance mahalanobis : ValueError: Must provide either V or VI for Mahalanobis distance

param_grid = {
        'features_droper__features_todrop':  [#None,
                                              ['imdb_score'],
                                    
        ],

        'NCA__nca_params': [{'random_state':42, 'n_components':5 }, {'random_state':42, 'n_components':150 }, {'random_state':42, 'n_components':200 }],

        'KNN__knn_params': [{'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'}, 
        ],
    }

grid_search_sim = GridSearchCV(recommendation_pipeline_NCA_KNN, param_grid, cv=5, verbose=2, error_score=np.nan)
grid_search_sim.fit(df_encoded, labels)


# In[81]:


grid_search_sim.best_estimator_


# In[86]:


df_grid_search_sim_results = pd.concat([pd.DataFrame(grid_search_sim.cv_results_["params"]),pd.DataFrame(grid_search_sim.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)


# In[93]:


df_grid_search_sim_results


# 
# ## Recherche de différents paramètres avec NCA + KNN

# In[103]:


'''
if (RECOMPUTE_GRIDSEARCH == True):
    #from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import GridSearchCV

    recommendation_pipeline_NCA_KNN = Pipeline([
        ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
        ('standardscaler', preprocessing.StandardScaler()),
        ('NCA', NCATransform(nca_params =  {'random_state':42, 'n_components':200 })),
        ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
        #('pipeline_final', PipelineFinal()),
    ])


    # Erreur avec la distance mahalanobis : ValueError: Must provide either V or VI for Mahalanobis distance

    param_grid = {
            'features_droper__features_todrop':  [#None,
                                                  ['imdb_score'],
                                                  ['imdb_score', 'actor_1_facebook_likes','actor_2_facebook_likes', 'actor_3_facebook_likes', 'num_user_for_reviews', 'num_critic_for_reviews', 'num_voted_users']                                              
            ],

            'NCA__nca_params': [{'random_state':42, 'n_components':50 }, {'random_state':42, 'n_components':100 }, 
                                {'random_state':42, 'n_components':150 }, {'random_state':42, 'n_components':200 }
            ],

            'KNN__knn_params': [{'n_neighbors':6, 'algorithm':'kd_tree', 'metric':'manhattan'},
                                {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'manhattan'}, 
                                {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'}, 
                                {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'euclidean'}, 
                                {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'jaccard'}, 
                                #{'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'mahalanobis'}, 
                                {'n_neighbors':6, 'algorithm':'kd_tree', 'metric':'minkowski'},
                                {'n_neighbors':6, 'algorithm':'brute', 'metric':'minkowski'},
            ],
        }

    grid_search = GridSearchCV(recommendation_pipeline_NCA_KNN, param_grid,scoring='neg_mean_squared_error', cv=2, verbose=2, error_score=np.nan)
    grid_search.fit(df_encoded, labels)
'''


# ### Sauvegarde et restauration des résultats

# In[104]:


'''
if (SAVE_GRID_RESULTS == True):
    df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
    df_grid_search_results.to_csv(GRIDSEARCH_CSV_FILE)
    
    with open(GRIDSEARCH_PICKLE_FILE, 'wb') as f:
        pickle.dump(grid_search.cv_results_, f, pickle.HIGHEST_PROTOCOL)
        
if (LOAD_GRID_RESULTS == True):
    if ((SAVE_GRID_RESULTS == True) or (RECOMPUTE_GRIDSEARCH == True)):
        print('Error : if want to load grid results, you should not have saved them or recomputed them before, or you will loose all your training data')
        
    else:
        with open(GRIDSEARCH_PICKLE_FILE, 'rb') as f:
            grid_search_cv_results = pickle.load(f)
           
        df_grid_search_results = pd.concat([pd.DataFrame(grid_search_cv_results["params"]),pd.DataFrame(grid_search_cv_results["mean_test_score"], columns=["Accuracy"])],axis=1)
        
''''''


# In[105]:


'''
if ((LOAD_GRID_RESULTS == True) or (RECOMPUTE_GRIDSEARCH == True)):
    grid_search_cv_results
'''


# ### Comme meilleur estimateur, avec la métrique prédiction on a donc : NCA avec 150 composants,  KNN avec algorithme kd_tree et distance manhattan

# In[106]:


'''
if ((LOAD_GRID_RESULTS == True) or (RECOMPUTE_GRIDSEARCH == True)):
    qgrid_show(df_grid_search_results)
''''''


# ### Les paramètres qui sortent du lot sont la dimension 150, la métrique de minkowski, en ne dropant pas d'autre feature que l'imdb_score

# # Annexe 3 : Vérification du ratio de variance expliquée :

# In[39]:


recommendation_pipeline_PCA_KNN['pca'].explained_variance_ratio_.sum()


# In[42]:


# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(df_encoded)
X_scaled = std_scale.transform(df_encoded)


# In[55]:


X_scaled.shape


# In[58]:


range(X_scaled.shape[0])


# In[51]:


# Calcul des composantes principales
pca = decomposition.PCA(n_components=4998)
pca.fit(X_scaled)


# In[56]:



pca.explained_variance_ratio_.cumsum()


# In[70]:


fig = plt.figure()
fig.suptitle('% de variance expliquée en fonction du nb de dimensions')
plt.plot(range(X_scaled.shape[0]), pca.explained_variance_ratio_.cumsum())
plt.ylabel("% explained variance")
plt.xlabel("Dimensions")

