#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ3 : IMDB dataset :  data clean and modelisation notebook 

# In[1]:


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

# In[23]:


df.head()


# ### Vérification s'il y a des doublons

# In[128]:


df[df.duplicated()]


# ### Suppression des doublons

# In[25]:


df.drop_duplicates(inplace=True)
#df = df.reset_index(drop=True)
df.reset_index(drop=True, inplace=True)


# In[26]:


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

# In[27]:


(df.count()/df.shape[0]).sort_values(axis=0, ascending=False)


# ## Identification des typologies de features à traiter 

# In[28]:


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

# In[30]:


df[['genres', 'plot_keywords']].sample(10)


# # Imputation des données manquantes

# In[31]:


# KNN imputer pas encore supporté par la version de sklearn que j'utilise :

#from sklearn.impute import KNNImputer

#imputer = KNNImputer(n_neighbors=2, weights="uniform")  
#imputer.fit_transform(df[numerical_features])


# In[32]:


numerical_features_columns = df[numerical_features].columns


# In[33]:


numerical_features_index = df[numerical_features].index


# In[34]:


numerical_features_columns.shape


# ## Imputation des données numériques par régression linéaire

# In[35]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)

transformed_data = imp.fit_transform(df[numerical_features])  


# In[48]:


df_numerical_features_imputed = pd.DataFrame(data=transformed_data, columns=numerical_features_columns, index=numerical_features_index)


# ### Visualisation de quelques résultats par comparaison avant/après :

# In[ ]:


qgrid_show(df[numerical_features])


# In[ ]:


qgrid_show(df_numerical_features_imputed)


# ### Constat que toutes les valeurs sont maintenant renseignées :

# In[40]:


(df_numerical_features_imputed.count()/df_numerical_features_imputed.shape[0]).sort_values(axis=0, ascending=False)


# ## Transformation des features de catégorie
# ### Voir le §  Identification des typologies de features à traiter  ci-dessus pour une explication des différents cas de figure

# In[41]:


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
        df_transformed = df[feature_totransform].str.lower().str.replace(r'[^\w\s]', '').str.get_dummies(sep=' ').add_prefix(feature_totransform +'_')
        df_target = pd.concat([df_target, df_transformed], axis=1)
        
    return(df_target)            


# In[42]:


df_imputed = add_categorical_features_1hot(df, df_numerical_features_imputed, categorical_features)
df_imputed = add_categorical_features_merge_and_1hot(df, df_imputed, categorical_features_tomerge, 'actors_names' )


# In[43]:


df_imputed.shape


# In[44]:


df_imputed = add_categorical_features_bow_and_1hot(df, df_imputed, categorical_features_tobow)


# In[45]:


df_imputed.shape


# In[51]:


df_imputed.head(10)


# In[ ]:


#df_imputed.describe()


# ## Comparaison avant/après de quelques valeurs 1hot encoded :

# In[53]:


df[['actor_1_name', 'actor_2_name', 'actor_3_name', 'actors_names', 'country', 'genres']].head(10)


# In[54]:


df_imputed[['actors_names_Johnny Depp', 'actors_names_Orlando Bloom', 'actors_names_Jack Davenport', 'actors_names_Joel David Moore', 'country_USA', 'country_UK', 'genres_Action', 'genres_Adventure']].head(10)


# In[55]:


df_imputed.loc[0]


# # Sélection des features pour le clustering

# La sélection des features restera à compléter, et à faire avant les 1 hot encode
# 
# 2 features à supprimer :
# movie_imdb_link              1.000000
# aspect_ratio                 0.934574 => information technique  
# 
# Elles ne sont déjà plus dans df_imputed,  pas besoin de les supprimer de df
# 
# facenumber_in_poster         0.997399  => voir la relation de ces variables avec les scores  
# num_user_for_reviews         0.995798  => voir la relation de ces variables avec les scores  
# num_critic_for_reviews       0.990196  => voir la relation de ces variables avec les scores  
# 
# 
# 
# 

# # Réduction de dimensionalité

# In[56]:


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


# In[57]:


X_reduced = pca.transform(X_scaled)


# In[58]:


X_reduced.shape


# In[59]:


from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X_reduced)
distances_matrix, reco_matrix = nbrs.kneighbors(X_reduced)


# In[60]:


distances_matrix.shape


# In[61]:


reco_matrix.shape


# In[62]:


reco_matrix


# In[63]:


print(f"{(df.iloc[0]['movie_title'])}")


# In[64]:


df[df['movie_title'].str.contains('Nixon')]


# In[65]:


pd.options.display.max_colwidth = 100
df.loc[[3820]]


# In[66]:


df.loc[[1116]]


# In[8]:


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

def get_similarity_df_scaled_input(df_scaled, index1, index2):
    # Transforming data so that values are between 0 and 1, positive
    # This function assumes that below code must be run before call
    '''
    scaler = MinMaxScaler() 
    array_scaled = scaler.fit_transform(df_encoded)
    df_scaled  = pd.DataFrame(data=array_scaled , columns=df_encoded.columns, index=df_encoded.index)
    '''
    
    # This line of code allows not to keep 1hot columns that are both 0  (for example, both films NOT having word "the" is not relevant :  1hot features are to sparse to keep 0 values like that)
    df_relevant_items = df_scaled[df_scaled.columns[(df_scaled.loc[index1] + df_scaled.loc[index2]) > 0]]
    
    # We substract from 1 because the higher the score, the higher the similarity
    # 1 hot columns that have 0 value as a result mean that 1 and only 1 of the 2 films has the attribute
    # (Those are differenciating attributes, as opposed to attributes that are both 0))
    return(1 - ((df_relevant_items.loc[index1] - df_relevant_items.loc[index2]).abs())).sort_values(ascending=False)


# In[9]:


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


# In[10]:


def afficher_recos_films(reco_matrix, df_encoded, with_similarity_display=False, films_temoins_indexes=[2703, 0, 3, 4820, 647, 124, 931, 1172, 3820]):
    for film_temoin_index in films_temoins_indexes:
        afficher_recos(film_temoin_index, reco_matrix, df_encoded, with_similarity_display=with_similarity_display)


# In[135]:


df.loc[2703]['country']


# In[137]:


afficher_recos_films(reco_matrix, df_encoded)


# In[140]:


afficher_recos_films(reco_matrix, df_encoded, with_similarity_display=True)


# In[77]:


df.to_numpy()


# In[78]:


df.shape[0]


# In[79]:


df[['movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes', 'imdb_score' , 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'facenumber_in_poster', 'duration', 'num_user_for_reviews', 'actor_3_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'budget', 'gross','title_year']]


# In[80]:


df[['movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes', 'imdb_score' , 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'facenumber_in_poster', 'duration', 'num_user_for_reviews', 'actor_3_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'budget', 'gross','title_year']].loc[4]


# In[81]:


df 


# # Industralisation du modèle avec Pipeline

# In[11]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF

from sklearn import decomposition
from sklearn import preprocessing
#from sklearn.neighbors import KNeighborsTransformer

from sklearn.neighbors import NeighborhoodComponentsAnalysis

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

            df_transformed = df[feature_tobow].str.lower().str.replace(r'[^\w\s]', '').str.get_dummies(sep=' ').add_prefix(feature_tobow +'_')
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

Elle est beaucoup trop lente, il faudrait l'optimiser :(
De plus elle ne fonctionne que si on passe à predict le même nombre de lignes qu'on a passé à fit

On peut calculer cette similarité en faisant un apply sur 2 variables:  
en appelant la fonction get_similarity_df(df_encoded, index1, index2).  
Pour ça, on passe df_encoded à la classe KNNTransform

Le paramètre devra être passé comme ceci:
recommendation_pipeline_PCA_KNN.fit(df_encoded, labels, KNN__df_encoded = df_encoded)
'''
    
class KNNTransform_predict_similarity(KNNTransform):
    def predict(self, X, y=None): # Quand on appelle predict, transform est appelé avant automatiquement
        print('KNN predict')

        distances_matrix, knn_matrix = self.nbrs.kneighbors(X)

        scoring_predictions = []
        
        scaler = MinMaxScaler() 
        array_scaled = scaler.fit_transform(self.df_encoded)
        df_scaled  = pd.DataFrame(data=array_scaled , columns=self.df_encoded.columns, index=self.df_encoded.index)
        
        for i in range(0, X.shape[0]):
            scoring_1 = get_similarity_df(df_scaled, knn_matrix[i, 0], knn_matrix[i, 1]).sum()
            scoring_2 = get_similarity_df(df_scaled, knn_matrix[i, 0], knn_matrix[i, 2]).sum()
            scoring_3 = get_similarity_df(df_scaled, knn_matrix[i, 0], knn_matrix[i, 3]).sum()
            scoring_4 = get_similarity_df(df_scaled, knn_matrix[i, 0], knn_matrix[i, 4]).sum()
            scoring_5 = get_similarity_df(df_scaled, knn_matrix[i, 0], knn_matrix[i, 5]).sum()
                
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


# In[12]:


# Récupération des étiquettes de scoring :

# D'abord, dropper les duplicates pour que les index de df soient alignés avec ceux de df_encoded (qui a déjà fait l'objet d'un drop duplicates dans le pipeline)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

labels = df['imdb_score'].to_numpy()


# In[15]:


df_encoded = preparation_pipeline.fit_transform(df)


# In[13]:


from sklearn.metrics import mean_squared_error

def print_rmse(labels, predictions):
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    print(f"Erreur moyenne de prédiction de l'IMDB score: {rmse}")


# ## Tests avec PCA_KNN :

# In[184]:


recommendation_pipeline_PCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('pca', decomposition.PCA(n_components=200)),
    ('KNN', KNNTransform_predict_similarity(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    #('pipeline_final', PipelineFinal()),
])

recommendation_pipeline_PCA_KNN.fit(df_encoded, labels, KNN__df_encoded = df_encoded)


# In[188]:


# recommendation_pipeline_PCA_KNN.predict(df_encoded)  # TROP LENT :(


# In[174]:


recommendation_pipeline_PCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('pca', decomposition.PCA(n_components=200)),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    #('pipeline_final', PipelineFinal()),
])

recommendation_pipeline_PCA_KNN.fit(df_encoded, labels)
predictions = recommendation_pipeline_PCA_KNN.predict(df_encoded)


# In[ ]:


print_rmse(labels, predictions)


# ### Résultat de tests effectués avec recommendation_pipeline_PCA_KNN, et n_components à 200 :
#  #### Erreur moyenne de prédiction de l'IMDB score avec la variable de scoring dans le JDD : 1.0095843224821195 
#  #### Erreur moyenne de prédiction de l'IMDB score sans la variable de scoring dans le JDD : 1.0486754159658844
# 

# ## Tests avec NCA_KNN :

# In[154]:


recommendation_pipeline_NCA_KNN.fit(df_encoded, labels)


# In[158]:


predictions = recommendation_pipeline_NCA_KNN.predict(df_encoded)

print('Avec n_components = 200 :')
print_rmse(labels, predictions)


# ### Résultats obtenus pour recommendation_pipeline_NCA_KNN avec différentes valeurs de n_components :
# 
# /home/francois/anaconda3/lib/python3.7/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
#   warnings.warn("Variables are collinear.")
# 
# Avec n_components = 2 
# Erreur moyenne de prédiction de l'IMDB score: 0.3803446132258534
# 
# 
# Avec n_components = 5 
# Erreur moyenne de prédiction de l'IMDB score: 0.3309076607635139
# 
# Avec n_components = 8 
# Erreur moyenne de prédiction de l'IMDB score: 0.3495230844373829
# 
# Avec n_components = 9 
# Erreur moyenne de prédiction de l'IMDB score: 0.7302570164839567
# 
# Avec n_components = 10 
# Erreur moyenne de prédiction de l'IMDB score: 0.7178573449279926
# 
# Avec n_components = 50 
# Erreur moyenne de prédiction de l'IMDB score: 0.8670957628917407
# 
# Avec n_components = 100 
# Erreur moyenne de prédiction de l'IMDB score: 0.935546695279086
# 
# Avec n_components = 200 
# Erreur moyenne de prédiction de l'IMDB score: 0.9944914304719338
# 
# Avec n_components = 300 
# Erreur moyenne de prédiction de l'IMDB score: 1.0142304399662543
# 
# Avec n_components = 500 
# Erreur moyenne de prédiction de l'IMDB score: 1.0635176481446682
# 

# In[159]:


distances_matrix, reco_matrix = recommendation_pipeline_NCA_KNN.transform(df_encoded)


# ### Affichage de recommendations pour NCA_KNN avec n_components 200 :

# In[161]:


afficher_recos_films(reco_matrix, df_encoded)


# ### => Les résultats semblent meilleurs avec NCA qu'avec PCA :
# ### Par exemple avec le 6ème sens on obtient plus de films avec le thème de la mort et les enfants
# ### Globalement les imdb score obtenus sont meilleurs  (erreur moyenne 0.99 contre 1.048,  confirmé par un examen visuel)

# In[ ]:


afficher_recos_films(reco_matrix, df_encoded, with_similarity_display=True)


# Affichage de recommendations pour NCA_KNN avec n_components 200 :

# In[165]:


recommendation_pipeline_NCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('NCA', NCATransform(nca_params =  {'random_state':42, 'n_components':5 })),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    #('pipeline_final', PipelineFinal()),
])

distances_matrix, reco_matrix = recommendation_pipeline_NCA_KNN.fit_transform(df_encoded, labels)


# In[166]:


afficher_recos_films(reco_matrix, df_encoded)


# ### => Les résultats sont clairement moins bons avec NCA components à 5  suivi de KNN : les imdb scores sont meilleurs mais les catégories sont moins bien capturées
# ### => Pourtant, l'erreur de prédiction de l'imdb score n'était que de 0.33 avec NBCA components = 5,  soit beaucoup moins qu'avec components = 200  :  cela montre que la métrique de prédiction de l'imdb score n'est pas parfaite

# In[167]:


afficher_recos_films(reco_matrix, df_encoded, with_similarity_display=True)


# 
# ## Test de différents paramètres avec NCA + KNN

# In[22]:


#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

recommendation_pipeline_NCA_KNN = Pipeline([
    ('features_droper', FeaturesDroper(features_todrop=['imdb_score'])),
    ('standardscaler', preprocessing.StandardScaler()),
    ('NCA', NCATransform(nca_params =  {'random_state':42, 'n_components':200 })),
    ('KNN', KNNTransform(knn_params =  {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'})),
    #('pipeline_final', PipelineFinal()),
])

'''
        'features_droper__features_todrop':  [None,
                                              ['imdb_score'],
                                              ['imdb_score'],
                                              ['imdb_score', 'actor_1_facebook_likes','actor_2_facebook_likes', 'actor_3_facebook_likes', 'num_user_for_reviews', 'num_critic_for_reviews', 'num_voted_users']
                                              ['imdb_score'],
                                              ['imdb_score'],
                                              ['imdb_score'],
                                              ['imdb_score'],
                                              
                                              
        ],
'''


'''
param_grid = {
        'features_droper__features_todrop':  [#None,
                                              ['imdb_score'],
                                              ['imdb_score', 'actor_1_facebook_likes','actor_2_facebook_likes', 'actor_3_facebook_likes', 'num_user_for_reviews', 'num_critic_for_reviews', 'num_voted_users']                                              
        ],
    
        'NCA__nca_params': [{'random_state':42, 'n_components':50 }, {'random_state':42, 'n_components':100 }, 
                            {'random_state':42, 'n_components':150 }, {'random_state':42, 'n_components':200 }
        ],
    
        'KNN__knn_params': [{'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'minkowski'}, 
                            {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'euclidean'}, 
                            {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'jaccard'}, 
                            {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'mahalanobis'}, 
                            {'n_neighbors':6, 'algorithm':'ball_tree', 'metric':'manhattan'}, 
                            
                            {'n_neighbors':6, 'algorithm':'kd_tree', 'metric':'minkowski'},
                            {'n_neighbors':6, 'algorithm':'kd_tree', 'metric':'manhattan'},
                            
                            {'n_neighbors':6, 'algorithm':'brute', 'metric':'minkowski'},
        ],
    }
    
'''
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


# In[23]:


grid_search.cv_results_


# In[31]:


grid_search.best_estimator_


# ### Comme meilleur estimateur on retiendra donc : NCA avec 150 composants,  KNN avec algorithme kd_tree et distance manhattan

# In[26]:


df_grid_search_results = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)


# In[32]:


qgrid_show(df_grid_search_results)


# In[28]:


df_grid_search_results.to_csv('grid_search_results.csv')


# In[30]:


import pickle

SAVE_GRID_RESULTS = False

if (SAVE_GRID_RESULTS == True):
    with open('grid_search_results.pickle', 'wb') as f:
        pickle.dump(grid_search.cv_results_, f, pickle.HIGHEST_PROTOCOL)


# ### Tentative de KNN avec cosine distance

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


# In[197]:


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


# # Features à transformer / ajouter

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

# In[149]:


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

