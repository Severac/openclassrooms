#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ3 : IMDB dataset :  data clean and modelisation notebook 

# In[7]:


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




# In[8]:


def qgrid_show(df):
    display(qgrid.show_grid(df, grid_options={'forceFitColumns': False, 'defaultColumnWidth': 170}))


# # Téléchargement et décompression des données

# In[9]:


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


# In[10]:


if (DOWNLOAD_DATA == True):
    fetch_dataset()


# # Import du fichier CSV

# ## Chargement des données

# In[11]:


import pandas as pd
pd.set_option('display.max_columns', None)

def load_data(data_path=DATA_PATH):
    csv_path = DATA_PATH_FILE
    return pd.read_csv(csv_path, sep=',', header=0, encoding='utf-8')


# In[48]:


df = load_data()


# ###  On vérifie que le nombre de lignes intégrées dans le Dataframe correspond au nombre de lignes du fichier

# In[13]:


num_lines = sum(1 for line in open(DATA_PATH_FILE, encoding='utf-8'))
message = (
f"Nombre de lignes dans le fichier (en comptant l'entête): {num_lines}\n"
f"Nombre d'instances dans le dataframe: {df.shape[0]}"
)
print(message)


# ### Puis on affiche quelques instances de données :

# In[14]:


df.head()


# ### Vérification s'il y a des doublons

# In[15]:


df[df.duplicated()]


# ### Suppression des doublons

# In[10]:


df.drop_duplicates(inplace=True)
df = df.reset_index(drop=True)


# In[12]:


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

# In[13]:


(df.count()/df.shape[0]).sort_values(axis=0, ascending=False)


# ## Identification des typologies de features à traiter 

# In[14]:


numerical_features = ['movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes', 'imdb_score' , 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'facenumber_in_poster', 'duration', 'num_user_for_reviews', 'actor_3_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'budget', 'gross','title_year']

# à 1 hot encoder, et à splitter avant si nécessaire  ('genres' et 'plot_keywords' doivent être splittées)
categorical_features = ['country', 'director_name', 'genres', 'plot_keywords', 'color', 'content_rating']

# à transformer en bag of words
categorical_features_tobow = ['movie_title']  

# à fusioner en 1 seule variable
categorical_features_tomerge = ['actor_1_name', 'actor_2_name', 'actor_3_name']  

# features qui ne seront pas conservées :
features_notkept = ['aspect_ratio', 'movie_imdb_link']




# ## Affichage des features qui seront splittées avant le 1hot encode :

# In[15]:


df[['genres', 'plot_keywords']]


# # Imputation des données manquantes

# In[16]:


# KNN imputer pas encore supporté par la version de sklearn que j'utilise :

#from sklearn.impute import KNNImputer

#imputer = KNNImputer(n_neighbors=2, weights="uniform")  
#imputer.fit_transform(df[numerical_features])


# In[17]:


numerical_features_columns = df[numerical_features].columns


# In[18]:


numerical_features_index = df[numerical_features].index


# In[19]:


numerical_features_columns.shape


# ## Imputation des données numériques par régression linéaire

# In[20]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)

transformed_data = imp.fit_transform(df[numerical_features])  


# In[21]:


df_numerical_features_imputed = pd.DataFrame(data=transformed_data, columns=numerical_features_columns, index=numerical_features_index)


# ### Visualisation de quelques résultats par comparaison avant/après :

# In[22]:


qgrid_show(df[numerical_features])


# In[23]:


qgrid_show(df_numerical_features_imputed)


# ### Constat que toutes les valeurs sont maintenant renseignées :

# In[24]:


(df_numerical_features_imputed.count()/df_numerical_features_imputed.shape[0]).sort_values(axis=0, ascending=False)


# ## Transformation des features de catégorie
# ### Voir le §  Identification des typologies de features à traiter  ci-dessus pour une explication des différents cas de figure

# In[25]:


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


# In[26]:


df_imputed = add_categorical_features_1hot(df, df_numerical_features_imputed, categorical_features)
df_imputed = add_categorical_features_merge_and_1hot(df, df_imputed, categorical_features_tomerge, 'actors_names' )


# In[27]:


df_imputed.shape


# In[28]:


df_imputed = add_categorical_features_bow_and_1hot(df, df_imputed, categorical_features_tobow)


# In[29]:


df_imputed.shape


# In[30]:


df_imputed.head(10)


# In[31]:


df_imputed.describe()


# ## Comparaison avant/après de quelques valeurs 1hot encoded :

# In[32]:


df[['actor_1_name', 'actor_2_name', 'actor_3_name', 'actors_names', 'country', 'genres']].head(10)


# In[33]:


df_imputed[['actors_names_Johnny Depp', 'actors_names_Orlando Bloom', 'actors_names_Jack Davenport', 'actors_names_Joel David Moore', 'country_USA', 'country_UK', 'genres_Action', 'genres_Adventure']].head(10)


# In[34]:


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

# In[35]:


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


# In[36]:


X_reduced = pca.transform(X_scaled)


# In[37]:


X_reduced.shape


# In[38]:


from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X_reduced)
distances_matrix, reco_matrix = nbrs.kneighbors(X_reduced)


# In[39]:


distances_matrix.shape


# In[40]:


reco_matrix.shape


# In[41]:


reco_matrix


# In[42]:


print(f"{(df.iloc[0]['movie_title'])}")


# In[57]:


df[df['movie_title'].str.contains('Nixon')]


# In[69]:


pd.options.display.max_colwidth = 100
df.loc[[3820]]


# In[66]:


df.loc[[1116]]


# In[44]:


def afficher_recos(film_index, reco_matrix):
    print(f"Film choisi : {(df.loc[film_index]['movie_title'])} - imdb score : {df.loc[film_index]['imdb_score']} - {df.loc[film_index]['movie_imdb_link']}")
          
    print(f"\nFilms recommandés : ")
    for nb_film in range(5):
        print(f"{df.loc[reco_matrix[film_index, nb_film+1]]['movie_title']} - imdb score : {df.loc[reco_matrix[film_index, nb_film+1]]['imdb_score']} - {df.loc[reco_matrix[film_index, nb_film+1]]['movie_imdb_link']}")


# In[45]:


afficher_recos(2703, reco_matrix)


# In[46]:


afficher_recos(0, reco_matrix)


# In[47]:


afficher_recos(3, reco_matrix)


# In[48]:


afficher_recos(4820, reco_matrix)


# In[49]:


afficher_recos(647, reco_matrix)


# In[50]:


afficher_recos(124, reco_matrix)


# In[51]:


afficher_recos(931, reco_matrix)


# In[52]:


afficher_recos(1172, reco_matrix)


# In[53]:


afficher_recos(3820, reco_matrix)


# # Industralisation du modèle avec Pipeline

# In[43]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class DuplicatesRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, df, y=None):      
        return self
    
    def transform(self, df):
        df.drop_duplicates(inplace=True)
        df = df.reset_index(drop=True)
        
        return(df)


# In[44]:


duplicates_remover = DuplicatesRemover()


# In[50]:


df = duplicates_remover.fit_transform(df)


# In[46]:


df = duplicates_remover.transform(df)


# In[51]:


df[df.duplicated()]


# In[52]:


preparation_and_recommendation_pipeline = Pipeline([
    ('duplicates_remover', DuplicatesRemover),

])

df2 = preparation_and_recommendation_pipeline.fit(df)


# In[53]:


type(df2)


# In[34]:


df = preparation_and_recommendation_pipeline.transform(df)


# # Features à transformer / ajouter

# movie_title                  1.000000   => La distance entre chaque valeur devra être une distance de chaîne de caractère.   Mais comment faire un algo de clustering qui ne calcule pas la distance de la même façon pour cet attribut là que pour les autres ?  Faire un vecteur one hot avec le nombre distinct de titres de films dedans, et réduire sa dimensionalité  ?

# # Annexe : inutile, à effacer plus tard

# In[54]:


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


# In[55]:


'''
df_transformed = df.copy(deep=True)

for feature_totransform in categorical_features_tosplit_andtransform:
    for i, row in df.iterrows():
        if (type(row[feature_totransform]) == str):        
            features_list = row[feature_totransform].split(sep='|')
            for feature_name in features_list:
                df_transformed.at[i, feature_totransform+'_'+feature_name] = 1
'''

