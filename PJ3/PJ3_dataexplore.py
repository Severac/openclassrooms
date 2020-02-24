#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ3 : IMDB dataset :  data exploration notebook 

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




# # Téléchargement et décompression des données

# In[2]:


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


# In[3]:


if (DOWNLOAD_DATA == True):
    fetch_dataset()


# # Import du fichier CSV

# ## Inspection de quelques lignes du fichier pour avoir un aperçu visuel du texte brut :

# In[4]:


def read_raw_file(nblines, data_path = DATA_PATH):
    csv_path = DATA_PATH_FILE
    
    fp = open(csv_path)
    
    line = ""
    
    for cnt_lines in range(nblines+1):
        line = fp.readline()
        
    print(">>>>>> Line %d" % (cnt_lines))
    print(line)
    
    


# In[5]:


read_raw_file(0)
read_raw_file(1)
read_raw_file(2)


# ## Chargement des données

# In[6]:


import pandas as pd

pd.set_option('display.max_columns', None)

def load_data(data_path=DATA_PATH):
    csv_path = DATA_PATH_FILE
    return pd.read_csv(csv_path, sep=',', header=0, encoding='utf-8')


# In[7]:


df = load_data()


# ###  On vérifie que le nombre de lignes intégrées dans le Dataframe correspond au nombre de lignes du fichier

# In[8]:


num_lines = sum(1 for line in open(DATA_PATH_FILE, encoding='utf-8'))
message = (
f"Nombre de lignes dans le fichier (en comptant l'entête): {num_lines}\n"
f"Nombre d'instances dans le dataframe: {df.shape[0]}"
)
print(message)


# ### Puis on affiche quelques instances de données :

# In[9]:


df.head(20)


# In[10]:


def qgrid_show(df):
    display(qgrid.show_grid(df, grid_options={'forceFitColumns': False, 'defaultColumnWidth': 170}))


# In[11]:


display(qgrid.show_grid(df, grid_options={'forceFitColumns': False, 'defaultColumnWidth': 170}))


# In[12]:


df.info()


# In[13]:


df.describe()


# ### Vérification s'il y a des doublons

# In[14]:


df[df.duplicated()]


# ### Suppression des doublons

# In[15]:


df.drop_duplicates(inplace=True)


# In[16]:


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

# In[17]:


(df.count()/df.shape[0]).sort_values(axis=0, ascending=False)


# ## Identification des typologies de features à traiter 

# In[18]:


numerical_features = ['movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes', 'imdb_score' , 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'facenumber_in_poster', 'duration', 'num_user_for_reviews', 'actor_3_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'budget', 'gross','title_year']

# à 1 hot encoder, et à splitter avant si nécessaire  ('genres' et 'plot_keywords' doivent être splittées)
categorical_features = ['country', 'director_name', 'genres', 'plot_keywords', 'color', 'content_rating']

# à transformer en bag of words
categorical_features_tobow = ['movie_title']  

# à fusioner en 1 seule variable
categorical_features_tomerge = ['actor_1_name', 'actor_2_name', 'actor_3_name']  

# features qui ne seront pas conservées :
features_notkept = ['aspect_ratio', 'movie_imdb_link']




# ## Affichage des features à valeurs multiples :

# In[19]:


df[['genres', 'plot_keywords']]


# In[20]:


df.hist(bins=50, figsize=(20,15))


# In[21]:


scatter_matrix(df[numerical_features], figsize=(30,30))
plt.suptitle('Diagramme de dispersion des données numériques')


# In[22]:


corr_matrix = df.corr()


# In[23]:


corr_matrix[numerical_features].loc[numerical_features]


# In[24]:


plt.figure(figsize=(16, 10))
plt.title('Corrélation entre les valeurs numériques')
sns.heatmap(corr_matrix[numerical_features].loc[numerical_features], 
        xticklabels=corr_matrix[numerical_features].loc[numerical_features].columns,
        yticklabels=corr_matrix[numerical_features].loc[numerical_features].columns, cmap='coolwarm', center=0.20)


# #### Il est intéressant de voir que budget et facenumber_in_poster sont très peu corrélées aux autres variables

# # Cercle des corrélations et première réduction de dimensionalité des variables numériques

# In[25]:


df['imdb_score_cat'] = pd.cut(df['imdb_score'], bins=5)


# In[26]:


df[['imdb_score', 'imdb_score_cat']]


# In[27]:


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

import matplotlib.cm as cm

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(30,30))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(16,9))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                
                x = np.arange(10)
                ys = [i+x+(i*x)**2 for i in range(10)]
                colors = iter(cm.rainbow(np.linspace(0, 1, len(ys))))
                
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, color=next(colors))
                
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            '''
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
            '''
            
            plt.xlim([-5,10])
            plt.ylim([-5,10])


            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des films (coloration : imdb score) (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

def plot_dendrogram(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
plt.show()


# In[28]:



from sklearn import decomposition
from sklearn import preprocessing

# Import `PCA` from `sklearn.decomposition`
from sklearn.decomposition import PCA

# Build the model
pca = PCA(n_components=2)

# choix du nombre de composantes à calculer
n_comp = 6

# import de l'échantillon
data = df

# selection des colonnes à prendre en compte dans l'ACP
#data_pca = df [numerical_features]

data_pca = df[['movie_facebook_likes',
 'num_voted_users',
 'cast_total_facebook_likes',
 'imdb_score',
 'actor_1_facebook_likes',
 'actor_2_facebook_likes',
 'facenumber_in_poster',
 'duration',
 'num_user_for_reviews',
 'actor_3_facebook_likes',
 'num_critic_for_reviews',
 'director_facebook_likes',
 'budget',
 'gross',
 'title_year']].copy()
 
    
#data_pca['imdb_score_cat'] = pd.cut(data_pca['imdb_score'], bins=5)



bins = [-np.inf,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data_pca['imdb_score_cat'] = pd.cut(data_pca['imdb_score'], bins=bins, labels=labels)

# Autres façons de faire :
#data_pca.at[:, 'imdb_score_cat'] = pd.cut(data_pca['imdb_score'], bins=bins, labels=labels)
#data_pca.loc[:, 'imdb_score_cat'] = pd.cut(data_pca['imdb_score'], bins=bins, labels=labels)


# préparation des données pour l'ACP
#data_pca = data_pca.fillna(data_pca.mean()) # Il est fréquent de remplacer les valeurs inconnues par la moyenne de la variable
data_pca = data_pca.dropna()

X = data_pca.values
#names = data["idCours"] # ou data.index pour avoir les intitulés

#features = data.columns
features = data_pca.columns

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca = decomposition.PCA(n_components=n_comp)
pca.fit(X_scaled)

# Eboulis des valeurs propres
display_scree_plot(pca)

# Cercle des corrélations
pcs = pca.components_
#plt.figure(figsize=(16,10))
plt.rcParams["figure.figsize"] = [16,9]
display_circles(pcs, n_comp, pca, [(0,1),(2,3),(4,5)], labels = np.array(features))


# Projection des individus
X_projected = pca.transform(X_scaled)
display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3),(4,5)], illustrative_var=data_pca[['imdb_score_cat']].values[:,0])
#display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3),(4,5)])


plt.show()


# ### Les analyses multivariées ci-dessus montrent plusieurs groupes principaux de variables:
# #### Les features liées aux acteurs 
#  => On voit notamment que cast_total_facebook_likes est très corrélé à actor_1_facebook_likes, actor_2_ ...
# #### Les features liées au film, au réalisateur, aux votes utilisateurs et au score imdb
# => num_voted_users est très corrélé à num_user_for_reviews (79%), num_critic_for_reviews (62%), gross (63%)
# #### Le budget (qui est très peu corrélé aux autres features)
# #### Le "face number in poster"  (qui est très peu corrélé aux autres features)

# # Features catégorielles

# In[29]:


qgrid_show(df[categorical_features])


# In[30]:


print(f'{df.shape[0]} valeurs au total dans le dataframe')
for col in df[categorical_features]:
    print(f'{col} : {df[col].unique().shape[0]} valeurs uniques')


# ## Exploration des informations facebook

# In[31]:


plt.scatter(df['actor_1_facebook_likes'], df['title_year'])


# In[32]:


df[df['title_year'] < 1980]


# In[33]:


df[(df['actor_1_facebook_likes'].notnull() == True) & (df['title_year'] < 2008)][['movie_title', 'title_year', 'actor_1_facebook_likes']].sample(50)


# In[34]:


df[(df['actor_1_facebook_likes'].notnull() == True) & (df['title_year'] < 1960)][['movie_title', 'title_year', 'actor_1_facebook_likes']].sample(50)


# In[35]:


len(df[df['title_year'] < 1980].index)


# ### Facebook est un outil récent, mais on voit que les valeurs sont tout de même renseignées pour les anciens films.
# ### Les likes sont bien présents de 1960 à 2000,  ils sont moins nombreux avant 1960-1980,  mais ils sont tout de même présents.  De plus, il y a beaucoup moins de films dans le jeu de données avant 1960-1980
# 
# movie_facebook_likes         1.000000  
# cast_total_facebook_likes    1.000000  
# actor_1_facebook_likes       0.998599  
# actor_2_facebook_likes       0.997399  
# actor_3_facebook_likes       0.995398  
# director_facebook_likes      0.979392  

# In[36]:


import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets

py.offline.init_notebook_mode(connected=True)

'''
trace_1 = go.Scatter(x = df.title_year, y = df['director_facebook_likes'],
                    name = 'Director facebook likes',
                    mode = 'markers',
                    text = df['movie_title']
                    )
'''

trace_2 = go.Scatter(x = df.title_year, y = df['actor_1_facebook_likes'],
                    name = 'Actor 1 facebook likes',
                    mode = 'markers',
                    text = df['actor_1_name']
                    )

layout = go.Layout(title = 'Likes / year graph ',
                   yaxis = dict(title='Nb FB likes director'),
                   xaxis = dict(title='Year'),
                   hovermode = 'closest')

fig = go.Figure(data = [trace_2], layout = layout)

py.offline.iplot(fig)

