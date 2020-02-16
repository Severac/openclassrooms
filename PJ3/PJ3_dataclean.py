#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ3 : IMDB dataset :  data cleaning notebook 

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

def load_data(data_path=DATA_PATH):
    csv_path = DATA_PATH_FILE
    return pd.read_csv(csv_path, sep=',', header=0, encoding='utf-8')


# In[124]:


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

# In[125]:


df.head()


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


# In[191]:


numerical_features = ['movie_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes', 'imdb_score' , 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'facenumber_in_poster', 'duration', 'num_user_for_reviews', 'actor_3_facebook_likes', 'num_critic_for_reviews', 'director_facebook_likes', 'budget', 'gross','title_year']
categorical_features = ['color', 'content_rating']

categorical_features_totransform = ['country', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']
categorical_features_totransform_specific = ['movie_title']
categorical_features_tosplit_andtransform = ['genres', 'plot_keywords'] #Sous ensemble des features to transform, qu'il va falloir splitter avant de 1hot encoder
#categorical features to transform : genres, movie_title, country, actor_1_name, language, actor_2_name, actor_3_name, director_name, plot_keywords
#features to remove : aspect_ratio, movie_imdb_link




# In[193]:


df[categorical_features_tosplit_andtransform]


# In[218]:


np.isnan(df.at[4,'actor_3_name'])


# In[232]:


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


# In[237]:


df_transformed = df.copy(deep=True)

for feature_totransform in categorical_features_tosplit_andtransform:
    for i, row in df.iterrows():
        if (type(row[feature_totransform]) == str):        
            features_list = row[feature_totransform].split(sep='|')
            for feature_name in features_list:
                df_transformed.at[i, feature_totransform+'_'+feature_name] = 1


# In[235]:


qgrid_show(df)


# In[233]:


qgrid_show(df_transformed)


# In[236]:


df_transformed.shape


# In[97]:


for line in df['genres'].str.split(pat='|'):
    print(line)


# In[79]:


df['genres'].str.split(pat='|',expand=True)


# In[86]:


df['genres'].str.split(pat='|', expand=True)


# In[96]:


df['genres'].str.split(pat='|')


# In[19]:


qgrid_show(df[categorical_features_totransform])


# ### Le jeu de données est globalement bien renseigné. Mais comment va-t-on compléter les données manquantes ?

# # Imputation des données manquantes

# In[34]:


from sklearn.impute import KNNImputer

#imputer = KNNImputer(n_neighbors=2, weights="uniform")
#imputer.fit_transform(df[numerical_features])


# In[52]:


numerical_features_columns = df[numerical_features].columns


# In[61]:


numerical_features_index = df[numerical_features].index


# In[53]:


numerical_features_columns.shape


# In[39]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)

transformed_data = imp.fit_transform(df[numerical_features])  

#>>> X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]

#imp.transform(X_test)


# In[62]:


df_numerical_features_imputed = pd.DataFrame(data=transformed_data, columns=numerical_features_columns, index=numerical_features_index)


# In[42]:


df[numerical_features].shape


# In[59]:


qgrid_show(df[numerical_features])


# In[63]:


qgrid_show(df_numerical_features_imputed)


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



plt.title('Corrélation entre les valeurs numériques')
sns.heatmap(corr_matrix[numerical_features].loc[numerical_features], 
        xticklabels=corr_matrix[numerical_features].loc[numerical_features].columns,
        yticklabels=corr_matrix[numerical_features].loc[numerical_features].columns, cmap='coolwarm', center=0.20)


# #### Il est intéressant de voir que budget et facenumber_in_poster sont très peu corrélées aux autres variables

# # Cercle des corrélations et première réduction de dimensionalité des variables numériques

# In[25]:


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

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
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
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

            plt.title("Projection des films (coloration : à définir) (sur F{} et F{})".format(d1+1, d2+1))
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


# In[26]:



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
data_pca = df [numerical_features]

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
#display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3),(4,5)], illustrative_var=data_pca[['nutrition_scoring']].values[:,0])
display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3),(4,5)])


plt.show()


# # Sélection des features pour le clustering

# 2 features à supprimer :
# movie_imdb_link              1.000000
# aspect_ratio                 0.934574 => information technique
# 
# facenumber_in_poster         0.997399  => voir la relation de ces variables avec les scores
# num_user_for_reviews         0.995798  => voir la relation de ces variables avec les scores
# num_critic_for_reviews       0.990196  => voir la relation de ces variables avec les scores
# 
# 
# 
# 

# # Features à transformer / ajouter

# Pour les variables qualitatives : créer autant de variables que de valeurs distinctes (one hot encode) ?
# 
# movie_title                  1.000000   => La distance entre chaque valeur devra être une distance de chaîne de caractère.   Mais comment faire un algo de clustering qui ne calcule pas la distance de la même façon pour cet attribut là que pour les autres ?  Faire un vecteur one hot avec le nombre distinct de titres de films dedans, et réduire sa dimensionalité  ?

# # Features catégorielles

# In[27]:


qgrid_show(df[categorical_features_totransform])


# In[28]:


df['country'].unique().shape[0]


# In[29]:


print(f'{df.shape[0]} valeurs au total dans le dataframe')
for col in df[categorical_features_totransform]:
    print(f'Feature {col} : {df[col].unique().shape[0]} valeurs uniques')

