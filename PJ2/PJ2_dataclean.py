#!/usr/bin/env python
# coding: utf-8

# # Openclassrooms PJ2 : Openfood facts dataset :  data cleaning notebook 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import zipfile
import urllib

import matplotlib.pyplot as plt

import numpy as np

DOWNLOAD_ROOT = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/parcours-data-scientist/P2/"
FOOD_PATH = os.path.join("datasets", "openfoodfacts")
FOOD_PATH_FILE = os.path.join(FOOD_PATH, "fr.openfoodfacts.org.products.csv")
FOOD_URL = DOWNLOAD_ROOT + "fr.openfoodfacts.org.products.csv.zip"

FOOD_PATH_FILE_OUTPUT = os.path.join(FOOD_PATH, "fr.openfoodfacts.org.products_transformed.csv")

DOWNLOAD_DATA = False  # A la première exécution du notebook, ou pour rafraîchir les données, mettre cette variable à True


import seaborn as sns
sns.set()


# # Fonctions d'affichage et de manipulation de données qui seront utilisées dans le notebook

# In[2]:


from IPython.display import display, Markdown

def display_freq_table(col_names):
    for col_name in col_names:    
        effectifs = food[col_name].value_counts(bins=5)

        modalites = effectifs.index # l'index de effectifs contient les modalités


        tab = pd.DataFrame(modalites, columns = [col_name]) # création du tableau à partir des modalités
        tab["Nombre"] = effectifs.values
        tab["Frequence"] = tab["Nombre"] / len(food) # len(data) renvoie la taille de l'échantillon
        tab = tab.sort_values(col_name) # tri des valeurs de la variable X (croissant)
        tab["Freq. cumul"] = tab["Frequence"].cumsum() # cumsum calcule la somme cumulée
        
        display(Markdown('#### ' + col_name))
        display(tab)


# In[3]:


'''
Cette fonction donne des informations pour aider à décider quelle feature on doit conserver, dans le cas où
on a 2 features qui semblent correspondre à la même notion

Elle remonte 3 informations :
% de cas où la valeur de la colonne 1 est renseignée, mais pas la 2
% de cas où la valeur de la colonne 2 est renseignée, mais pas la 1
% de cas où les valeurs de la colonne 1 et 2 sont renseignées toutes les deux

'''

def compare_na(df, col1, col2):
    num_rows, num_cols = df.shape
    
    col1notnull_col2null = round ( ( ( ( df[ (df[col1].notnull()) & ( df[col2].isna() ) ][[col1,col2]].shape[0] ) / num_rows ) * 100), 5)
    col2notnull_col1null = round ( ( ( ( df[ (df[col2].notnull()) & ( df[col1].isna() ) ][[col1,col2]].shape[0] ) / num_rows ) * 100), 5)
    col1notnull_col2notnull = round ( ( ( ( df[ (df[col1].notnull()) & ( df[col2].notnull() ) ][[col1,col2]].shape[0] ) / num_rows ) * 100), 5)
    
    print(f'Cas où {col1} est renseigné mais pas {col2} : {col1notnull_col2null}%')
    print(f'Cas où {col2} est renseigné mais pas {col1} : {col2notnull_col1null}%')
    print(f'Cas où {col1} et {col2} sont renseignés tous les deux : {col1notnull_col2notnull}%')


# # Téléchargement et décompression des données

# In[4]:


#PROXY_DEF = 'BNP'
PROXY_DEF = None

def fetch_food_data(food_url=FOOD_URL, food_path=FOOD_PATH):
    if not os.path.isdir(food_path):
        os.makedirs(food_path)
    archive_path = os.path.join(food_path, "fr.openfoodfacts.org.products.csv.zip")
    
    if (PROXY_DEF == 'BNP'):
        #create the object, assign it to a variable
        proxy = urllib.request.ProxyHandler({'https': 'https:/login:password@ncproxy:8080'})
        # construct a new opener using your proxy settings
        opener = urllib.request.build_opener(proxy)
        # install the openen on the module-level
        urllib.request.install_opener(opener)    
    
    urllib.request.urlretrieve(food_url, archive_path)
    food_archive = zipfile.ZipFile(archive_path)
    food_archive.extractall(path=food_path)
    food_archive.close()


# In[5]:


if (DOWNLOAD_DATA == True):
    fetch_food_data()


# # Import du fichier CSV

# ## Inspection de quelques lignes du fichier pour avoir un aperçu visuel du texte brut :

# In[6]:


def read_raw_file(nblines, food_path = FOOD_PATH):
    csv_path = os.path.join(food_path, "fr.openfoodfacts.org.products.csv")
    
    fp = open(csv_path)
    
    line = ""
    
    for cnt_lines in range(nblines+1):
        line = fp.readline()
        
    print(">>>>>> Line %d" % (cnt_lines))
    print(line)
    
    


# In[7]:


read_raw_file(0)


# In[8]:


read_raw_file(1)
read_raw_file(2)
read_raw_file(3)
read_raw_file(4)
read_raw_file(5)


# ### On voit que le séparateur des données semble être la tabulation.
# De plus, dans la documentation des différents champs qui a été fournie, il est indiqué d'utiliser le séparateur tabulation et l'encodage UTF-8 :
# https://world.openfoodfacts.org/data/data-fields.txt
# 
# ### On fait donc un chargement en spécifiant le séparateur tabulation, avec encodage utf-8
# 

# In[9]:


import pandas as pd

def load_food_data(food_path=FOOD_PATH):
    csv_path = os.path.join(food_path, "fr.openfoodfacts.org.products.csv")
    return pd.read_csv(csv_path, sep='\t', header=0, encoding='utf-8', low_memory=False)


# In[10]:


food = load_food_data()


# ###  On vérifie que le nombre de lignes intégrées dans le Dataframe correspond au nombre de lignes du fichier

# In[11]:


num_lines = sum(1 for line in open(FOOD_PATH_FILE, encoding='utf-8'))
message = (
f"Nombre de lignes dans le fichier (en comptant l'entête): {num_lines}\n"
f"Nombre d'instances dans le dataframe: {food.shape[0]}"
)
print(message)


# ### Puis on affiche quelques instances de données :

# In[12]:


food.head()


# # Correction d'une limitation de volume sur jupyter notebook
# 
# Lors de la première tentativement de chargement, j'ai eu l'erreur suivante à partir du poste Linux (cette erreur ne s'est pas produite sur le poste windows) :
# 
# IOPub data rate exceeded.  
# The notebook server will temporarily stop sending output  
# to the client in order to avoid crashing it.  
# To change this limit, set the config variable  
# `--NotebookApp.iopub_data_rate_limit`.  
# 
# Current values:  
# NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)  
# NotebookApp.rate_limit_window=3.0 (secs)  
# 
# 
# J'ai donc redémarré le jupyter notebook comme suit :  
# $ jupyter notebook --generate-config  
# Puis j'ai édité le fichier jupyter_notebook_config.py pour changer la valeur comme suit :  
# 
# c.NotebookApp.iopub_data_rate_limit = 10000000000  
# 
# Et j'ai relancé jupyter notebook  
# Ensuite, j'ai pu relancer load_food_data() dans la cellule ci-dessus  
# 
# A partir du poste Windows je n'ai pas eu l'erreur ci-dessus, en revanche j'ai eu le warning suivant :  
# 
# C:\Users\a42692\.conda\envs\FBO\lib\site-packages\IPython\core\interactiveshell.py:3242: DtypeWarning: Columns  (0,3,5,19,20,24,25,26,27,28,35,36,37,38,39,48) have mixed types. Specify dtype option on import or set low_memory=False.  
#   if (await self.run_code(code, result,  async_=asy)):  
#   
#   J'ai réglé ce warning en rajoutant low_memory=False à la fonction pd.read_csv. Avec ce paramètre, pandas charge d'abord le tableau en entier dans la RAM afin de pouvoir déterminer dynamiquement le type de chaque instance.

# # Inspection générale des données

# In[13]:


#pd.options.display.max_columns = 1000
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows",1000)


# ## Vérfication s'il y a des doublons

# In[14]:


food[food.duplicated()]


# #### => Aucun doublon détecté

# In[15]:


food.head()


# In[16]:


food.tail()


# ## Affichage des colonnes :

# In[17]:


food.info(verbose=True, null_counts=True)


# In[18]:


food.describe()


# ## Affichage des champs renseignés (non NA) avec leur pourcentage de complétude
# L'objectif est de voir quelles sont les features qui seront les plus fiables en terme de qualité de donnée, et quelles sont celles pour lesquelles on devra faire des choix

# In[19]:


(food.count()/food.shape[0]).sort_values(axis=0, ascending=False)


# ### Premiers constats :
# #### De nombreux champs ont très peu de valeurs renseignées, voire aucune
# #### Certaines colonnes sont présentes dans le fichier CSV mais pas dans l'url de description des champs fournie :
# 
# allergens 28344 non-null object
# 
# allergens_fr 19 non-null object
# 
# traces_fr 24352 non-null object
# 
# additives_fr 154680 non-null object
# 
# nutrition_grade_uk (mais 0 valeurs sont présentes)
# 
# pnns_groups_1  91513 non-null object
# 
# pnns_groups_2  94491 non-null object
# 
# states         320726 non-null object
# 
# states_tags    320726 non-null object
# 
# states_fr      320726 non-null object      
# 
# 

# ### Des url sont fournies pour les produits.  Voir si on pourra compléter les données avec les url fournies :

# In[20]:


pd.options.display.max_colwidth = 100
food['url']


# ### Un clic sur un ensemble d'url ci-dessus montre qu'elles ne fonctionnent pas : on ne pourra donc pas y récupérer des informations additionnelles
# Pour aller plus loin, on note que la colonne "creator" contient l'origine des données : on pourra donc remonter jusqu'à l'origine de ces données pour essayer d'avoir plus d'informations si nécessaire.
# 
# On ne va pas creuser cet aspect qui risque de prendre beaucoup de temps et qui est hors périmètre du projet, et on va se concentrer sur les données qui sont déjà à notre disposition.

# # Analyse plus détaillée des données et des colonnes manquantes

# ## Pourcentage des colonnes qui ont 70% de données manquantes :

# In[21]:


def analyse_donnees_manquantes(df, seuil = .7):
    nb_rows, nb_cols = df.shape

    nb_col_many_nulls = (((df.isnull().sum()) / nb_rows) > seuil)

    percentage_col_many_nulls = round(((nb_col_many_nulls.sum()) / nb_cols) * 100, 2)

    message = ( 
        f"{percentage_col_many_nulls} % des colonnes ont >= {seuil*100:0.0f}% de données manquantes \n"  
        f"Ces colonnes sont : \n"
        f"{nb_col_many_nulls[nb_col_many_nulls].to_string()}"
    )

    print(message)
    
analyse_donnees_manquantes(food)


# ### On voit que 73.46 % des colonnes ont >=70% de données manquantes

# # Filtre sur les produits vendus en France
# L'objectif étant de préparer des données pour des recettes pour un site français, on va faire l'hypothèse que les produits de ces recettes sont vendus en France.
# => **Poser la question au client pour confirmer** => **Le mentor a indiqué qu'on peut bien filtrer sur des produits français**

# ## Décider de quel champ utiliser pour filtrer sur le pays
# 
# 3 champs sont disponibles pour cela.
# Ils ont le même pourcentage de valeurs présentes :  
# countries                                     0.999127  
# countries_tags                                0.999127  
# countries_fr                                  0.999127  

# #### Consultation des valeurs des tags des pays pour voir sur quelles valeurs filtrer pour conserver uniquement la France :

# In[22]:


food['countries_tags'].value_counts()


# In[23]:


food['countries'].value_counts()


# In[24]:


food['countries_fr'].value_counts()


# #### Voici les filtres envisageables pour la France dans ces 3 colonnes :

# In[25]:


len(food[food['countries_tags'].str.contains("france")==True])


# In[26]:


len(food[ ( food['countries'].str.contains("France", case=False)==True ) | ( food['countries'].str.contains("FR")==True )  ] )


# In[27]:


len(food[food['countries_fr'].str.contains("France")==True])


# #### On voit que les filtres countries_fr et countries_tags sont équivalents car ils renvoient le même nombre de lignes
# #### En revanche le filtre sur countries ne permet pas de renvoyer autant de lignes que les autres.
# 
# #### Le tableau ci-dessous permet de lister toutes les valeurs qui détectent la France selon le filtre sur countries_fr, mais qui ne la détectent pas selon le filtre sur countries.   
# #### Il permet de voir pourquoi le filtre sur countries renvoie moins de lignes  (c'est parce que dans le champ countries, le nom du pays est traduit dans différents langages), et il permet aussi de confirmer par une inspection visuelle que les filtres sur countries_tags et countries_fr sont équivalents

# In[28]:


food[( food['countries_fr'].str.contains("France", na=False) == True ) &  
     ( 
         ( food['countries'].str.contains("France", case=False, na=False)==True ) | 
         ( food['countries'].str.contains("FR", na=False)==True ) == False 
     )][['countries', 'countries_fr', 'countries_tags']]


# In[29]:


food = food[food['countries_tags'].str.contains("france")==True].copy()


# In[30]:


food.info(verbose=True, null_counts=True)


# ## Nouvelle analyse des données manquantes, avec uniquement les produits vendus en France

# In[31]:


analyse_donnees_manquantes(food)


# # Sélection des features à nettoyer et analyser
# 
# Vu le nombre important de colonnes disponibles, il va falloir faire une sélection :    
# * Par rapport à la qualité des données au pourcentage de valeurs renseignées  
# * Par rapport à l'objectif, c'est à dire par rapport aux principes de base d'une alimentation saine  
# 

# ## Suppression des features par rapport à la qualité des données 
# ### (suppression des features dont le nombre de données est inférieur au seuil minimum défini)

# In[32]:


import collections

min_percentage_feature_values_tokeep = collections.defaultdict(lambda : 0.01)
min_percentage_feature_values_tokeep['vitamin-b2_100g'] = 0.01

def drop_lowquality_values(df, min_percentage_feature_values_tokeep):
    num_rows, num_cols = df.shape
    
    for column_name in df.columns:
        if (len(food[food[column_name].notnull()]) < min_percentage_feature_values_tokeep[column_name] * num_rows):
            df.drop([column_name], axis='columns', inplace = True)


def drop_lowquality_values_does_not_work(df, min_percentage_feature_values_tokeep):
    # Ne fonctionne pas :  subset ne permet de définir les colonnes que quand axis définit les lignes alors qu'ici, axis définit déjà les colonnes
    for column_name in df.columns:
        df.dropna(axis = 'columns', thresh = min_percentage_feature_values_tokeep['code'], subset = ['code'], inplace = True)
  
 


# In[33]:


drop_lowquality_values(food, min_percentage_feature_values_tokeep)


# ## Affichage des features restantes avec leur pourcentage de complétude

# In[34]:


(food.count()/food.shape[0]).sort_values(axis=0, ascending=False)


# ## Suppression des features par rapport à l'objectif
# ### On commence par rechercher les principes de base d'une alimentation saine
# ### Chaque principe de base fera l'objet de features dans le dataset qui seront conservées

# ####  Scoring de la qualité nutritionnelle
# 
# Pour faire ce scoring, on peut utiliser :
# * le nutriscore : https://quoidansmonassiette.fr/comment-est-calcule-le-nutri-score-logo-nutritionnel/  
# * un scoring fondé sur le nutriscore en intégrant également les notions d'additifs et de bio, comme ce que fait par exemple l'application Yuka : https://yuka.io/questions/notation-produits-alimentaires/
# 
# #### Dimension écologique
# Si on souhaite intégrer la dimension écologique, on pourra prendre en compte la présence d'huile de palme dans les aliments
# 
# #### Consommer un maximum de « raw food » (produits bruts non transformés) avec une priorité absolue sur les fruits et légumes
# Eviter les aliments trop sucrés  
# privilégier le bio français  (versus bio industriel intensif dont la charte change d'un pays à l'autre)  
# 
# Source : https://alimentsain.fr/  
# 
# 
# #### manger des produits de saison :
# Indiqués ici : https://alimentsain.fr/aliment/calendrier-fruits-legumes/
# 
# 
# #### Fruits, aliments riches en fibres, poissons gras (saumon, sardine, maquereau), légumineuses ( : haricots, lentilles, soja, pois entiers ou cassés, pois chiches, fèves, luzerne ou lupins), fruits secs, viande blanche, huile végétale
# 
# #### Alimentation variée et équilibrée
# 
# ####  Eviter les additifs alimentaires considérés comme nocifs / cancérigènes
# 
# #### Eviter les aliments trop salés
# 
# #### Selon l'Organisation mondiale de la santé, un régime alimentaire sain est composé des éléments suivants1 :
# 
#     des fruits, des légumes, des légumes secs (comme des lentilles et des pois), des fruits secs et des céréales complètes (par exemple du maïs, millet, avoine, blé et riz brun non transformés)1
#     au moins 400 g (soit 5 portions) de fruits et légumes par jour, hors pommes de terre, patates douces, manioc et autres racines amylacées1
#     moins de 10 % de l’apport énergique total provenant de sucres libres, soit l’équivalent de 50 g (ou environ 12 cuillères à café rases) pour une personne de poids normal consommant environ 2000 calories par jour ; dans l’idéal, pour préserver davantage la santé, cette part devrait être inférieure à 5 % des apports énergiques totaux1
#     moins de 30 % de l’apport énergétique total provenant des matières grasses. Les graisses insaturées (poisson, avocat et noix, et les huiles de tournesol, de soja, de colza et d’olive) sont à préférer aux graisses saturées (viande grasse, beurre, huile de palme et de noix de coco, crème, fromage, beurre clarifié et saindoux) et aux acides gras trans (aliments industriels et viande et produits laitiers provenant des animaux ruminants). Il est proposé de réduire l’apport en graisses saturées à moins de 10 % de l’apport énergétique total et celui en acides gras trans à moins de 1 % ; les acides gras trans sont à exclure d'une alimentation saine1.
#     moins de 5 g de sel (soit environ une cuillère à café) par jour, de préférence iodé1
# 
# Source : Wikipedia
# 
# 
# ####  AJR (Apports Journaliers Recommandés) en France :
# 
# Nutriment 	Apport journalier recommandé  
# Vitamine A (rétinol) 	800 μg  
# Vitamine B1 (thiamine) 	1,1 mg  
# Vitamine B2 (riboflavine) 	1,4 mg  
# Vitamine B3 (ou PP, niacine) 	16 mg  
# Vitamine B5 (acide pantothénique) 	6 mg  
# Vitamine B6 (pyridoxine) 	1,4 mg  
# Vitamine B8 ou H (biotine) 	50 μg  
# Vitamine B9 (acide folique)(Folacine) 	200 μg  
# Vitamine B12 (cobalamine) 	2,5 μg  
# Vitamine C (acide ascorbique) 	80 mg  
# Vitamine D (cholécalciférol) 	5 μg  
# Vitamine E (tocophérol) 	12 mg  
# Vitamine K (anti-AVK) 	75 μg  
# Calcium 	800 mg  
# Fer 	14 mg  
# Iode 	150 μg  
# Magnésium 	375 mg  
# Phosphore 	700 mg  
# Sélénium 	55 μg  
# Zinc 	10 mg  
# Potassium 	2 000 mg  
# Chlorure 	800 mg  
# Cuivre 	1 mg  
# Manganèse 	2 mg  
# Fluorure 	3,5 mg  
# Chrome 	40 μg  
# Molybdène 	50 μg  
# 
# Source: Wikipedia : https://fr.wikipedia.org/wiki/Apports_journaliers_recommand%C3%A9s
# 
# 

# ## Features les plus importantes pour chaque critère d'une alimentation de qualité
# Pour chaque critère de recherche réalisée en début de document sur les principes de base d'une alimentation saine, voici les features que l'on pourra appliquer avec leur pourcentage de champs renseignés (non NA) :
# 
# ####  Scoring de la qualité nutritionnelle
# Choisir entre l'une des 2 variables pour le nutrition score (numérique) ou grade (lettre) :  
# nutrition-score-fr_100g                    0.623883  
# nutrition_grade_fr                         0.623883
# 
# => Vérifier que les variables ont la même information 2 à 2  (boxplot avec en X la valeur groupée et en Y la valeur continue)
# 
# energy_100g                                0.656166  
# salt_100g                                  0.635656  
# sodium_100g                                0.635626  
# sugars_100g                                0.635057  
# saturated-fat_100g                         0.633635  
# 
# % de fruits et légumes :
# fruits-vegetables-nuts_100g                0.030181  
# => Cette donnée n'est pas assez renseignée.  On ne pourra donc pas se baser dessus.
# 
# fiber_100g                                 0.464476  
# proteins_100g                              0.653373  
# 
# 
# la teneur en sodium correspond à la teneur en sel mentionnée sur la déclaration obligatoire divisée par 2,5. 
# (Source : Nutri-score_reglement_usage_041019.pdf page 14)
# 
# => On construira 1 feature de teneur en sodium à partir des 2 features sodium_100g et salt_100g
# 
# 
# #### Dimension écologique
# ingredients_from_palm_oil_n                0.543133  
# ingredients_that_may_be_from_palm_oil_n    0.543133  
# 
# Cette dimension écologique est facultative car elle n'influe pas sur la qualité nutritionnelle des aliments
# 
# #### Consommer un maximum de « raw food » (produits bruts non transformés) avec une priorité absolue sur les fruits et légumes
# Pour reconnaître les légumes :
# 
# pnns_groups_2                              0.681004    =  'Legumes', 'Vegetables', 'vegetables' , 'fruits', legumes'  
# pnns_groups_1                              0.659092    =  'Fruits and vegetables'  
# 
# Nombre d'ingrédients et nombre d'additifs (plus il y en a, plus le produit est transformé) :  
# ingredients_text                              0.543133  
# additives_n                                   0.543133  
# 
# 
# #### privilégier le bio français  (versus bio industriel intensif dont la charte change d'un pays à l'autre)  
# Combiner les 3 features ci-dessous pour reconnaître le bio et le bio français :   
# labels_tags                                0.356959  
# labels_fr                                  0.356959  
# labels                                     0.356573  
# 
# Pour récupérer les produits vendus (et non pas fabriqués) en France :
# countries_tags                             1.000000  
# 
# #### manger des produits de saison :
# On pourrait construire un tableau contenant les produits par saison de manière à proposer des recettes ciblées selon la date.
# 
# En récupérant par exemple les informations ici :
# https://alimentsain.fr/aliment/calendrier-fruits-legumes/
# 
# Mais il s'agit d'une source de données externe hors périmètre de ce projet. 
# 
# 
# #### Fruits, aliments riches en fibres, poissons gras (saumon, sardine, maquereau), légumineuses ( : haricots, lentilles, soja, pois entiers ou cassés, pois chiches, fèves, luzerne ou lupins), fruits secs, viande blanche, huile végétale
# Intégré en grande partie dans le nutriscore.
# 
# #### Alimentation variée et équilibrée
# 
# ####  Eviter les additifs alimentaires considérés comme nocifs / cancérigènes
# additives_tags                             0.309386  
# additives_fr                                  0.309386
# 
# additives_n                                   0.543133
# additives                                     0.543001
# 
# => Il faudra conserver l'un de ces champs
# 
# 
# Il sera nécessaire de croiser avec une source de données externe pour identifier les additifs cancérigènes
# 
# #### Eviter les aliments trop salés
# Intégré dans le nutriscore
# 
# 
# ####  AJR (Apports Journaliers Recommandés) en France :
# Les features concernées ne sont pas assez représentées dans les données fournies.
# On ne pourra pas les utiliser :
# 
# vitamin-b1_100g                            0.008736  
# magnesium_100g                             0.008279  
# vitamin-e_100g                             0.008035  
# vitamin-b6_100g                            0.007578  
# vitamin-pp_100g                            0.007213  
# vitamin-b9_100g                            0.007009  
# vitamin-b2_100g                            0.006562  
# vitamin-d_100g                             0.006004  
# vitamin-a_100g                             0.005983  
# phosphorus_100g                            0.005719  
# vitamin-b12_100g                           0.005638  
# 
# #### Identification des produits
# 
# On pourra égalementconserver l'url de l'image du produit afin de permettre au client de faire des visuels :  
# image_url                                  0.533726
# 
# Ainsi que le code produit, la date de modification, le nom du produit, les marques et la catégorie :
# 
# code                                       1.000000  
# last_modified_t                            1.000000  
# product_name                               0.926930  
# brands                                     0.877956  
# brands_tags                                0.877915  
# main_category_fr                           0.629368  
# 
# 
# Ces champs contiennent l'état de certaines données (renseignées ou non, à vérifier ou non). 
# Il sera inutile de s'en servir pour vérifier les données qui sont renseignées ou non (on peut le faire nous mêmes en regardant les valeurs effectivement renseignées dans le dataframe)
# On revanche on conservera l'un des 3 champs ci-dessous pour écarter les valeurs qui contiennent "en:to-be-checked"
# 
# La signification des états (states) est documentée ici:  
# https://en.wiki.openfoodfacts.org/State#To_be_checked
# 
# En revanche il est indiqué : "As of March 2019 this state is still under development and evaluation. "
# Il sera judicieux d'attendre que ce champ soit stabilisé avant de l'utiliser
# 
# states                                        1.000000  
# states_tags                                   1.000000  
# states_fr                                     1.000000

# ### Une insepection visuelle sur les champs pnns_groups_2 et pnns_groups_1 montre qu'ils ne seront pas utiles pour détecter les légumes non transformés 
# Par exemple le beurre de cacahuète est taggé "Legumes"

# In[35]:


food[food['pnns_groups_2'].str.contains("legumes", case=False, na=False)][['product_name', 'main_category_fr', 'pnns_groups_2', 'pnns_groups_1']].head(1000)


# In[36]:


food[food['pnns_groups_1'].str.contains("legumes", case=False, na=False)][['product_name', 'main_category_fr', 'pnns_groups_2', 'pnns_groups_1']].head(1000)


# ### Nutriscore : décider quelle feature conserver
# nutrition-score-fr_100g ou nutrition_grade_fr

# nutrition-score-fr_100g 0.623883
# nutrition_grade_fr 0.623883

# In[37]:


food.groupby(['nutrition_grade_fr'])['nutrition_grade_fr'].count().plot(kind='pie')


# In[38]:


food.groupby(['nutrition_grade_fr'])['nutrition_grade_fr'].count().plot(kind='bar')


# #### nutrition-score-fr_100g : les valeurs sont bien comprises entre -15 et 40 : pas d'anomalie apparente quand le champ est renseigné

# In[39]:


sns.distplot(food[food['nutrition-score-fr_100g'].notnull()]['nutrition-score-fr_100g'], kde=True)


# ### Box plot avec en abscisse la feature nutrition_grade_fr, et en ordonnée la feature nutrition-score-fr_100g

# In[40]:


plt.figure(figsize=(16, 10))
sns.boxplot(x='nutrition_grade_fr', y='nutrition-score-fr_100g', data=food.sort_values('nutrition_grade_fr'))


# #### A partir de la page 18 du document Nutri-score_reglement_usage_041019.pdf on peut vérifier que :  
# A : valeurs entre Min et -1  (sauf pour les eaux qui sont obligatoirement de classe A)  
# B : Min à 2  
# C : 2 à 10  
# D : 6 à 18  
# E : 10 à max  
# 
# On se rend compte en lisant la page 18 du PDF que le nutri grade(A, B, C, D, E) est différent (décalé d'un cran) selon que l'aliment est une boisson ou non.  
# 
# **Le champ nutrition grade porte donc plus d'information. C'est lui que l'on retiendra pour la suite. On pourra le transformer en chiffre pour faciliter un scoring**

# #### Affichage des nutrtion score > -1 :  on voit que ce sont des eaux. C''est pour cela qu'il y a quelques outliers pour le nutrition grade A

# In[41]:


food[( food['nutrition-score-fr_100g'] > -1) & 
     (food['nutrition_grade_fr'] == 'a')][['product_name', 'nutrition-score-fr_100g', 'nutrition_grade_fr']]


# ### sodium_100g, salt_100g : décider quelle feature conserver ou combiner
# sodium_100g 0.635626  
# salt_100g 0.635656  

# In[42]:


compare_na(food, 'sodium_100g', 'salt_100g')


# #### Ci-dessous on voit que le rapport entre les deux features est constant :

# In[43]:


(food['sodium_100g'] / food['salt_100g']).hist()


# #### => On conservera la variable salt_100g qui est plus complète dans 0.00305% des cas (3 valeurs de plus que sodium_100g, les 2 variables sont quasi équivalentes)

# ### Huile de palme : décide quelle feature conserver ou combiner
# ingredients_from_palm_oil_n                0.543133  
# ingredients_that_may_be_from_palm_oil_n    0.543133  

# In[44]:


compare_na(food, 'ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n')


# #### Visualisation graphique de la différence de valeur entre les deux variables  (0 = pas de différence)

# In[45]:


(food['ingredients_from_palm_oil_n'] - food['ingredients_that_may_be_from_palm_oil_n']).hist()


# #### Quelques exemples d'écarts de valeur entre les 2 variables :

# In[46]:


food[(food['ingredients_from_palm_oil_n'] - food['ingredients_that_may_be_from_palm_oil_n']) != 0][['product_name', 'ingredients_from_palm_oil_n','ingredients_that_may_be_from_palm_oil_n']].head(1000)


# #### On voit que dans une grande majorité des cas, la différence entre les 2 variables est nulle et qu'elles sont donc équivalentes
# #### On conservera donc ingredients_from_palm_oil_n

# ### labels* : décider quelles features conserver / combiner

# labels_tags                                0.356959  
# labels_fr                                  0.356959  
# labels                                     0.356573  

# In[47]:


compare_na(food, 'labels_tags', 'labels_fr')


# In[48]:


compare_na(food, 'labels_tags', 'labels')


# In[49]:


compare_na(food, 'labels_fr', 'labels')


# #### L'inspection montre que 2 valeurs sont intéressantes pour détecter ce qui est bio et ce qui est fabriqué en France : bio, AB, organic et made-in-france.
# #### Elle montre aussi les attributs suivants qui permettront de reconnaître un produit bio :
# Bio:  
# en:organic  
# 
# Bio européen:  
# en:eu-organic  
# 
# Bio français:  
# fr:ab-agriculture-biologique
# 
# Produits fabriqués en France:  
# en:made-in-france  
# fr:cuisine-en-france  
# fr:viande-francaise  
# fr:volaille-francaise  
# 
# Produit contenat des ogm :
# fr:contient-des-ogm  

# In[50]:


food[['product_name', 'labels', 'labels_fr', 'labels_tags']].head(1000)


# In[51]:


food[food['labels_tags'].str.contains("bio|france|organic", case=False, na=False)][['product_name', 'labels', 'labels_fr', 'labels_tags']].head(1000)


# #### Ci-dessous on voit que quand labels_tags correspond à du bio ou du français, il n'y a pas d'occurence de labels_fr qui ne contient pas la même information.  Les 2 champs sont équivalents.

# In[52]:


food[( food['labels_tags'].str.contains("bio|organic|france", case=False, na=False) == True ) &  
     ( 
         ( food['labels_fr'].str.contains("bio|organic|france", case=False, na=False)==False ) 
     )][['labels_tags', 'labels', 'labels_fr']]


# In[53]:


food[( food['labels_fr'].str.contains("bio|organic|france", case=False, na=False) == True ) &  
     ( 
         ( food['labels_tags'].str.contains("bio|organic|france", case=False, na=False)==False ) 
     )][['labels_tags', 'labels', 'labels_fr']]


# #### Ci-dessous une comparaison entre labels_tags et labels montre que le champ labels est moins standard que les deux autres  : il faudrait filtrer sur SE-EKO-01 ou DE-ÖKO ou désigner du bio, là où il suffit de filtrer sur "bio" ou "organic" dans les champs labels_tags ou labels_fr

# In[54]:


food[( food['labels_tags'].str.contains("bio|organic|france", case=False, na=False) == True ) &  
     ( 
         ( food['labels'].str.contains("bio|organic|france|AB", case=False, na=False)==False ) 
     )][['labels_tags', 'labels', 'labels_fr']]


# #### => On retiendra donc le champ labels_tags pour identifier les produits bio ou fabriqués en France

# ### additives* : décider quelle feature conserver
# 
# additives_tags                             0.309386  
# additives_fr                                  0.309386   
# 
# additives_n                                   0.543133  
# additives                                     0.543001  
# 

# #### L'inspection visuelle ci-dessous montre que le champ additives ne correspond pas vraiment à ce qu'on cherche, et que additives_n reflète bien le nombre d'additifs renseignés dans les champs additives_tags et additives_fr

# In[55]:


food[['product_name', 'additives_tags', 'additives_fr', 'additives_n', 'additives']].head(1000)


# #### La comparaison des valeurs NA et l'inspection visuelle  montrent que les champs additives_tags et additives_fr sont équivalents => On conservera la feature additives_tags, ainsi que additives_n pour le nombre d'additifs

# In[56]:


compare_na(food, 'additives_tags', 'additives_fr')


# ### states* : décider quelle feature conserver
# states                                        1.000000  
# states_tags                                   1.000000  
# states_fr                                     1.000000  
# 

# In[57]:


food[['product_name', 'states', 'states_tags', 'states_fr']].head(1000)


# #### Un exam visuel montre qu'on pourra filtrer sur "to-be-checked" ou sur "A vérifier" si on souhaite écarter les instances de données qui ne sont pas encore validées. Cela permettra au client de sortir des recettes fondées uniquement sur les informations les plus fiables

# #### Les contrôles ci-dessous et l'inspection visuelle montrent que les 3 champs contiennent le même niveau d'information  => on retiendra le champ states_tags

# In[58]:


food[( food['states_tags'].str.contains("to-be-checked", case=False, na=False) == True ) &  
     ( 
         ( food['states'].str.contains("to-be-checked", case=False, na=False)==False ) 
     )][['states', 'states_tags', 'states_fr']]


# In[59]:


food[( food['states'].str.contains("to-be-checked", case=False, na=False) == True ) &  
     ( 
         ( food['states_tags'].str.contains("to-be-checked", case=False, na=False)==False ) 
     )][['states', 'states_tags', 'states_fr']]


# In[60]:


food[( food['states_tags'].str.contains("to-be-checked", case=False, na=False) == True ) &  
     ( 
         ( food['states_fr'].str.contains("A vérifier", case=False, na=False)==False ) 
     )][['states', 'states_tags', 'states_fr']]


# In[61]:


food[( food['states_tags'].str.contains("A vérifier", case=False, na=False) == True ) &  
     ( 
         ( food['states_fr'].str.contains("to-be-checked", case=False, na=False)==False ) 
     )][['states', 'states_tags', 'states_fr']]


# ### main_category*
# main_category_fr                           0.629368  
# main_category
# 
# 

# In[62]:


food[['product_name', 'main_category_fr', 'main_category']].head(1000)


# => On conservera main_category_fr 

# ### ingredients_text*
# ingredients_text                              0.543133  
# 

# #### On conservera ce champ afin de compter le nombre d'ingrédients :

# In[63]:


food[['product_name', 'ingredients_text']].sample(1000)


# In[64]:


food[food['ingredients_text'].notnull()]


# ### Comptage du nombre d'ingrédients

# In[65]:


food[food['ingredients_text'].notnull()]['ingredients_text'].str.strip().str.split(',').apply(len)


# ## Note : les features ci-dessous n'ont pas été conservées car redondantes ou non utiles au regard du crible des critères qui identifient une alimentation de qualité
Champs inutiles pour déterminer la qualité des aliments (par rapport aux critères indiqués plus haut sur une alimentation de qualité) :

packaging                                     78960 non-null object
packaging_tags                                78961 non-null object


emb_codes                                     29306 non-null object
emb_codes_tags                                29303 non-null object
first_packaging_code_geo                      18803 non-null object

cities                                        23 non-null object

Ces 2 champs sont redondants avec countries_tags  (qui lui aussi a le même niveau de qualité de donnée), comme déjà vu plus haut
countries                                     320492 non-null object
countries_fr                                  320492 non-null object

Pas assez de valeurs :
allergens_fr                                  19 non-null object

traces_tags
traces                                        24353 non-null object
traces_fr                                     24352 non-null object

no_nutriments                                 0 non-null float64

serving_size                                  211331 non-null object

cocoa_100g                                    948 non-null float64

Champs pas du tout ou très peu renseignés :
ingredients_from_palm_oil                     0 non-null float64
ingredients_that_may_be_from_palm_oil         0 non-null float64
nutrition_grade_uk                            0 non-null float64

butyric-acid_100g                             0 non-null float64
caproic-acid_100g                             0 non-null float64
caprylic-acid_100g                            1 non-null float64
capric-acid_100g                              2 non-null float64
lauric-acid_100g                              4 non-null float64
myristic-acid_100g                            1 non-null float64
palmitic-acid_100g                            1 non-null float64
stearic-acid_100g                             1 non-null float64
arachidic-acid_100g                           24 non-null float64
behenic-acid_100g                             23 non-null float64
lignoceric-acid_100g                          0 non-null float64
cerotic-acid_100g                             0 non-null float64
montanic-acid_100g                            1 non-null float64
melissic-acid_100g                            0 non-null float64

omega-3-fat_100g                              841 non-null float64
alpha-linolenic-acid_100g                     186 non-null float64
eicosapentaenoic-acid_100g                    38 non-null float64
docosahexaenoic-acid_100g                     78 non-null float64
omega-6-fat_100g                              188 non-null float64
linoleic-acid_100g                            149 non-null float64
arachidonic-acid_100g                         8 non-null float64
gamma-linolenic-acid_100g                     24 non-null float64
dihomo-gamma-linolenic-acid_100g              23 non-null float64
omega-9-fat_100g                              21 non-null float64
oleic-acid_100g                               13 non-null float64
elaidic-acid_100g                             0 non-null float64
gondoic-acid_100g                             14 non-null float64
mead-acid_100g                                0 non-null float64
erucic-acid_100g                              0 non-null float64
nervonic-acid_100g                            0 non-null float64

sucrose_100g                                  72 non-null float64
glucose_100g                                  26 non-null float64
fructose_100g                                 38 non-null float64
lactose_100g                                  262 non-null float64
maltose_100g                                  4 non-null float64
maltodextrins_100g                            11 non-null float64
starch_100g                                   266 non-null float64
polyols_100g                                  414 non-null float64

casein_100g                                   27 non-null float64
serum-proteins_100g                           16 non-null float64
nucleotides_100g                              9 non-null float64

beta-carotene_100g                            34 non-null float64

biotin_100g                                   330 non-null float64
silica_100g                                   38 non-null float64
bicarbonate_100g                              81 non-null float64
chloride_100g                                 158 non-null float64

fluoride_100g                                 79 non-null float64
chromium_100g                                 20 non-null float64
molybdenum_100g                               11 non-null float64

caffeine_100g                                 78 non-null float64
taurine_100g                                  29 non-null float64
ph_100g                                       49 non-null float64


chlorophyl_100g                               0 non-null float64
glycemic-index_100g                           0 non-null float64
water-hardness_100g                           0 non-null float64

collagen-meat-protein-ratio_100g              165 non-null float64


A la place de ces champs, on se contentera du comptage ingredients_from_palm_oil_n et ingredients_that_may_be_from_palm_oil_n :
ingredients_from_palm_oil_tags                4835 non-null object
ingredients_that_may_be_from_palm_oil_tags    11696 non-null object


Redondant avec main_category_fr:
main_category                                 84366 non-null object

# # Nettoyage des données pour les features cibles

# ## Conserver uniquement les features retenues

# In[66]:


features_list = ['code', 'last_modified_t', 'product_name' , 'states_tags', 'main_category_fr','brands','brands_tags', 'nutrition_grade_fr','energy_100g','sugars_100g','salt_100g','saturated-fat_100g','fiber_100g','proteins_100g','ingredients_from_palm_oil_n','pnns_groups_2','pnns_groups_1','labels_tags','countries_tags','additives_tags','additives_n','ingredients_text','image_url']
food = food[features_list]


# ## Ajout d'une feature numérique de scoring pour le nutrition grade

# In[67]:


def convert_category_to_number(cat):
    cat_table = {
        'a' : 5,
        'b' : 4,
        'c' : 3,
        'd' : 2,
        'e' : 1,
        'nan' : None,
    }
    
    return (cat_table.get(cat,None))


food_cat = pd.DataFrame(food['nutrition_grade_fr'].apply(convert_category_to_number))

food['nutrition_scoring'] = food_cat


# ## Ajout d'une feature pour le comptage du nombre d'ingrédients

# In[68]:


food_no_ingredients = pd.DataFrame(food[food['ingredients_text'].notnull()]['ingredients_text'].str.strip().str.split(',').apply(len))

food['no_ingredients'] = food_no_ingredients


# In[69]:


pd.set_option('display.max_colwidth', -1)
food[food['ingredients_text'].notnull()][['product_name','ingredients_text', 'no_ingredients']].sample(100)


# In[70]:


no_ingredients_mean = food[food['no_ingredients'].notnull()]['no_ingredients'].mean()
no_ingredients_median = food[food['no_ingredients'].notnull()]['no_ingredients'].median()


# In[71]:


plt.figure(figsize=(16, 10))
plt.axvline(no_ingredients_mean, 0, 1, color='red')
plt.axvline(no_ingredients_median, 0, 1, color='green')
sns.distplot(food[food['no_ingredients'].notnull()]['no_ingredients'], kde=True)


# #### Moyenne et écart type sur le graphe ci-dessus ?  Beaucoup de valeurs proches de 0 ??

# In[72]:


food[food['no_ingredients'].notnull()]['no_ingredients'].describe()


# ### L'observation ci-dessous montre que le nombre d'additifs (additives_n) est déjà inclus dans le nombre d'ingrédients (no_ingredients: feature que l'on a créée)

# In[73]:


food[food['no_ingredients'].notnull()][['product_name', 'no_ingredients', 'ingredients_text', 'additives_n','additives_tags']].sample(100)


# ### Ajout de la feature de scoring par rapport au nombre d'ingrédients

# In[74]:


no_ingredients_scoring_bins = [0, 3, 5, 7, 10, np.inf]
no_ingredients_scoring_labels = [5, 4, 3, 2, 1]
                                 
food['no_ingredients_scoring'] = pd.cut(food['no_ingredients'], bins=no_ingredients_scoring_bins, labels=no_ingredients_scoring_labels)


# In[75]:


food[food['ingredients_text'].notnull()][['product_name','ingredients_text', 'no_ingredients', 'no_ingredients_scoring']].sample(100)


# ## Ajout d'une feature de scoring par rapport au nombre d'additifs nocifs
# ### Si un additif nocif est présent : score = 1
# ### Si pas d'additif nocif : score = 5

# In[76]:


# Pour une application réelle, il faudra récupérer la liste des additifs noctifs sur une source de données externe à déterminer
additives_nocive_list = ['e100', 'e101', 'e103','e104', 'e111', 'e124', 'e128', 'e131', 'e132', 'e133', 'e143', 'e171', 'e173', 'e199', 'e214', 'e215', 'e216', 'e217', 'e218', 'e219', 'e240', 'e249', 'e250', 'e251', 'e386', 'e620', 'e621','e622','e623','e624','e625', 'e924', 'e924a', 'e924b', 'e926', 'e950', 'e951', 'e952', 'e952i','e952ii','e952iii','e952iv']

additives_nocive_list_search_exp = '|'.join(additives_nocive_list)

def additives_nocive_score_item(additives_tags):
    additives_tags_list = additives_tags.split(',')
    
    additives_tags_list = [item.strip() for item in additives_tags_list]
    
    for additive_nocive in additives_nocive_list:
        if ('en:'+additive_nocive in additives_tags_list):
            return(1)
    
    return(5)

food['additives_nocive_scoring'] = pd.DataFrame(food[food['additives_tags'].notnull()]['additives_tags'].apply(additives_nocive_score_item))


# ## Ajout de features de scoring par rapport aux proportions
# energy_100g                                0.656166  
# salt_100g                               
# sugars_100g                                0.635057  
# saturated-fat_100g                         0.633635  
# 
# fiber_100g                                 0.464476  
# proteins_100g                              0.653373  
# 

# In[77]:


# Ces valeurs de scoring ont été remplies par rapport au document Nutri-score_reglement_usage_041019.pdf

'''

proportions_scoring_bins = {
    'energy_100g': {
        'bins': [30, 90, 150, 210, 270,np.inf],
        'labels': [5, 4, 3, 2, 1]
    },
    
    'salt_100g': {
        'bins': [0.225, 0.675, 1.125, 1.575, 2.025,np.inf],
        'labels': [5, 4, 3, 2, 1]
    },
    
    'sugars_100g': {
        'bins': [4.5, 13.5, 22.5, 31, 40,np.inf],
        'labels': [5, 4, 3, 2, 1]
    },

    'saturated-fat_100g': {
        'bins': [1, 3, 5, 7, 9,np.inf],
        'labels': [5, 4, 3, 2, 1]
    },  
    
    'fiber_100g': {
        'bins': [0.9, 1.9, 2.8, 3.7, 4.7,np.inf],
        'labels': [1, 2, 3, 4, 5]
    },  
    
    'proteins_100g': {
        'bins': [1.6, 3.2, 4.8, 6.4, 8,np.inf],
        'labels': [1, 2, 3, 4, 5]
    },    
}
'''

proportions_scoring_bins = {
    'energy_100g': {
        'bins': [-np.inf, 335, 1005, 1675, 2345, np.inf],
        'labels': [5, 4, 3, 2, 1]
    },
    
    'salt_100g': {
        'bins': [-np.inf, 0.225, 0.675, 1.125, 1.575, np.inf],
        'labels': [5, 4, 3, 2, 1]
    },
    
    'sugars_100g': {
        'bins': [-np.inf, 4.5, 13.5, 22.5, 31,np.inf],
        'labels': [5, 4, 3, 2, 1]
    },

    'saturated-fat_100g': {
        'bins': [-np.inf, 1, 3, 5, 7,np.inf],
        'labels': [5, 4, 3, 2, 1]
    },  
    
    'fiber_100g': {
        'bins': [-np.inf, 1.9, 2.8, 3.7, 4.7,np.inf],
        'labels': [1, 2, 3, 4, 5]
    },  
    
    'proteins_100g': {
        'bins': [-np.inf, 3.2, 4.8, 6.4, 8,np.inf],
        'labels': [1, 2, 3, 4, 5]
    },    
}


for feature_name in proportions_scoring_bins.keys():
    feature_name_scoring = feature_name + '_scoring'
    
    food[feature_name_scoring] = pd.cut(food[feature_name], bins=proportions_scoring_bins[feature_name]['bins'], labels=proportions_scoring_bins[feature_name]['labels'])


# In[78]:


food.info()


# In[79]:


#food[['energy_100g_scoring']].plot()


# In[80]:


food.groupby(['energy_100g_scoring'])['energy_100g_scoring'].count().plot(kind='pie')


# In[81]:


food.groupby(['salt_100g_scoring'])['salt_100g_scoring'].count().plot(kind='pie')


# In[82]:


food.info()


# In[83]:


analyse_donnees_manquantes(food, 0.7)


# ## Ajout de la feature de scoring par rapport au bio et au bio français
# labels_tags                                0.356959   

# ### Si bio français : scoring = 5
# ### Si bio européen : scoring = 4
# ### Si bio non français, non européen : scoring = 3
# ### Si produit non bio, mais fabriqué en France: scoring = 2
# ### Si non bio : scoring = 1
# 
# Bio:  
# en:organic  
# 
# Bio européen:  
# en:eu-organic  
# 
# Bio français:  
# fr:ab-agriculture-biologique
# 
# Produits fabriqués en France:  
# en:made-in-france  
# fr:cuisine-en-france  
# fr:viande-francaise  
# fr:volaille-francaise  

# In[84]:


bio_list = ['en:organic']
bio_europeen_list = ['en:eu-organic']
bio_francais_list = ['fr:ab-agriculture-biologique']
madeinfrance_list = ['en:made-in-france', 'fr:cuisine-en-france', 'fr:viande-francaise', 'fr:volaille-francaise ']

def bio_score_item(labels_tags):
    labels_tags_list = labels_tags.split(',')
    
    labels_tags_list = [item.strip() for item in labels_tags_list]
    
    for bio_francais in bio_francais_list:
        if (bio_francais in labels_tags_list):
            return(5)

    for bio_europeen in bio_europeen_list:
        if (bio_europeen in labels_tags_list):
            return(4)

    for bio in bio_list:
        if (bio in labels_tags_list):
            return(3)

    for madeinfrance in madeinfrance_list:
        if (madeinfrance in labels_tags_list):
            return(2)
    
    return(1)

food['bio_scoring'] = pd.DataFrame(food[food['labels_tags'].notnull()]['labels_tags'].apply(bio_score_item))


# In[85]:


food[['bio_scoring','labels_tags']].sample(100)


# ## Affichage des champs renseignés (non NA) avec leur pourcentage de complétude
# L'objectif est de voir quelles sont les features qui seront les plus fiables en terme de qualité de donnée

# In[86]:


(food.count()/food.shape[0]).sort_values(axis=0, ascending=False)


# In[87]:


food.columns


# In[88]:


food.sample(1000).to_csv('sample.csv')


# ## Analyse des features quantitatives

# In[89]:


food.describe()


# In[90]:


food.skew()


# ### On constate une dysymétrie très importante de la plupart des valeurs

# In[91]:


food.hist(bins=50, figsize=(20,15))


# ### Affichage des tableaux de fréquence

# In[92]:


from IPython.display import display, Markdown

def display_freq_table(col_names):
    for col_name in col_names:    
        effectifs = food[col_name].value_counts(bins=5)

        modalites = effectifs.index # l'index de effectifs contient les modalités


        tab = pd.DataFrame(modalites, columns = [col_name]) # création du tableau à partir des modalités
        tab["Nombre"] = effectifs.values
        tab["Frequence"] = tab["Nombre"] / len(food) # len(data) renvoie la taille de l'échantillon
        tab = tab.sort_values(col_name) # tri des valeurs de la variable X (croissant)
        tab["Freq. cumul"] = tab["Frequence"].cumsum() # cumsum calcule la somme cumulée
        
        display(Markdown('#### ' + col_name))
        display(tab)


# In[93]:


display_freq_table(['energy_100g','salt_100g','sugars_100g','saturated-fat_100g','fiber_100g','proteins_100g','ingredients_from_palm_oil_n'])


# #### NB : les fréquences ci-dessus ne sont pas de 100% car il y a des valeurs non renseignées

# # Sauvegarde des données cleanées dans un nouveau CSV

# In[94]:


food.to_csv(FOOD_PATH_FILE_OUTPUT, index=False)


# # Rebut : Code divers / inutile pour l'instant

# ### Feature product_name

# In[95]:


food[food['product_name'].duplicated(keep=False) & food['product_name'].notnull()].count()


# In[96]:


food[food['product_name'].duplicated(keep=False) & food['product_name'].notnull()].sort_values(['product_name']).head(500)


# In[97]:


sf=food[food['product_name'].duplicated() & food['product_name'].notnull()].sort_values(['code']).sample(500)


# In[98]:


food.shape


# ### Features proportions pour 100g
# energy_100g                                0.656166  
# salt_100g                                  0.635656  
# sodium_100g                                0.635626  
# sugars_100g                                0.635057  
# saturated-fat_100g                         0.633635  
# 
# fiber_100g                                 0.464476  
# proteins_100g                              0.653373  
# 
# 

# In[99]:


food[food['energy_100g'] == 0]['energy_100g'].count()


# In[100]:


food[food['energy_100g'] == 0]


# In[101]:


food.query('salt_100g > 100')


# In[102]:


q = food['salt_100g'].quantile(0.99)


# In[103]:


q


# In[104]:


food[food['salt_100g'] > q].sort_values("salt_100g")


# In[105]:


sns.set_style("whitegrid")


# In[106]:


sns.boxplot(x='salt_100g', y='nutrition_grade_fr', data=food)


# In[107]:


sns.boxplot(x='salt_100g', y='nutrition_grade_fr', data=food)


# In[108]:


sns.boxplot(x='sugars_100g', y='nutrition_grade_fr', data=food)


# In[109]:


sns.boxplot(x='saturated-fat_100g', y='nutrition_grade_fr', data=food)


# In[110]:


sns.boxplot(x='energy_100g', y='nutrition_grade_fr', data=food)


# In[111]:


sns.boxplot(x='fiber_100g', y='nutrition_grade_fr', data=food)


# In[112]:


sns.boxplot(x='proteins_100g', y='nutrition_grade_fr', data=food)


# In[113]:


sns.pairplot(food[['energy_100g', 'sugars_100g', 'nutrition_grade_fr']], hue='nutrition_grade_fr')


# In[114]:


food_energy_notnull = food[food['energy_100g'].notnull()]


# In[115]:


food_energy_notnull


# In[116]:


sns.distplot(food_energy_notnull['energy_100g'], kde=True)


# In[117]:


#food_energy_notnull[food_energy_notnull['energy_100g'] != 0]['energy_100g']
f2 = pd.cut(food_energy_notnull['energy_100g'], bins=[0., 1., np.inf], labels=[1,2])
f2.hist()


# In[118]:


sns.distplot(food_energy_notnull[food_energy_notnull['energy_100g'] != 0]['energy_100g'], kde=True)


# In[119]:


food['energy_100g'].hist(bins=50, figsize=(20,15))


# In[120]:


food.query('energy_100g < 5000').shape


# In[121]:


food_energy_notnull[food_energy_notnull['energy_100g'] == 0]['energy_100g']


# In[122]:


food.pnns_groups_2.unique()


# In[123]:


food.pnns_groups_1.unique()


# In[124]:


pd.set_option("display.max_rows",10000)
pd.set_option("display.max_columns", 10000)

food.labels_tags.unique()


# In[125]:


food.additives_tags.unique()


# In[126]:


# Conversion du champ catégorique en numérique. 
#Ne fonctionne pas :
food_energyscoring_numerical = pd.DataFrame(food[food['energy_100g_scoring'].notnull()]['energy_100g_scoring'].astype('category').cat.codes + 1).sample(100)

# Semble modifier le type de la colonne, mais on ne peut quand même pas faire food[['energy_100g_scoring']].plot()
food.astype({'energy_100g_scoring': 'float64'}).dtypes

#food['energyscoring_numerical'] = food_energyscoring_numerical

