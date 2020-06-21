#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:42:51 2020

@author: francois


PJ5 Openclassrooms : this code is the graphical user interface 
It takes orders as input,  and outputs client IDs and their assigned clusters.

Interface is hosted here :  https://pj5.analysons.com/

See "README API.txt" for installation instructions :
    https://github.com/Severac/openclassrooms/blob/master/PJ5/README%20API.txt

"""

import streamlit as st
import pickle

from PIL import Image

from functions import *

import pandas as pd
import base64

import altair as alt
import matplotlib.pyplot as plt

#import pandas as pd

API_MODEL_PICKLE_FILE = 'API_model_PJ5.pickle'

LOGO_IMAGE_FILE = 'ecommerce.png'
TEMPLATE_FILE = 'UI_input_template.csv'
GRAPH_MODEL_FILE = 'graph_model_ui.png'
FEATURE_FILE_NAMES = ['model2_featuredistribution_Recency.png',\
                      'model2_featuredistribution_RfmScore.png',
                      'model2_featuredistribution_BoughtTopValueProduct.png',\
                      'model2_featuredistribution_HasEverCancelled.png',]

# This function is a hack to be able to start streamlit in "wide mode" by default
def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
    
@st.cache(allow_output_mutation=True)
def load_model(pickled_file=API_MODEL_PICKLE_FILE):
    #st.write('Cached function execution')
    print('Cache update')
    
    #Force garbage collector
    import gc
    gc.collect()
    
    if ('model' in globals()):
        print('Deleting previous model from memory : this code is never called')
        del model
    
    with open(pickled_file, 'rb') as f:
        model_object = pickle.load(f)
    
    #model = model_object['model']
    
    return(model_object)


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="template.csv">Download template file</a>'
    return(href)

_max_width_()

model_object = load_model(API_MODEL_PICKLE_FILE)
model = model_object['model']
model_agregate = model_object['model_agregate']
model_before_clustering = model_object['model_before_clustering']

st.title('Openclassrooms Data Science training project 5 : e-commerce clients segmentation') 
st.title('François BOYER')

st.write('\n')


image = Image.open(LOGO_IMAGE_FILE)
st.image(image,
         width=200)
         #use_column_width=True)


######################" Left Panel : options ##############################################"
st.sidebar.title('Order characteristics')


st.sidebar.markdown('By default, predictions are made from template input data')
st.sidebar.markdown('To change input data: download template file, update downloaded file with CSV/text editor, and import')

df_template = pd.read_csv(TEMPLATE_FILE, encoding='utf-8', converters={'InvoiceNo': str, 'StockCode':str, 'Description': str, \
                                   'CustomerID':str, 'Country': str})

st.sidebar.markdown(get_table_download_link(df_template), unsafe_allow_html=True)


uploaded_file = st.sidebar.file_uploader("Import CSV input file", type="csv")
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file, encoding='utf-8', converters={'InvoiceNo': str, 'StockCode':str, 'Description': str, \
                                       'CustomerID':str, 'Country': str})
    
    st.header('Input data for clustering : data loaded by user')    


else:
    df_input = df_template
    st.header('Input data applied for clustering : data loaded from default template')    
    st.write('(To change data : go the left panel)')

######################" Main Panel : prédictions ###############################"
st.header('Assigned clusters for input data :')    
series_predictions = model.predict(df_input)   

st.dataframe(pd.DataFrame(series_predictions).reset_index().rename(columns={0 : 'Cluster number'}), width=1000, height=1000)
#st.write(pd.DataFrame(series_predictions).rename(columns={0 : 'Cluster number'}))


####### Old code copied to UI_PJ5_oldcode.py ########"


######################" Left Panel : model analysis ##############################################"

st.sidebar.header('Model analysis')

display_clustesrfeatures = st.sidebar.checkbox('Clusters/features distribution', value=False)

if (display_clustesrfeatures == True):
    st.header('Distribution of features accross clusters')
    st.write('25-75% of the values are located in colored part of the boxes below')
    for feature in FEATURE_FILE_NAMES:
        image = Image.open(feature)
        st.image(image, width=800)

display_interpretation_tree = st.sidebar.checkbox('Interpretation tree', value=False)

if (display_interpretation_tree == True):
    st.header('Tree interpretation of clusters (simplified to depth 3)')
    st.write('This tree helps to interpret which main steps are processed by the model in most cases when it guesses which cluster to assign to a client')
    image = Image.open(GRAPH_MODEL_FILE)
    st.image(image)
    #st.image(image,
    #         width=2251)
    

debug_mode = st.sidebar.checkbox('Display debug data', value=False)

if (debug_mode == True):
    st.header('Step 0 : Input data')
    st.table(df_input)
    
    st.header('Step 1 : Data agregated at client level (Unscaled feature)')
    df_clients_agg = model_agregate.transform(df_input)
    st.table(df_clients_agg.reset_index())
    
    st.header('Step 2 : Data before clustering, at client level (scaled features)')
    df_clients = model_before_clustering.transform(df_input)
    st.table(df_clients.reset_index())


del model_object
# Manual memory cleaning at the of the program is necessary to avoid memory leak 
# (due to streamlit bug ?)
import gc
gc.collect()







