#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:42:51 2020

@author: francois
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

st.title('Openclassrooms training projet 5 : e-commerce clients segmentation (François BOYER)') 

st.write('\n')

'''
image = Image.open('plane_image.png')
st.image(image,
         width=300, format='PNG')
         #use_column_width=True)
'''


st.sidebar.title('Order characteristics')


st.sidebar.markdown('By default, predictions are made from template input data')
st.sidebar.markdown('To change input data: export csv template, update csv, and import')

df_template = pd.read_csv('UI_input_sample.csv', encoding='utf-8', converters={'InvoiceNo': str, 'StockCode':str, 'Description': str, \
                                   'CustomerID':str, 'Country': str})

st.sidebar.markdown(get_table_download_link(df_template), unsafe_allow_html=True)


uploaded_file = st.sidebar.file_uploader("Import CSV input file", type="csv")
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file, encoding='utf-8', converters={'InvoiceNo': str, 'StockCode':str, 'Description': str, \
                                       'CustomerID':str, 'Country': str})


else:
    df_input = df_template


st.header('Assigned clusters')    
series_predictions = model.predict(df_input)   

st.dataframe(pd.DataFrame(series_predictions).reset_index().rename(columns={0 : 'Cluster number'}), width=1000, height=1000)
#st.write(pd.DataFrame(series_predictions).rename(columns={0 : 'Cluster number'}))


df_clients = model_before_clustering.transform(df_input)

reductor = DimensionalityReductor(algorithm_to_use='TSNE', n_dim=2, features_totransform='ALL')
df_clients_reduced = reductor.fit_transform(df_clients)

#chart = alt.Chart(df_clients_reduced, width=1000, height=600).mark_circle().encode(alt.X('0'), alt.Y('1')).properties(title=f'Clients')

df_clients_reduced.rename(columns={0: 'X', 1 : 'Y'}, inplace=True)

st.table(df_clients)

plt.rcParams["figure.figsize"] = [16,9]
plt.scatter(df_clients_reduced['X'], df_clients_reduced['Y'])
st.pyplot()

chart = alt.Chart(df_clients_reduced).mark_circle(size=60).encode(
    x='X',
    y='Y',
).interactive()


text = chart.mark_text(
    align='left',
    baseline='middle',
    dx=7
).encode(
    text='label'
)
st.altair_chart(chart + text, use_container_width=True)

st.header('Tree interpretation of clusters (simplified to depth 3)')
image = Image.open('graph_model2.png')
st.image(image,
         width=2251)
    
    
debug_mode = st.checkbox('Display input data', value=False)

if (debug_mode == True):
    st.header('Input data')
    st.table(df_input)











