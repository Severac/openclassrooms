#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:42:51 2020

@author: francois


PJ6 Openclassrooms : this code is the user interface, that acts as an API to call the model 
It takes object and body of a post as input,  and outputs predicted tags for that post

Interface will be hosted here :  https://pj6.analysons.com/

See "README API.txt" for installation instructions :
    < Insert GitHub link>

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

#API_MODEL_PICKLE_FILE = 'API_model_PJ6.pickle'
API_MODEL_PICKLE_FILE = 'knn_modelprediction_pipeline.pickle'

LOGO_IMAGE_FILE = 'so-logo.png'

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
    
class PrepareTextData(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fitted = False
    
    def fit(self, df, labels=None):      
        if (DEBUG_LEVEL >= 1) :
            print('PrepareTextData : Fit data')
            
        self.fitted = True
        
        return self
    
    def transform(self, df):
        if (DEBUG_LEVEL >= 1) :
            print('PrepareTextData : Transform data')
            
        if (self.fitted == False):
            self.fit(df)
        
        df.loc[:, 'Body'] = df['Body'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
        df.loc[:, 'Title'].fillna(value='', inplace=True)
        df.loc[:, 'all_text'] = df['Title'].astype(str) + '. ' +  df['Body'].astype(str)
       
        #df.loc[:, 'all_text'] = (df['Title'].astype(str) + '. ' +  df['Body'].astype(str)).copy(deep=True)
        
        return(df[['all_text']])

_max_width_()

model = load_model(API_MODEL_PICKLE_FILE)
#model_agregate = model_object['model_agregate']
#model_before_clustering = model_object['model_before_clustering']

st.title('Openclassrooms Data Science training project 6 : categorize questions') 
st.title('François BOYER')

st.write('\n')


image = Image.open(LOGO_IMAGE_FILE)
st.image(image,
         width=400)
         #use_column_width=True)


######################" Left Panel : options ##############################################"
st.sidebar.title('Model analysis')


######################" Main Panel : prédictions ###############################"
st.header('Enter object and body ')    

post_object = str(st.text_input('Post object:'))
post_body = str(st.text_area('Post body:'))



df_input = pd.DataFrame([[post_object, post_body]],\
                        columns=['Title', 'Body'],)



######################" Left Panel : model analysis ##############################################"

debug_mode = st.sidebar.checkbox('Display debug data', value=False)

if (debug_mode == True):
    st.header('Step 0 : Input data')
    st.table(df_input)
    
    st.header('Step 1 : Data transformed')
    df_input_transformed = PrepareTextData().fit_transform(df_input)
    st.table(df_input_transformed)
    
    
del model
# Manual memory cleaning at the of the program is necessary to avoid memory leak 
# (due to streamlit bug ?)
import gc
gc.collect()







