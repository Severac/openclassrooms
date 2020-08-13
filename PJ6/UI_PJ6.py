#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:42:51 2020

@author: francois


PJ6 Openclassrooms : this code is the user interface, that acts as an API to call the model 
It takes object and body of a post as input,  and outputs predicted tags for that post

Interface will be hosted here :  https://pj6.analysons.com/

See "README API.txt" for installation instructions :
    https://github.com/Severac/openclassrooms/blob/master/PJ6/README%20API.txt

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
API_MODEL_PICKLE_FILE = 'grid_search_results_gridsearch_PJ6.pickle'
PREDICTIONS_TRIGGER = 0.3

LOGO_IMAGE_FILE = 'so-logo.png'

# Possible tag values (columns of prediction dataframe)
TAG_VALUES = ['Tags_.htaccess', 'Tags_.net', 'Tags_.net-core', 'Tags_ajax', 'Tags_algorithm', 'Tags_amazon-ec2', 'Tags_amazon-s3', 'Tags_amazon-web-services', 'Tags_anaconda', 'Tags_android', 'Tags_android-fragments', 'Tags_android-layout', 'Tags_android-recyclerview', 'Tags_android-studio', 'Tags_angular', 'Tags_angular-material', 'Tags_angular6', 'Tags_angular7', 'Tags_angularjs', 'Tags_animation', 'Tags_ansible', 'Tags_apache', 'Tags_apache-kafka', 'Tags_apache-spark', 'Tags_apache-spark-sql', 'Tags_api', 'Tags_arraylist', 'Tags_arrays', 'Tags_asp.net', 'Tags_asp.net-core', 'Tags_asp.net-core-mvc', 'Tags_asp.net-mvc', 'Tags_asp.net-web-api', 'Tags_assembly', 'Tags_async-await', 'Tags_asynchronous', 'Tags_authentication', 'Tags_automation', 'Tags_awk', 'Tags_aws-lambda', 'Tags_axios', 'Tags_azure', 'Tags_azure-active-directory', 'Tags_azure-devops', 'Tags_azure-functions', 'Tags_bash', 'Tags_batch-file', 'Tags_beautifulsoup', 'Tags_bootstrap-4', 'Tags_button', 'Tags_c', 'Tags_c#', 'Tags_c++', 'Tags_c++11', 'Tags_c++17', 'Tags_caching', 'Tags_canvas', 'Tags_class', 'Tags_cmake', 'Tags_cmd', 'Tags_codeigniter', 'Tags_cordova', 'Tags_css', 'Tags_csv', 'Tags_curl', 'Tags_d3.js', 'Tags_dart', 'Tags_data-structures', 'Tags_database', 'Tags_dataframe', 'Tags_datatable', 'Tags_datatables', 'Tags_date', 'Tags_datetime', 'Tags_debugging', 'Tags_deep-learning', 'Tags_delphi', 'Tags_dependency-injection', 'Tags_design-patterns', 'Tags_dictionary', 'Tags_django', 'Tags_django-forms', 'Tags_django-models', 'Tags_django-rest-framework', 'Tags_django-views', 'Tags_docker', 'Tags_docker-compose', 'Tags_dom', 'Tags_dplyr', 'Tags_eclipse', 'Tags_ecmascript-6', 'Tags_elasticsearch', 'Tags_electron', 'Tags_eloquent', 'Tags_email', 'Tags_encryption', 'Tags_entity-framework', 'Tags_entity-framework-core', 'Tags_excel', 'Tags_excel-formula', 'Tags_exception', 'Tags_expo', 'Tags_express', 'Tags_facebook', 'Tags_ffmpeg', 'Tags_file', 'Tags_filter', 'Tags_firebase', 'Tags_firebase-authentication', 'Tags_firebase-realtime-database', 'Tags_flask', 'Tags_flexbox', 'Tags_flutter', 'Tags_for-loop', 'Tags_forms', 'Tags_function', 'Tags_gcc', 'Tags_generics', 'Tags_ggplot2', 'Tags_git', 'Tags_github', 'Tags_gitlab', 'Tags_go', 'Tags_google-api', 'Tags_google-app-engine', 'Tags_google-apps-script', 'Tags_google-bigquery', 'Tags_google-chrome', 'Tags_google-chrome-extension', 'Tags_google-cloud-firestore', 'Tags_google-cloud-functions', 'Tags_google-cloud-platform', 'Tags_google-maps', 'Tags_google-sheets', 'Tags_gradle', 'Tags_graphql', 'Tags_groovy', 'Tags_group-by', 'Tags_hadoop', 'Tags_haskell', 'Tags_heroku', 'Tags_hibernate', 'Tags_highcharts', 'Tags_hive', 'Tags_html', 'Tags_http', 'Tags_https', 'Tags_if-statement', 'Tags_iis', 'Tags_image', 'Tags_image-processing', 'Tags_import', 'Tags_indexing', 'Tags_inheritance', 'Tags_input', 'Tags_intellij-idea', 'Tags_ionic-framework', 'Tags_ionic3', 'Tags_ionic4', 'Tags_ios', 'Tags_java', 'Tags_java-8', 'Tags_java-stream', 'Tags_javafx', 'Tags_javascript', 'Tags_jdbc', 'Tags_jenkins', 'Tags_jenkins-pipeline', 'Tags_jestjs', 'Tags_jmeter', 'Tags_join', 'Tags_jpa', 'Tags_jquery', 'Tags_json', 'Tags_junit', 'Tags_jupyter-notebook', 'Tags_jwt', 'Tags_keras', 'Tags_kotlin', 'Tags_kubernetes', 'Tags_lambda', 'Tags_laravel', 'Tags_laravel-5', 'Tags_laravel-5.7', 'Tags_linq', 'Tags_linux', 'Tags_list', 'Tags_listview', 'Tags_logging', 'Tags_loops', 'Tags_machine-learning', 'Tags_macos', 'Tags_mariadb', 'Tags_math', 'Tags_matlab', 'Tags_matplotlib', 'Tags_matrix', 'Tags_maven', 'Tags_memory', 'Tags_merge', 'Tags_model-view-controller', 'Tags_mongodb', 'Tags_mongoose', 'Tags_ms-access', 'Tags_multidimensional-array', 'Tags_multithreading', 'Tags_mvvm', 'Tags_mysql', 'Tags_mysqli', 'Tags_nativescript', 'Tags_neo4j', 'Tags_networking', 'Tags_neural-network', 'Tags_nginx', 'Tags_nlp', 'Tags_node.js', 'Tags_npm', 'Tags_numpy', 'Tags_oauth-2.0', 'Tags_object', 'Tags_objective-c', 'Tags_oop', 'Tags_opencv', 'Tags_opengl', 'Tags_optimization', 'Tags_oracle', 'Tags_outlook', 'Tags_pandas', 'Tags_pandas-groupby', 'Tags_parsing', 'Tags_pdf', 'Tags_performance', 'Tags_perl', 'Tags_php', 'Tags_pip', 'Tags_plot', 'Tags_plsql', 'Tags_pointers', 'Tags_post', 'Tags_postgresql', 'Tags_powerbi', 'Tags_powershell', 'Tags_promise', 'Tags_pycharm', 'Tags_pygame', 'Tags_pyqt5', 'Tags_pyspark', 'Tags_python', 'Tags_python-2.7', 'Tags_python-3.x', 'Tags_python-requests', 'Tags_pytorch', 'Tags_qt', 'Tags_r', 'Tags_random', 'Tags_razor', 'Tags_react-native', 'Tags_react-redux', 'Tags_react-router', 'Tags_reactjs', 'Tags_recursion', 'Tags_redis', 'Tags_redux', 'Tags_regex', 'Tags_rest', 'Tags_ruby', 'Tags_ruby-on-rails', 'Tags_ruby-on-rails-5', 'Tags_rust', 'Tags_rxjs', 'Tags_sass', 'Tags_scala', 'Tags_scikit-learn', 'Tags_scipy', 'Tags_scrapy', 'Tags_search', 'Tags_security', 'Tags_sed', 'Tags_select', 'Tags_selenium', 'Tags_selenium-webdriver', 'Tags_server', 'Tags_session', 'Tags_shell', 'Tags_shiny', 'Tags_socket.io', 'Tags_sockets', 'Tags_sorting', 'Tags_spring', 'Tags_spring-boot', 'Tags_spring-data-jpa', 'Tags_spring-mvc', 'Tags_spring-security', 'Tags_sql', 'Tags_sql-server', 'Tags_sqlalchemy', 'Tags_sqlite', 'Tags_ssh', 'Tags_ssis', 'Tags_ssl', 'Tags_stored-procedures', 'Tags_string', 'Tags_struct', 'Tags_svg', 'Tags_swift', 'Tags_swing', 'Tags_symfony', 'Tags_templates', 'Tags_tensorflow', 'Tags_terminal', 'Tags_testing', 'Tags_text', 'Tags_three.js', 'Tags_time', 'Tags_tkinter', 'Tags_tomcat', 'Tags_tsql', 'Tags_twitter-bootstrap', 'Tags_types', 'Tags_typescript', 'Tags_ubuntu', 'Tags_uitableview', 'Tags_unit-testing', 'Tags_unity3d', 'Tags_unix', 'Tags_url', 'Tags_user-interface', 'Tags_uwp', 'Tags_validation', 'Tags_variables', 'Tags_vb.net', 'Tags_vba', 'Tags_vector', 'Tags_visual-studio', 'Tags_visual-studio-2017', 'Tags_visual-studio-code', 'Tags_vue-component', 'Tags_vue.js', 'Tags_vuejs2', 'Tags_web', 'Tags_web-scraping', 'Tags_webpack', 'Tags_websocket', 'Tags_winapi', 'Tags_windows', 'Tags_winforms', 'Tags_woocommerce', 'Tags_wordpress', 'Tags_wpf', 'Tags_xamarin', 'Tags_xamarin.android', 'Tags_xamarin.forms', 'Tags_xaml', 'Tags_xcode', 'Tags_xml', 'Tags_xpath', 'Tags_xslt']

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


df_input_transformed = PrepareTextData().fit_transform(df_input)

#df_predictions = pd.DataFrame(model.predict(df_input_transformed), columns=TAG_VALUES)
predictions_proba = model.predict_proba(df_input_transformed)
df_predictions_proba = pd.DataFrame(np.array(predictions_proba)[:, :, 1].T, columns=TAG_VALUES)
df_predictions = pd.DataFrame(np.where(df_predictions_proba >= PREDICTIONS_TRIGGER, 1, 0), columns=TAG_VALUES)

df_predictions_todisplay = df_predictions.loc[:, df_predictions.gt(0).any() ]
df_predictions_proba_todisplay = df_predictions_proba.loc[:, df_predictions_proba.gt(0).any() ]

st.header('Model predictions :')

debug_mode = st.sidebar.checkbox('Display debug data', value=False)

if ((post_object != '') and (post_body != '')):
    #st.table(df_predictions.sum(axis=1))
    
    st.write(df_predictions_todisplay)

    st.header('Tag probabilities')
    st.table(df_predictions_proba_todisplay)
    
    ######################" Left Panel : model analysis ##############################################"
    
    if (debug_mode == True):
        st.header('Step 0 : Input data')
        st.table(df_input)
        
        st.header('Step 1 : Data transformed')
        st.table(df_input_transformed)
        
else:
    st.write('Please enter text above to get tag predictions')
        
del model
# Manual memory cleaning at the of the program is necessary to avoid memory leak 
# (due to streamlit bug ?)
import gc
gc.collect()







