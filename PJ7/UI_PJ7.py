#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:42:51 2020

@author: francois


PJ7 Openclassrooms : this code is the user interface, that acts as an API to call the model 
It takes a picture of a dog as input, and outputs predicted probabilities of the race appartenance of the dog

Interface will be hosted here :  https://pj7.analysons.com/

See "README API.txt" for installation instructions :


"""

import streamlit as st
import pickle

from PIL import Image

#from functions import *

import pandas as pd
import numpy as np
import base64

import altair as alt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import decode_predictions

import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *

API_MODEL_PATH = 'model_endsave'
#PREDICTIONS_TRIGGER = 0.3

LOGO_IMAGE_FILE = 'logo-chien.jpg'

# Possible tag values (columns of prediction dataframe)
TAG_VALUES = ['Tags_.htaccess', 'Tags_.net', 'Tags_.net-core', 'Tags_ajax', 'Tags_algorithm', 'Tags_amazon-ec2', 'Tags_amazon-s3', 'Tags_amazon-web-services', 'Tags_anaconda', 'Tags_android', 'Tags_android-fragments', 'Tags_android-layout', 'Tags_android-recyclerview', 'Tags_android-studio', 'Tags_angular', 'Tags_angular-material', 'Tags_angular6', 'Tags_angular7', 'Tags_angularjs', 'Tags_animation', 'Tags_ansible', 'Tags_apache', 'Tags_apache-kafka', 'Tags_apache-spark', 'Tags_apache-spark-sql', 'Tags_api', 'Tags_arraylist', 'Tags_arrays', 'Tags_asp.net', 'Tags_asp.net-core', 'Tags_asp.net-core-mvc', 'Tags_asp.net-mvc', 'Tags_asp.net-web-api', 'Tags_assembly', 'Tags_async-await', 'Tags_asynchronous', 'Tags_authentication', 'Tags_automation', 'Tags_awk', 'Tags_aws-lambda', 'Tags_axios', 'Tags_azure', 'Tags_azure-active-directory', 'Tags_azure-devops', 'Tags_azure-functions', 'Tags_bash', 'Tags_batch-file', 'Tags_beautifulsoup', 'Tags_bootstrap-4', 'Tags_button', 'Tags_c', 'Tags_c#', 'Tags_c++', 'Tags_c++11', 'Tags_c++17', 'Tags_caching', 'Tags_canvas', 'Tags_class', 'Tags_cmake', 'Tags_cmd', 'Tags_codeigniter', 'Tags_cordova', 'Tags_css', 'Tags_csv', 'Tags_curl', 'Tags_d3.js', 'Tags_dart', 'Tags_data-structures', 'Tags_database', 'Tags_dataframe', 'Tags_datatable', 'Tags_datatables', 'Tags_date', 'Tags_datetime', 'Tags_debugging', 'Tags_deep-learning', 'Tags_delphi', 'Tags_dependency-injection', 'Tags_design-patterns', 'Tags_dictionary', 'Tags_django', 'Tags_django-forms', 'Tags_django-models', 'Tags_django-rest-framework', 'Tags_django-views', 'Tags_docker', 'Tags_docker-compose', 'Tags_dom', 'Tags_dplyr', 'Tags_eclipse', 'Tags_ecmascript-6', 'Tags_elasticsearch', 'Tags_electron', 'Tags_eloquent', 'Tags_email', 'Tags_encryption', 'Tags_entity-framework', 'Tags_entity-framework-core', 'Tags_excel', 'Tags_excel-formula', 'Tags_exception', 'Tags_expo', 'Tags_express', 'Tags_facebook', 'Tags_ffmpeg', 'Tags_file', 'Tags_filter', 'Tags_firebase', 'Tags_firebase-authentication', 'Tags_firebase-realtime-database', 'Tags_flask', 'Tags_flexbox', 'Tags_flutter', 'Tags_for-loop', 'Tags_forms', 'Tags_function', 'Tags_gcc', 'Tags_generics', 'Tags_ggplot2', 'Tags_git', 'Tags_github', 'Tags_gitlab', 'Tags_go', 'Tags_google-api', 'Tags_google-app-engine', 'Tags_google-apps-script', 'Tags_google-bigquery', 'Tags_google-chrome', 'Tags_google-chrome-extension', 'Tags_google-cloud-firestore', 'Tags_google-cloud-functions', 'Tags_google-cloud-platform', 'Tags_google-maps', 'Tags_google-sheets', 'Tags_gradle', 'Tags_graphql', 'Tags_groovy', 'Tags_group-by', 'Tags_hadoop', 'Tags_haskell', 'Tags_heroku', 'Tags_hibernate', 'Tags_highcharts', 'Tags_hive', 'Tags_html', 'Tags_http', 'Tags_https', 'Tags_if-statement', 'Tags_iis', 'Tags_image', 'Tags_image-processing', 'Tags_import', 'Tags_indexing', 'Tags_inheritance', 'Tags_input', 'Tags_intellij-idea', 'Tags_ionic-framework', 'Tags_ionic3', 'Tags_ionic4', 'Tags_ios', 'Tags_java', 'Tags_java-8', 'Tags_java-stream', 'Tags_javafx', 'Tags_javascript', 'Tags_jdbc', 'Tags_jenkins', 'Tags_jenkins-pipeline', 'Tags_jestjs', 'Tags_jmeter', 'Tags_join', 'Tags_jpa', 'Tags_jquery', 'Tags_json', 'Tags_junit', 'Tags_jupyter-notebook', 'Tags_jwt', 'Tags_keras', 'Tags_kotlin', 'Tags_kubernetes', 'Tags_lambda', 'Tags_laravel', 'Tags_laravel-5', 'Tags_laravel-5.7', 'Tags_linq', 'Tags_linux', 'Tags_list', 'Tags_listview', 'Tags_logging', 'Tags_loops', 'Tags_machine-learning', 'Tags_macos', 'Tags_mariadb', 'Tags_math', 'Tags_matlab', 'Tags_matplotlib', 'Tags_matrix', 'Tags_maven', 'Tags_memory', 'Tags_merge', 'Tags_model-view-controller', 'Tags_mongodb', 'Tags_mongoose', 'Tags_ms-access', 'Tags_multidimensional-array', 'Tags_multithreading', 'Tags_mvvm', 'Tags_mysql', 'Tags_mysqli', 'Tags_nativescript', 'Tags_neo4j', 'Tags_networking', 'Tags_neural-network', 'Tags_nginx', 'Tags_nlp', 'Tags_node.js', 'Tags_npm', 'Tags_numpy', 'Tags_oauth-2.0', 'Tags_object', 'Tags_objective-c', 'Tags_oop', 'Tags_opencv', 'Tags_opengl', 'Tags_optimization', 'Tags_oracle', 'Tags_outlook', 'Tags_pandas', 'Tags_pandas-groupby', 'Tags_parsing', 'Tags_pdf', 'Tags_performance', 'Tags_perl', 'Tags_php', 'Tags_pip', 'Tags_plot', 'Tags_plsql', 'Tags_pointers', 'Tags_post', 'Tags_postgresql', 'Tags_powerbi', 'Tags_powershell', 'Tags_promise', 'Tags_pycharm', 'Tags_pygame', 'Tags_pyqt5', 'Tags_pyspark', 'Tags_python', 'Tags_python-2.7', 'Tags_python-3.x', 'Tags_python-requests', 'Tags_pytorch', 'Tags_qt', 'Tags_r', 'Tags_random', 'Tags_razor', 'Tags_react-native', 'Tags_react-redux', 'Tags_react-router', 'Tags_reactjs', 'Tags_recursion', 'Tags_redis', 'Tags_redux', 'Tags_regex', 'Tags_rest', 'Tags_ruby', 'Tags_ruby-on-rails', 'Tags_ruby-on-rails-5', 'Tags_rust', 'Tags_rxjs', 'Tags_sass', 'Tags_scala', 'Tags_scikit-learn', 'Tags_scipy', 'Tags_scrapy', 'Tags_search', 'Tags_security', 'Tags_sed', 'Tags_select', 'Tags_selenium', 'Tags_selenium-webdriver', 'Tags_server', 'Tags_session', 'Tags_shell', 'Tags_shiny', 'Tags_socket.io', 'Tags_sockets', 'Tags_sorting', 'Tags_spring', 'Tags_spring-boot', 'Tags_spring-data-jpa', 'Tags_spring-mvc', 'Tags_spring-security', 'Tags_sql', 'Tags_sql-server', 'Tags_sqlalchemy', 'Tags_sqlite', 'Tags_ssh', 'Tags_ssis', 'Tags_ssl', 'Tags_stored-procedures', 'Tags_string', 'Tags_struct', 'Tags_svg', 'Tags_swift', 'Tags_swing', 'Tags_symfony', 'Tags_templates', 'Tags_tensorflow', 'Tags_terminal', 'Tags_testing', 'Tags_text', 'Tags_three.js', 'Tags_time', 'Tags_tkinter', 'Tags_tomcat', 'Tags_tsql', 'Tags_twitter-bootstrap', 'Tags_types', 'Tags_typescript', 'Tags_ubuntu', 'Tags_uitableview', 'Tags_unit-testing', 'Tags_unity3d', 'Tags_unix', 'Tags_url', 'Tags_user-interface', 'Tags_uwp', 'Tags_validation', 'Tags_variables', 'Tags_vb.net', 'Tags_vba', 'Tags_vector', 'Tags_visual-studio', 'Tags_visual-studio-2017', 'Tags_visual-studio-code', 'Tags_vue-component', 'Tags_vue.js', 'Tags_vuejs2', 'Tags_web', 'Tags_web-scraping', 'Tags_webpack', 'Tags_websocket', 'Tags_winapi', 'Tags_windows', 'Tags_winforms', 'Tags_woocommerce', 'Tags_wordpress', 'Tags_wpf', 'Tags_xamarin', 'Tags_xamarin.android', 'Tags_xamarin.forms', 'Tags_xaml', 'Tags_xcode', 'Tags_xml', 'Tags_xpath', 'Tags_xslt']

TARGETX = 224
TARGETY = 224

breed_list = ((0, 'n02085620-Chihuahua'),
 (1, 'n02085782-Japanese_spaniel'),
 (2, 'n02085936-Maltese_dog'),
 (3, 'n02086079-Pekinese'),
 (4, 'n02086240-Shih-Tzu'),
 (5, 'n02086646-Blenheim_spaniel'),
 (6, 'n02086910-papillon'),
 (7, 'n02087046-toy_terrier'),
 (8, 'n02087394-Rhodesian_ridgeback'),
 (9, 'n02088094-Afghan_hound'),
 (10, 'n02088238-basset'),
 (11, 'n02088364-beagle'),
 (12, 'n02088466-bloodhound'),
 (13, 'n02088632-bluetick'),
 (14, 'n02089078-black-and-tan_coonhound'),
 (15, 'n02089867-Walker_hound'),
 (16, 'n02089973-English_foxhound'),
 (17, 'n02090379-redbone'),
 (18, 'n02090622-borzoi'),
 (19, 'n02090721-Irish_wolfhound'),
 (20, 'n02091032-Italian_greyhound'),
 (21, 'n02091134-whippet'),
 (22, 'n02091244-Ibizan_hound'),
 (23, 'n02091467-Norwegian_elkhound'),
 (24, 'n02091635-otterhound'),
 (25, 'n02091831-Saluki'),
 (26, 'n02092002-Scottish_deerhound'),
 (27, 'n02092339-Weimaraner'),
 (28, 'n02093256-Staffordshire_bullterrier'),
 (29, 'n02093428-American_Staffordshire_terrier'),
 (30, 'n02093647-Bedlington_terrier'),
 (31, 'n02093754-Border_terrier'),
 (32, 'n02093859-Kerry_blue_terrier'),
 (33, 'n02093991-Irish_terrier'),
 (34, 'n02094114-Norfolk_terrier'),
 (35, 'n02094258-Norwich_terrier'),
 (36, 'n02094433-Yorkshire_terrier'),
 (37, 'n02095314-wire-haired_fox_terrier'),
 (38, 'n02095570-Lakeland_terrier'),
 (39, 'n02095889-Sealyham_terrier'),
 (40, 'n02096051-Airedale'),
 (41, 'n02096177-cairn'),
 (42, 'n02096294-Australian_terrier'),
 (43, 'n02096437-Dandie_Dinmont'),
 (44, 'n02096585-Boston_bull'),
 (45, 'n02097047-miniature_schnauzer'),
 (46, 'n02097130-giant_schnauzer'),
 (47, 'n02097209-standard_schnauzer'),
 (48, 'n02097298-Scotch_terrier'),
 (49, 'n02097474-Tibetan_terrier'),
 (50, 'n02097658-silky_terrier'),
 (51, 'n02098105-soft-coated_wheaten_terrier'),
 (52, 'n02098286-West_Highland_white_terrier'),
 (53, 'n02098413-Lhasa'),
 (54, 'n02099267-flat-coated_retriever'),
 (55, 'n02099429-curly-coated_retriever'),
 (56, 'n02099601-golden_retriever'),
 (57, 'n02099712-Labrador_retriever'),
 (58, 'n02099849-Chesapeake_Bay_retriever'),
 (59, 'n02100236-German_short-haired_pointer'),
 (60, 'n02100583-vizsla'),
 (61, 'n02100735-English_setter'),
 (62, 'n02100877-Irish_setter'),
 (63, 'n02101006-Gordon_setter'),
 (64, 'n02101388-Brittany_spaniel'),
 (65, 'n02101556-clumber'),
 (66, 'n02102040-English_springer'),
 (67, 'n02102177-Welsh_springer_spaniel'),
 (68, 'n02102318-cocker_spaniel'),
 (69, 'n02102480-Sussex_spaniel'),
 (70, 'n02102973-Irish_water_spaniel'),
 (71, 'n02104029-kuvasz'),
 (72, 'n02104365-schipperke'),
 (73, 'n02105056-groenendael'),
 (74, 'n02105162-malinois'),
 (75, 'n02105251-briard'),
 (76, 'n02105412-kelpie'),
 (77, 'n02105505-komondor'),
 (78, 'n02105641-Old_English_sheepdog'),
 (79, 'n02105855-Shetland_sheepdog'),
 (80, 'n02106030-collie'),
 (81, 'n02106166-Border_collie'),
 (82, 'n02106382-Bouvier_des_Flandres'),
 (83, 'n02106550-Rottweiler'),
 (84, 'n02106662-German_shepherd'),
 (85, 'n02107142-Doberman'),
 (86, 'n02107312-miniature_pinscher'),
 (87, 'n02107574-Greater_Swiss_Mountain_dog'),
 (88, 'n02107683-Bernese_mountain_dog'),
 (89, 'n02107908-Appenzeller'),
 (90, 'n02108000-EntleBucher'),
 (91, 'n02108089-boxer'),
 (92, 'n02108422-bull_mastiff'),
 (93, 'n02108551-Tibetan_mastiff'),
 (94, 'n02108915-French_bulldog'),
 (95, 'n02109047-Great_Dane'),
 (96, 'n02109525-Saint_Bernard'),
 (97, 'n02109961-Eskimo_dog'),
 (98, 'n02110063-malamute'),
 (99, 'n02110185-Siberian_husky'),
 (100, 'n02110627-affenpinscher'),
 (101, 'n02110806-basenji'),
 (102, 'n02110958-pug'),
 (103, 'n02111129-Leonberg'),
 (104, 'n02111277-Newfoundland'),
 (105, 'n02111500-Great_Pyrenees'),
 (106, 'n02111889-Samoyed'),
 (107, 'n02112018-Pomeranian'),
 (108, 'n02112137-chow'),
 (109, 'n02112350-keeshond'),
 (110, 'n02112706-Brabancon_griffon'),
 (111, 'n02113023-Pembroke'),
 (112, 'n02113186-Cardigan'),
 (113, 'n02113624-toy_poodle'),
 (114, 'n02113712-miniature_poodle'),
 (115, 'n02113799-standard_poodle'),
 (116, 'n02113978-Mexican_hairless'),
 (117, 'n02115641-dingo'),
 (118, 'n02115913-dhole'),
 (119, 'n02116738-African_hunting_dog'))

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
def load_model_instreamlit(model_path=API_MODEL_PATH):
    #st.write('Cached function execution')
    print('Cache update')
    
    #Force garbage collector
    import gc
    gc.collect()
    
    if ('model' in globals()):
        print('Deleting previous model from memory : this code is never called in theory, just in case since streamlit is hasardous about memory handling')
        del model
    
    model = load_model(API_MODEL_PATH)
        
    return(model)

_max_width_()

model = load_model_instreamlit(API_MODEL_PATH)

st.title('Openclassrooms Data Science training project 7 : recognize dog races') 
st.title('François BOYER')

st.write('\n')


image = Image.open(LOGO_IMAGE_FILE)
st.image(image,
         width=400)
         #use_column_width=True)


######################" Left Panel : options ##############################################"
st.sidebar.title('Model analysis')
debug_mode = st.sidebar.checkbox('Display feat maps of 1st conv layer', value=False)

######################" Main Panel : prédictions ###############################"
st.header('Upload image')    

uploaded_image = st.file_uploader('Upload a dog photo here (JPEG or PNG. real photo, not drawing)', encoding=None)

if (uploaded_image != None):
	#st.image(mpimg.imread(uploaded_image))
	st.image(uploaded_image, width=400)

	img = Image.open(uploaded_image).convert('RGB').resize((TARGETX, TARGETY))
	
	print(img.size)
	
	
	print(img)
	
	probabilities = model.predict(preprocess_input(np.expand_dims(img, axis=0)))

	for i in probabilities[0].argsort()[-5:][::-1]: 
	    st.write(probabilities[0][i], "  :  " , breed_list[i])


	if (debug_mode == True):
		#fig, ax = plt.subplots()
		fig = plt.figure()
		
		model = Model(inputs=model.inputs, outputs=model.layers[4].output)	

		model.summary()
		# load the image with the required shape
		
		# convert the image to an array
		img = img_to_array(img)
		# expand dimensions so that it represents a single 'sample'
		img = np.expand_dims(img, axis=0)
		# prepare the image (e.g. scale pixel values for the vgg)
		img = preprocess_input(img)
		# get feature map for first hidden layer
		feature_maps = model.predict(img)
		# plot 32 maps of 1st conv layer
		display_lines = 4
		display_cols = 8
		ix = 1
		for _ in range(display_lines):
			for _ in range(display_cols):
				# specify subplot and turn of axis
				ax = plt.subplot(display_lines, display_cols, ix)
				ax.set_xticks([])
				ax.set_yticks([])
				# plot filter channel in grayscale
				plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
				ix += 1

		# show the figure
		#pyplot.show()
		st.pyplot(fig)

        
del model
# Manual memory cleaning at the of the program is necessary to avoid memory leak 
# (due to streamlit bug ?)
import gc
gc.collect()







