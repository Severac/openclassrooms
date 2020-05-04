import streamlit as st
import pickle

from PIL import Image

from functions import *

#import pandas as pd

API_MODEL_PICKLE_FILE = 'API_model_PJ4.pickle'

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
        model = pickle.load(f)
    
    return(model)


_max_width_()

st.title('Openclassrooms training projet 4 : predicting flight delays (François BOYER)') 

st.write('\n')

image = Image.open('plane_image.png')
st.image(image,
         width=300, format='PNG')
         #use_column_width=True)

st.sidebar.title('Flight characteristics')
crs_dep_time = st.sidebar.text_input('Scheduled Departure time (HHMM)', '1000')
crs_arr_time = st.sidebar.text_input('Scheduled Arrival time (HHMM)', '1800')


crs_elapsed_time = st.sidebar.number_input('Scheduled Elapsed time (minutes)', value=60, step=10)
day_of_month = st.sidebar.selectbox('Day of Month (1-31)', range(1,32))
day_of_week = st.sidebar.selectbox('Day of week (1-7)', range(1,8))
distance = st.sidebar.number_input('Distance (miles)', value=1000, step=100)
month = st.sidebar.selectbox('Month (1-12)', range(1,13))

origin_list = ['CLE', 'SFO', 'DSM', 'CLT', 'DEN', 'LAX', 'SJC', 'STL', 'SLC',
       'MCO', 'OKC', 'BTV', 'HOU', 'MIA', 'ATL', 'BOS', 'SYR', 'ONT',
       'TPA', 'SAN', 'BNA', 'DCA', 'SNA', 'LAS', 'SCC', 'SEA', 'PHX',
       'JFK', 'DFW', 'ORD', 'RDU', 'MSP', 'LGA', 'PSP', 'JAX', 'PHL',
       'AUS', 'MDW', 'BUR', 'CRW', 'ATW', 'EWR', 'GRK', 'TUS', 'SMF',
       'OAK', 'HNL', 'ANC', 'TUL', 'BRD', 'DAL', 'GEG', 'EWN', 'TRI',
       'RIC', 'IAH', 'FLL', 'SAT', 'GSP', 'SPS', 'CMH', 'DTW', 'CVG',
       'VPS', 'OGG', 'SDF', 'CHS', 'MSY', 'ELP', 'FCA', 'SBA', 'ACY',
       'SJU', 'BOI', 'PIT', 'LIH', 'CAK', 'PDX', 'LIT', 'MOB', 'BTR',
       'PVD', 'LRD', 'MYR', 'ALB', 'COS', 'LGB', 'MSN', 'RAP', 'IND',
       'GUC', 'CHA', 'JNU', 'GRR', 'GRB', 'BUF', 'PBI', 'ABE', 'MKE',
       'ORF', 'ICT', 'ABQ', 'IAD', 'RSW', 'GTF', 'LBB', 'BHM', 'PSC',
       'AVP', 'BDL', 'BWI', 'MCI', 'ITO', 'SPI', 'EGE', 'SBN', 'TTN',
       'CAE', 'OMA', 'ISP', 'MGM', 'FSM', 'FAT', 'COD', 'BZN', 'EYW',
       'PHF', 'HPN', 'MEI', 'TYS', 'LNK', 'MAF', 'FWA', 'SRQ', 'SHV',
       'PNS', 'HSV', 'CPR', 'PBG', 'MHT', 'GJT', 'ELM', 'AGS', 'JAC',
       'RNO', 'ROC', 'AZO', 'BRO', 'AEX', 'CRP', 'PIA', 'DAB', 'SAV',
       'BMI', 'KTN', 'RKS', 'PWM', 'OTZ', 'MLU', 'TWF', 'CHO', 'ECP',
       'BIL', 'MOT', 'ILM', 'MLI', 'KOA', 'ASE', 'EKO', 'MEM', 'CSG',
       'XNA', 'JAN', 'YUM', 'BGM', 'AVL', 'IDA', 'SBP', 'MRY', 'MDT',
       'RDM', 'GCK', 'BJI', 'HLN', 'FAR', 'EAU', 'MSO', 'ISN', 'TLH',
       'FAY', 'ACV', 'MFE', 'PSG', 'GSO', 'DAY', 'SAF', 'BIS', 'PIB',
       'TVC', 'BQN', 'HOB', 'AMA', 'RST', 'MTJ', 'GUM', 'OAJ', 'VLD',
       'CID', 'GCC', 'BGR', 'TYR', 'LAW', 'HIB', 'SGF', 'GPT', 'LFT',
       'SJT', 'HRL', 'DLH', 'STT', 'GTR', 'EVV', 'LEX', 'ABY', 'CLL',
       'MBS', 'STX', 'CDV', 'HDN', 'ERI', 'LAN', 'YAK', 'GNV', 'FLG',
       'DLG', 'GFK', 'ORH', 'CIU', 'RHI', 'LSE', 'ESC', 'MKG', 'PSE',
       'EUG', 'DRO', 'DHN', 'MLB', 'ADQ', 'HYS', 'ITH', 'LCH', 'ROW',
       'BFL', 'FSD', 'JLN', 'MQT', 'PIH', 'DVL', 'FNT', 'BRW', 'ABI',
       'JMS', 'BPT', 'FAI', 'HYA', 'TXK', 'ROA', 'UST', 'SGU', 'PGD',
       'MFR', 'GGG', 'MVY', 'ACT', 'GRI', 'APN', 'OTH', 'SWF', 'RDD',
       'SMX', 'SCE', 'ADK', 'SIT', 'LAR', 'GST', 'LBE', 'INL', 'IMT',
       'PLN', 'CMX', 'ACK', 'CWA', 'BET', 'BLI', 'OME', 'BTM', 'BQK',
       'LWS', 'PAH', 'SUN', 'WRG', 'CDC', 'PPG', 'MMH', 'ABR', 'AKN',
       'WYS']

origin_list.sort()

dest_list = ['LGA', 'JFK', 'ORD', 'ORF', 'BIS', 'SFO', 'PHX', 'MDW', 'SAN',
       'CLT', 'STL', 'OKC', 'ATL', 'MCO', 'HOU', 'EWR', 'DEN', 'CHS',
       'TPA', 'BRW', 'PSP', 'PDX', 'SEA', 'BOS', 'FLL', 'VPS', 'ROA',
       'DTW', 'BTV', 'DCA', 'GRR', 'IAD', 'IAH', 'JAC', 'PHL', 'LAS',
       'TUS', 'MCI', 'SJU', 'SMF', 'PVD', 'LAX', 'SLC', 'MIA', 'BOI',
       'MSP', 'FAT', 'AUS', 'OGG', 'BRO', 'SBP', 'ONT', 'SCC', 'BNA',
       'SJC', 'EUG', 'LAR', 'DAL', 'MMH', 'RSW', 'HNL', 'JAX', 'MKE',
       'PNS', 'SNA', 'ANC', 'DFW', 'BDL', 'IND', 'ATW', 'MEM', 'RDU',
       'BQK', 'BWI', 'CVG', 'RNO', 'DAY', 'PBI', 'LGB', 'TUL', 'CID',
       'LBB', 'LIT', 'ELP', 'BHM', 'BTR', 'KOA', 'SBA', 'SPI', 'LCH',
       'ABQ', 'PIT', 'ROC', 'SHV', 'SBN', 'MSY', 'CMH', 'CHA', 'OAK',
       'LNK', 'CLE', 'CHO', 'BUF', 'ASE', 'GNV', 'MDT', 'HLN', 'GRB',
       'SAT', 'ALB', 'DLH', 'ITO', 'MSN', 'LEX', 'TYS', 'AEX', 'SRQ',
       'CRW', 'RIC', 'JAN', 'CRP', 'MFE', 'GSO', 'AVL', 'FWA', 'MOB',
       'MYR', 'HRL', 'PIH', 'BZN', 'SAV', 'FAR', 'RHI', 'MHT', 'COS',
       'DHN', 'OMA', 'FSM', 'PIB', 'SWF', 'CMX', 'JNU', 'EAU', 'PSG',
       'HPN', 'ECP', 'BGM', 'BUR', 'STT', 'SGF', 'BGR', 'RDM', 'SYR',
       'CAE', 'FAY', 'GPT', 'RAP', 'SJT', 'EGE', 'SAF', 'FAI', 'LIH',
       'PSC', 'ISP', 'EWN', 'YUM', 'TVC', 'ICT', 'XNA', 'EYW', 'FLG',
       'GEG', 'DSM', 'ILM', 'MKG', 'GTR', 'PWM', 'AGS', 'MLI', 'MGM',
       'IDA', 'MAF', 'GJT', 'SIT', 'ABY', 'LAW', 'ACV', 'RKS', 'OAJ',
       'LRD', 'AZO', 'CSG', 'GSP', 'TTN', 'GTF', 'MSO', 'GFK', 'SGU',
       'DLG', 'HYS', 'MEI', 'MLU', 'ACY', 'BMI', 'ADQ', 'MFR', 'AMA',
       'FCA', 'TLH', 'CAK', 'OME', 'BLI', 'PIA', 'BPT', 'ABE', 'FSD',
       'LFT', 'INL', 'BQN', 'MQT', 'BIL', 'JMS', 'FNT', 'ISN', 'SDF',
       'JLN', 'LAN', 'HOB', 'HSV', 'PSE', 'ELM', 'BTM', 'LBE', 'GCC',
       'WRG', 'SPS', 'IMT', 'KTN', 'SUN', 'TRI', 'ORH', 'DAB', 'MBS',
       'CDV', 'ACT', 'MRY', 'GRK', 'PLN', 'HIB', 'STX', 'BJI', 'RST',
       'CPR', 'MLB', 'ACK', 'CIU', 'BFL', 'MOT', 'SMX', 'MVY', 'DRO',
       'DVL', 'PAH', 'EKO', 'LWS', 'HDN', 'SCE', 'VLD', 'TYR', 'ERI',
       'MTJ', 'EVV', 'TXK', 'CLL', 'ROW', 'RDD', 'COD', 'APN', 'ITH',
       'PHF', 'GRI', 'BRD', 'AKN', 'YAK', 'CDC', 'WYS', 'TWF', 'CWA',
       'ABR', 'PBG', 'AVP', 'BET', 'ABI', 'IAG', 'GUM', 'GUC', 'GGG',
       'GCK', 'LSE', 'UST', 'OTZ', 'ESC', 'HYA', 'GST', 'OTH', 'PPG']

dest_list.sort()

carrier_list = ['AA', 'AS', 'B6', 'DL', 'F9', 'HA', 'EV', 'NK', 'OO', 'UA', 'VX',
       'WN']
carrier_list.sort()

origin = st.sidebar.selectbox('Origin airport', (origin_list))
dest = st.sidebar.selectbox('Destination airport', (dest_list))
carrier = st.sidebar.selectbox('Flight company', (carrier_list))


nbflights_forday_forairport = st.sidebar.number_input('Mean number of flights within the day, in the airport',  value=200,step=1)
nbflights_fordayhour_forairport = st.sidebar.number_input('Mean number of flights per hour within the day, in the airport',  value=200,step=1)

model = load_model(API_MODEL_PICKLE_FILE)

#st.write(model['dataprep'])


df_input = pd.DataFrame([[str(crs_arr_time), str(crs_dep_time), crs_elapsed_time, day_of_month, day_of_week, distance, month, origin, dest, carrier, \
                          nbflights_fordayhour_forairport, nbflights_forday_forairport]],\
                        columns=['CRS_ARR_TIME', 'CRS_DEP_TIME', 'CRS_ELAPSED_TIME', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DISTANCE', 'MONTH', 'ORIGIN', 'DEST', 'UNIQUE_CARRIER',\
                                 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'NBFLIGHTS_FORDAY_FORAIRPORT'],)
                                 #dtype=['str', 'str', ])

df_transformed = model['dataprep'].transform(df_input)

st.header('Arrival delay prediction')
df_delay_prediction = model['prediction'].predict(df_transformed)
delay_prediction = df_delay_prediction[0]

st.write(f'{delay_prediction:.2f} minutes')

debug_mode = st.checkbox('Display debug information', value=False)

if (debug_mode == True):
    st.header('Input data')
    st.table(df_input)
    #st.write(crs_arr_time)
    
    print('crs arr time : ' + str(crs_arr_time))
    
    st.header('Transformed data passed to the model')
    if (crs_arr_time != '' and crs_dep_time != '' and crs_elapsed_time != 0):
        st.table(df_transformed)
        

del model

import gc
gc.collect()