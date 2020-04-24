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
    with open(pickled_file, 'rb') as f:
        model = pickle.load(f)
    
    return(model)


_max_width_()

st.title('Openclassrooms training projet 4 : predicting flight delays (François BOYER)') 

image = Image.open('plane_image.jpg')
st.image(image,
         width=300)
         #use_column_width=True)

st.sidebar.title('Flight characteristics')
crs_dep_time = st.sidebar.text_input('Scheduled Departure time (HHMM)', '1000')
crs_arr_time = st.sidebar.text_input('Scheduled Arrival time (HHMM)', '1800')


crs_elapsed_time = st.sidebar.number_input('Scheduled Elapsed time (minutes)', value=60, step=10)
day_of_month = st.sidebar.selectbox('Day of Month (1-31)', range(1,32))
day_of_week = st.sidebar.selectbox('Day of week (1-7)', range(1,8))
distance = st.sidebar.number_input('Distance (miles)', value=1000, step=100)
month = st.sidebar.selectbox('Month (1-12)', range(1,13))

origin_list = ['BOS', 'JFK', 'LAX', 'DFW', 'OKC', 'OGG', 'HNL', 'SFO', 'ORD',
       'MIA', 'IAH', 'DTW', 'SEA', 'MSP', 'LGA', 'ATL', 'LAS', 'CLT',
       'DCA', 'SAN', 'COS', 'PDX', 'TUS', 'SJC', 'DEN', 'PHX', 'SNA',
       'MCO', 'AUS', 'STL', 'KOA', 'MEM', 'SLC', 'PHL', 'LIH', 'MCI',
       'JAX', 'MSY', 'IAD', 'SJU', 'ORF', 'ABQ', 'FLL', 'IND', 'SAT',
       'EWR', 'BWI', 'RDU', 'TPA', 'ONT', 'TUL', 'BNA', 'SMF', 'DSM',
       'RNO', 'DAY', 'BDL', 'FAT', 'OMA', 'MKE', 'SDF', 'PIT', 'RSW',
       'CMH', 'STT', 'STX', 'PBI', 'ELP', 'PSP', 'ICT', 'AMA', 'PNS',
       'CLE', 'XNA', 'MFE', 'RIC', 'HOU', 'OAK', 'JAC', 'EGE', 'PVD',
       'BUF', 'ILM', 'SYR', 'MDT', 'CHS', 'ALB', 'PWM', 'GSO', 'ROC',
       'BOI', 'GEG', 'LBB', 'ANC', 'ADQ', 'BET', 'BRW', 'SCC', 'FAI',
       'SIT', 'JNU', 'KTN', 'CDV', 'YAK', 'WRG', 'PSG', 'OME', 'OTZ',
       'ADK', 'BUR', 'LGB', 'BTV', 'HPN', 'SRQ', 'SWF', 'DAB', 'SAV',
       'ORH', 'ACK', 'MVY', 'BQN', 'PSE', 'HYA', 'TLH', 'BHM', 'ATW',
       'SHV', 'GRB', 'HSV', 'GSP', 'AVL', 'BZN', 'GRR', 'MDW', 'ROA',
       'MSN', 'TYS', 'CVG', 'CHA', 'FAR', 'FSD', 'JAN', 'BIS', 'FNT',
       'AVP', 'GPT', 'BIL', 'CAE', 'LEX', 'LIT', 'TVC', 'CAK', 'ECP',
       'MYR', 'MSO', 'SGF', 'MLB', 'CHO', 'MHT', 'RAP', 'EYW', 'GNV',
       'DAL', 'PSC', 'PHF', 'OAJ', 'FCA', 'BMI', 'MOB', 'VPS', 'EVV',
       'AGS', 'BTR', 'CRW', 'TRI', 'BGR', 'FAY', 'ABE', 'LFT', 'UST',
       'CID', 'TTN', 'ITO', 'VLD', 'PPG', 'ACY', 'PBG', 'IAG', 'LBE',
       'FLG', 'GJT', 'RDM', 'DRO', 'BFL', 'YUM', 'SBA', 'ROW', 'SBP',
       'MRY', 'ASE', 'EUG', 'SUN', 'HLN', 'SBN', 'IDA', 'DLH', 'LNK',
       'PIA', 'AZO', 'LSE', 'FWA', 'GTF', 'MLI', 'MFR', 'GFK', 'ISN',
       'SPI', 'DVL', 'JMS', 'CMX', 'OTH', 'MBS', 'RKS', 'HYS', 'LAR',
       'PAH', 'SGU', 'GCC', 'MOT', 'MAF', 'HDN', 'CWA', 'ACV', 'MTJ',
       'CPR', 'SCE', 'GUC', 'RDD', 'SMX', 'HRL', 'BRO', 'RHI', 'CDC',
       'BRD', 'ABR', 'COD', 'PLN', 'EKO', 'HIB', 'INL', 'PIH', 'BTM',
       'LWS', 'ITH', 'MQT', 'BJI', 'IMT', 'CIU', 'APN', 'WYS', 'TWF',
       'ESC', 'BGM', 'LAN', 'RST', 'EAU', 'MKG', 'CRP', 'CLL', 'LCH',
       'ERI', 'AEX', 'BQK', 'DHN', 'GUM', 'ISP', 'MGM', 'TXK', 'GCK',
       'LRD', 'ACT', 'LAW', 'GRI', 'GGG', 'MLU', 'SPS', 'SJT', 'FSM',
       'SAF', 'MEI', 'PIB', 'GRK', 'HOB', 'JLN', 'ABI', 'EWN', 'CSG',
       'ELM', 'ABY', 'GTR', 'GST', 'AKN', 'DLG', 'BPT', 'BLI', 'TYR',
       'PGD', 'MMH']

origin_list.sort()

dest_list = ['JFK', 'LAX', 'HNL', 'DFW', 'OGG', 'SFO', 'LAS', 'BOS', 'MIA',
       'MCO', 'ORD', 'DTW', 'SEA', 'LGA', 'SJC', 'CLT', 'DCA', 'PDX',
       'SLC', 'SAN', 'STL', 'PHX', 'EWR', 'AUS', 'SAT', 'BWI', 'MSP',
       'TPA', 'IAH', 'KOA', 'MEM', 'DEN', 'LIH', 'MCI', 'MSY', 'SNA',
       'RDU', 'ONT', 'ABQ', 'ORF', 'TUL', 'IND', 'SJU', 'BNA', 'PHL',
       'BDL', 'ELP', 'ATL', 'DAY', 'SMF', 'OMA', 'STX', 'SDF', 'PIT',
       'RSW', 'RIC', 'DSM', 'JAX', 'STT', 'TUS', 'IAD', 'FLL', 'FAT',
       'OKC', 'CMH', 'RNO', 'MKE', 'CLE', 'COS', 'AMA', 'PNS', 'PBI',
       'ICT', 'PSP', 'XNA', 'MFE', 'LBB', 'HOU', 'BOI', 'JAC', 'EGE',
       'CHS', 'ILM', 'MDT', 'GSO', 'PVD', 'PWM', 'SYR', 'ALB', 'ROC',
       'BUF', 'OAK', 'GEG', 'ANC', 'BET', 'ADQ', 'BRW', 'SCC', 'FAI',
       'JNU', 'KTN', 'YAK', 'CDV', 'SIT', 'PSG', 'WRG', 'OME', 'OTZ',
       'ADK', 'BUR', 'LGB', 'BTV', 'DAB', 'SRQ', 'SWF', 'SAV', 'HPN',
       'ORH', 'ACK', 'MVY', 'BQN', 'PSE', 'HYA', 'CVG', 'TLH', 'SHV',
       'GRB', 'HSV', 'TYS', 'AVL', 'BZN', 'GRR', 'MDW', 'ROA', 'MSN',
       'CHA', 'JAN', 'GSP', 'ATW', 'LEX', 'GPT', 'AGS', 'CRW', 'BHM',
       'BIL', 'LIT', 'CAE', 'CAK', 'BIS', 'MYR', 'ECP', 'FAY', 'TVC',
       'MLB', 'FNT', 'MSO', 'BGR', 'EYW', 'DAL', 'FAR', 'FCA', 'BMI',
       'CHO', 'RAP', 'PSC', 'VPS', 'TRI', 'EVV', 'LFT', 'SGF', 'ABE',
       'AVP', 'GNV', 'MOB', 'PHF', 'FSD', 'BTR', 'MHT', 'OAJ', 'UST',
       'CID', 'TTN', 'ITO', 'PPG', 'ACY', 'PBG', 'IAG', 'LBE', 'SBP',
       'FLG', 'GJT', 'DRO', 'BFL', 'RDM', 'YUM', 'SBA', 'ROW', 'MRY',
       'ASE', 'EUG', 'SUN', 'HLN', 'SBN', 'FWA', 'IDA', 'AZO', 'DLH',
       'LSE', 'GTF', 'PIA', 'MLI', 'LNK', 'GFK', 'RST', 'ISN', 'MFR',
       'JMS', 'DVL', 'LAR', 'OTH', 'RKS', 'GCC', 'MBS', 'HYS', 'CMX',
       'EAU', 'MKG', 'SPI', 'CPR', 'MOT', 'MAF', 'CWA', 'SCE', 'MTJ',
       'SMX', 'RDD', 'HDN', 'ACV', 'BRO', 'HRL', 'LWS', 'BJI', 'RHI',
       'BTM', 'ABR', 'CDC', 'BRD', 'COD', 'PLN', 'EKO', 'HIB', 'APN',
       'CIU', 'INL', 'PIH', 'SGU', 'IMT', 'ITH', 'WYS', 'TWF', 'MQT',
       'ESC', 'BGM', 'LAN', 'PAH', 'GUC', 'CRP', 'GUM', 'ISP', 'AEX',
       'MGM', 'LAW', 'SJT', 'GCK', 'GRI', 'GGG', 'TXK', 'CLL', 'ACT',
       'LRD', 'MLU', 'FSM', 'SAF', 'LCH', 'MEI', 'PIB', 'SPS', 'GRK',
       'ERI', 'HOB', 'ABI', 'JLN', 'CSG', 'DHN', 'GTR', 'ELM', 'VLD',
       'BQK', 'EWN', 'ABY', 'GST', 'DLG', 'AKN', 'BPT', 'BLI', 'TYR',
       'PGD', 'MMH']

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


df_input = pd.DataFrame([[crs_arr_time, crs_dep_time, crs_elapsed_time, day_of_month, day_of_week, distance, month, origin, dest, carrier, \
                          nbflights_fordayhour_forairport, nbflights_forday_forairport]],\
                        columns=['CRS_ARR_TIME', 'CRS_DEP_TIME', 'CRS_ELAPSED_TIME', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DISTANCE', 'MONTH', 'ORIGIN', 'DEST', 'UNIQUE_CARRIER',\
                                 'NBFLIGHTS_FORDAYHOUR_FORAIRPORT', 'NBFLIGHTS_FORDAY_FORAIRPORT'])

df_transformed = model['dataprep'].transform(df_input)

st.header('Arrival delay prediction')
df_delay_prediction = model['prediction'].predict(df_transformed)
delay_prediction = df_delay_prediction[0]

st.write(f'{delay_prediction:.2f} minutes')

debug_mode = st.checkbox('Display debug information', value=False)

if (debug_mode == True):
    st.header('Input data')
    st.table(df_input)
    
    print('crs arr time : ' + str(crs_arr_time))
    
    st.header('Transformed data passed to the model')
    if (crs_arr_time != '' and crs_dep_time != '' and crs_elapsed_time != 0):
        st.table(df_transformed)
        

