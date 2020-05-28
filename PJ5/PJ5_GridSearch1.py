#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:23:51 2020

@author: francois

PJ5 Openclassrooms : this script does Grid search validation and saves results
Results are saved in pickle file and loadable in notebook PJ5_modelisation

Usage :
    Launch a python3 console
    Run PJ5_GridSearch_prerequisites.py
    Run this script

1 candidate (2 validations) launched in 2.5 min
"""

import sys, importlib

# The first time PJ5_GridSearch1.py is called by the interpretor : first "from...import" needs to be called
# But the second time, "from...import" must be called after reload
from functions import *
importlib.reload(sys.modules['functions'])
from functions import *

# Pipeline and GridSearch
df_train = df_train_ori
df_test = df_test_ori

complete_pipeline = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    ('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 
                                                              'RfmScore', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
    #('scaler', LogScalerMultiple(features_toscale=['RfmScore'])),
    
    
    # Faire la réduction dimensionnelle à part pour les bag of words et pour les autres features
   
    ('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    ('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
                                                            algorithm_to_use='NCA', n_dim=3, labels_featurename=None)),
    ('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    ('clusterer', Clusterer(n_clusters=11, algorithm_to_use='WARD'))
])


param_grid_bow_always_NCA_ideal = [
    #{'agregate_to_client_level__top_value_products' : [
    #    
    #],
    {'features_selector__features_toselect': [
        ['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 
        'RfmScore']  ,

        ['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 
        'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']  ,
        
        ['DescriptionNormalized', 
        'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']  ,

        ['DescriptionNormalized', 
        'RfmScore']  ,

        ['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled']  ,

        ['DescriptionNormalized', 'HasEverCancelled']  ,

        ['DescriptionNormalized', 'BoughtTopValueProduct']  ,
                   
        ['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 
        'TotalPricePerMonth', 'TotalQuantityPerMonth']  ,

    ],

     'minmaxscaler__features_toscale' : [  
         'ALL_FEATURES', 
     ],
     
     'dimensionality_reductor__features_totransform' : [  
         ['DescriptionNormalized'], 
         'ALL',
     ],
     
     'dimensionality_reductor__algorithm_to_use' : [  
         'NCA', 
     ],

     'dimensionality_reductor__n_dim' : [  
         3, 10, 50, 100, 150, 200, 300
     ],
     
     'dimensionality_reductor__labels_featurename': [
         'RfmScore',
          bow_labels_train,
     ],

    'minmaxscaler_final__features_toscale' : [  
             'ALL_FEATURES', 
         ],
    
     'clusterer__algorithm_to_use' : [  
         'WARD', 
         'KMEANS' 
     ],
  
     'clusterer__n_clusters' : [  
         3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 30, 40, 50
     ],

    },
    

    {'features_selector__features_toselect': [
        ['BoughtTopValueProduct', 'HasEverCancelled', 
        'RfmScore']  ,

        ['BoughtTopValueProduct', 'HasEverCancelled', 
        'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']  ,
        
        ['TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']  ,

        ['RfmScore']  ,

        ['BoughtTopValueProduct', 'HasEverCancelled']  ,

        ['HasEverCancelled']  ,

        ['BoughtTopValueProduct']  ,
                   
        ['BoughtTopValueProduct', 'HasEverCancelled', 
        'TotalPricePerMonth', 'TotalQuantityPerMonth']  ,

    ],

     'minmaxscaler__features_toscale' : [  
         'ALL_FEATURES', 
     ],
     
     'dimensionality_reductor__features_totransform' : [  
         None,
     ],
     
     'dimensionality_reductor__algorithm_to_use' : [  
         'NCA', 
     ],

     'dimensionality_reductor__n_dim' : [  
         3, 10, 50, 100, 150, 200, 300
     ],
     
     'dimensionality_reductor__labels_featurename': [
         'RfmScore',
          bow_labels_train,
     ],

    'minmaxscaler_final__features_toscale' : [  
             'ALL_FEATURES', 
         ],

     'clusterer__algorithm_to_use' : [  
         'WARD', 
         'KMEANS' 
     ],
  
     'clusterer__n_clusters' : [  
         3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 30, 40, 50
     ],

    },
]

param_grid = [
    #{'agregate_to_client_level__top_value_products' : [
    #    
    #],
    {'features_selector__features_toselect': [
        ['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 
        'RfmScore']  ,

        ['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 
        'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']  ,
        
        ['DescriptionNormalized', 
        'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']  ,

        ['DescriptionNormalized', 
        'RfmScore']  ,

        ['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled']  ,

        ['DescriptionNormalized', 'HasEverCancelled']  ,

        ['DescriptionNormalized', 'BoughtTopValueProduct']  ,
                   
        ['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 
        'TotalPricePerMonth', 'TotalQuantityPerMonth']  ,

    ],

     'minmaxscaler__features_toscale' : [  
         'ALL_FEATURES', 
     ],
     
     'dimensionality_reductor__features_totransform' : [  
         ['DescriptionNormalized'], 
         'ALL',
     ],
     
     'dimensionality_reductor__algorithm_to_use' : [  
         'NCA', 
     ],

     'dimensionality_reductor__n_dim' : [  
         3, 10, 200
     ],
     
     'dimensionality_reductor__labels_featurename': [
         'RfmScore',
          bow_labels_train,
     ],

    'minmaxscaler_final__features_toscale' : [  
             'ALL_FEATURES', 
         ],
    
     'clusterer__algorithm_to_use' : [  
         'WARD', 
         'KMEANS' 
     ],
  
     'clusterer__n_clusters' : [  
         4, 8, 20, 50
     ],

    },
    

    {'features_selector__features_toselect': [
        ['BoughtTopValueProduct', 'HasEverCancelled', 
        'RfmScore']  ,

        ['BoughtTopValueProduct', 'HasEverCancelled', 
        'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']  ,
        
        ['TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency']  ,

        ['RfmScore']  ,

        ['BoughtTopValueProduct', 'HasEverCancelled']  ,

        ['HasEverCancelled']  ,

        ['BoughtTopValueProduct']  ,
                   
        ['BoughtTopValueProduct', 'HasEverCancelled', 
        'TotalPricePerMonth', 'TotalQuantityPerMonth']  ,

    ],

     'minmaxscaler__features_toscale' : [  
         'ALL_FEATURES', 
     ],
     
     'dimensionality_reductor__features_totransform' : [  
         None,
     ],
     
     'dimensionality_reductor__algorithm_to_use' : [  
         'NCA', 
     ],

     'dimensionality_reductor__n_dim' : [  
         3, 10, 200
     ],
     
     'dimensionality_reductor__labels_featurename': [
         'RfmScore',
          bow_labels_train,
     ],

    'minmaxscaler_final__features_toscale' : [  
             'ALL_FEATURES', 
         ],

     'clusterer__algorithm_to_use' : [  
         'WARD', 
         'KMEANS' 
     ],
  
     'clusterer__n_clusters' : [  
         4,8,20,50
     ],

    },
]
    
'''    
# For debug
param_grid = [
    #{'agregate_to_client_level__top_value_products' : [
    #    
    #],
    {'features_selector__features_toselect': [
        ['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 
        'RfmScore']  ,
    ],

     'minmaxscaler__features_toscale' : [  
         'ALL_FEATURES', 
     ],
     
     'dimensionality_reductor__features_totransform' : [  
         ['DescriptionNormalized'], 
     ],
                                                                             
     'dimensionality_reductor__algorithm_to_use' : [  
         'NCA', 
     ],

     'dimensionality_reductor__n_dim' : [  
         5
     ],
     
     'dimensionality_reductor__labels_featurename': [
         'RfmScore',
     ],

    'minmaxscaler_final__features_toscale' : [  
             'ALL_FEATURES', 
         ],
    
     'clusterer__algorithm_to_use' : [  
         'WARD', 
     ],
  
     'clusterer__n_clusters' : [  
         3
     ],

    },
]
'''


param_grid_debug = [
    #{'agregate_to_client_level__top_value_products' : [
    #    
    #],
    {'features_selector__features_toselect': [
        ['DescriptionNormalized', 'HasEverCancelled']  ,
    ],

     'minmaxscaler__features_toscale' : [  
         'ALL_FEATURES', 
     ],
     
     'dimensionality_reductor__features_totransform' : [  
         ['DescriptionNormalized'], 
     ],
                                                                             
     'dimensionality_reductor__algorithm_to_use' : [  
         'NCA', 
     ],

     'dimensionality_reductor__n_dim' : [  
         3,
     ],
     
     #'dimensionality_reductor__labels_featurename': [
     #    rfm_score_train,
     #],

    'minmaxscaler_final__features_toscale' : [  
             'ALL_FEATURES', 
         ],
    
     'clusterer__algorithm_to_use' : [  
         'WARD', 
     ],
  
     'clusterer__n_clusters' : [  
         4,
     ],

    },
]


complete_pipeline_debug2 = Pipeline([
    #('features_selector', FeaturesSelector(features_toselect=MODEL_FEATURES)),
    ('bow_encoder', BowEncoder()),
    ('agregate_to_client_level', AgregateToClientLevel(top_value_products, compute_rfm=True)),
    #('features_selector', FeaturesSelector(features_toselect=['DescriptionNormalized', 'BoughtTopValueProduct', 'HasEverCancelled', 
    #                                                          'RfmScore', 'TotalPricePerMonth', 'TotalQuantityPerMonth', 'Recency'])),
   
    #('minmaxscaler', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    #('dimensionality_reductor', DimensionalityReductor(features_totransform=['DescriptionNormalized'], \
    #                                                    algorithm_to_use='NCA', n_dim=3, labels_featurename='RfmScore')),
    #('minmaxscaler_final', MinMaxScalerMultiple(features_toscale='ALL_FEATURES')),
    #('clusterer', Clusterer(n_clusters=11, algorithm_to_use='WARD'))
])


'''
Fit Dimensionality Reductor
/home/francois/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_validation.py:547: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
ValueError: Input array must be 1 dimensional

'''


print('\nGrid Search launch')
if (RECOMPUTE_GRIDSEARCH == True):
    model = complete_pipeline
        
    # error_score = np.nan means an error on 1 combination does is not blocking every other models.  scoring=None means our predictor's score method will be used
    # For later : try custom cv splitter
    
    # Debug purposes
    '''
    complete_pipeline_debug2.fit(df_train)
    df_train_transformed = complete_pipeline_debug2.transform(df_train)
    df_test_transformed = complete_pipeline_debug2.transform(df_test)
    '''
    
    
    grid_search = GridSearchCV(model, param_grid_debug, cv=2, verbose=10, error_score=np.nan, scoring=None) # TO CHANGE back to cv=5
    grid_search.fit(df_train, dimensionality_reductor__labels=rfm_score_train)

    grid_search_res, df_grid_search_results_res = save_or_load_search_params(grid_search, 'gridsearch_BoWAlways_NCAAlways_debug')
    