import pickle

#import pandas as pd

API_MODEL_PICKLE_FILE = 'API_model_PJ3.pickle'

def getRecommendations(id_film):
    model = None
    
    with open(API_MODEL_PICKLE_FILE, 'rb') as f:
        model = pickle.load(f)
        
    JSON_output = {"_results": [] }

    print(f'id_film = {id_film}')
    
    for nb_film in range(5):
        JSON_output["_results"].append({"id": model['reco_matrix'][id_film, nb_film+1].item(), "name" : model['movie_names'][model['reco_matrix'][id_film, nb_film+1].item()] })
        
    return(JSON_output)
    
    
if __name__ == '__main__':
    res = getRecommendations(2703)