
from flask import Flask, jsonify
from flask import request

import read_corpus

import nltk

import string
import operator

import recommender_API


from nltk.corpus import stopwords
from collections import defaultdict

def tokenize(text):
    swords = stopwords.words('french')
    for w in ["...","'s","n't","’", "''","``"]:
        swords.append(w)

    stem = nltk.stem.SnowballStemmer('french')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        if token in swords: continue
        #yield stem.stem(token)
        yield(token)



def vectorize(doc):
    features = defaultdict(int)
    for token in tokenize(doc):
        features[token] += 1
    return features


app = Flask(__name__)

################## Used for old Texte application (a sort of hello word ;)), do not remove for the moment #############"

#@app.route('/', methods=['GET', 'POST'])
@app.route('/api', methods=['POST'])
def get_output():
    myinput = request.json['input']
    output = myinput
    return jsonify({'output': output})

@app.route('/analyzetext2', methods=['POST'])
def get_output3():
    myinput = request.json['input']

    features = defaultdict(int)

    text=myinput

    if text == "": return jsonify({'output': text})

    for token in tokenize(text):
        features[token] += 1

    sorted_feats = sorted(features.items(), key=operator.itemgetter(1), reverse=True)

    output=""
    for val in sorted_feats:
        output = output + str(val[0]) + ':' + str(val[1]) + '\n'

    print("output = {}".format(output), flush=True)

    #output = 'Texte du retour API.'
    return jsonify({'output': output})


@app.route('/analyzetext', methods=['POST'])
def get_output2():
    myinput = request.json['input']


    #output = 'Texte du retour API.'
    #output = ['1', '2', '3']
    output = { 'item1':'value1', 'item2':'value2'}

    return jsonify({'output': output})
##############################################################################################
    
    
@app.route('/prototube_api', methods=['POST'])
def get_data():
    action = request.json['action']
    params = request.json['params']
    emotionfilter = request.json['emotionfilter']
    searchperiod = request.json['searchperiod']
    corpuscategory = request.json['corpuscategory']
    
    #print("searchperiod = {}".format(searchperiod), flush=True)
    
    result=read_corpus.getVideosOfTheDay(emotionfilter, searchperiod, corpuscategory)
    
    return jsonify({'output': result})

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    id_film = request.json['id_film']
    
    result = recommender_API.getRecommendations(id_film)
    
    print('result type : ' + str(type(result)))
    
    return jsonify(result)


