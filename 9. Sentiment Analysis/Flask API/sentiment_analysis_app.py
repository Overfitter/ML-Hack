# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 21:28:45 2021

@author: Jitendra
"""
# =============================================================================
# Load Required Modules
# =============================================================================

## General Utilities:
import numpy as np
import json
import pickle
import re

## Flask Utilities:
from flask import Flask, request

## NLP Utitlies:
from nltk.stem.porter import PorterStemmer 
stemmer = PorterStemmer()

## ML Utlities:
import xgboost as xgb

# =============================================================================
# Load Constants & Path
# =============================================================================
## Define sentiment labels
LABELS = ['negative', 'neutral', 'positive']

# =============================================================================
# Flask App Initialization
# =============================================================================
app = Flask(__name__)

# =============================================================================
# Load XGBoost Model & TF-IDF Vectorizer
# =============================================================================
tfidf_model = pickle.load(open("models/twitter_sentiment_tfidf.pkl", 'rb'))
xgboost_model = pickle.load(open("models/twitter_sentiment_xgb_model.pkl", 'rb'))

# =============================================================================
# Help Functions
# =============================================================================
def text_preprocessing(input_txt):
    """
    Text Pre-processing
    """
    
    input_txt = str(input_txt)
    ## Remove specific pattern words
    pattern = "@[\w]*"
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    
    input_txt = re.sub(r'http\S+|www\.\S+', '', input_txt)
    
    for punct in '?!.,"$%\'()*+-/:;<=>@[\\]^_`{|}~&' + '“”’':
        input_txt = input_txt.replace(punct, ' ')
    
    input_txt = input_txt.replace("[^a-zA-Z#]", " ") 
    
    if bool(re.search(r'\d', input_txt)):
        input_txt = re.sub('[0-9]', ' ', input_txt)
            
    input_txt = input_txt.replace('"', "")
    input_txt = input_txt.replace("'", "")
    input_txt = input_txt.split()
    
    # Normalise the tokenized words:
    input_txt = " ".join([stemmer.stem(i) for i in input_txt])
    
    return input_txt.lower().strip()

# =============================================================================
# Main Functions
# =============================================================================
@app.route('/get_sentiment', methods=['GET','POST'])
def get_sentiment():
    """
    Function to get twitter sentiments with its confidence score
    """
    
    input_object = [
              {
                "id" : "1",
                "full_text" : "We are pleased to work with this @TCS client building #Cleantech in 🇨🇦. Congratulations to @QDSolar on your recent funding! 🥳 https://t.co/oghGwcQstZ"
              },
              {
                "id" : "2",
                "full_text" : "I got a call from manager and he told me it is compulsory to dispatch the product even @ekartlogistics are not going to deliver. I am not expecting such type of service by @Flipkart ."
              },
              {
                "id" : "3",
                "full_text" : "I have fantastic experience with @delhivery for large/Heavy Items, that's 5* but for others its just shame.. By the way @Flipkart @ekartlogistics (Del exec Mr Manoranjan Sahoo , Ref Id FMPN0001555570 , who has been consistently serving 5* , a big shutout to his service)."
              }
            ]

    output_object = []
    
    try:
        ## Iterate through each input object
        for item in input_object:
            text = item["full_text"]
            text = text_preprocessing(text)
            text_transform = tfidf_model.transform([text])
            scores = xgboost_model.predict(xgb.DMatrix(text_transform), ntree_limit = xgboost_model.best_ntree_limit)[0]
            out_dict = {}
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            for i in range(scores.shape[0]):
                l = LABELS[ranking[i]]
                s = scores[ranking[i]]
                out_dict[l] = s  
            item["sentiment"] = max(out_dict, key=out_dict.get) 
            item["sentiment_score"] =  np.round(out_dict[item["sentiment"]]*100,2)
            del item["full_text"]
            output_object.append(item)
        return json.dumps(output_object)
    except Exception as e:
        error = str(e)
        return app.response_class(response=json.dumps(error), status=500, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=9696, use_reloader=False)
