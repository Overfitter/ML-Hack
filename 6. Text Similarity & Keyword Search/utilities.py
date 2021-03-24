import re
import os
from io import StringIO, BytesIO
from collections import OrderedDict
import warnings
import time
warnings.filterwarnings("ignore")

import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np

# Nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.tokenize.toktok import ToktokTokenizer
stop_words = set(stopwords.words('english')) 
from nltk import ngrams

import en_core_web_sm
nlp = en_core_web_sm.load()

from tqdm import tqdm
            

#########################clean_sentence###########################
# Removing Punctuations
def clean_text(x):
    """Removing Punctuations from the text"""
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, ' ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

# Clean numbers
def clean_numbers(x):
    """Removing Numbers (0-9) from text"""
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]', ' ', x)
    return x

# Lemmatize the text
def lemma_text(x):
    """Converting the word into the root word (Lemmatization) using Scispacy Module"""
    s = [token.lemma_ for token in nlp(x) if token.lemma_ != '-PRON-']
    output = ' '.join(s)
    return output

# Remove Stopwords -
def remove_stopwords(text, is_lower_case=True):
    """Removing stopwords after tokenizing the text using ToktokTokenizer"""
    stopword_list = list(set(list(stop_words) + ['may','also','across','among','beside','however','yet','within']))
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text

# Function to clean sentence
def clean_sentence(x):
    
    """
    Purpose: Function to Clean the text
       1. Lowering the text
       2. Removing Punctuations
       3. Removing Numbers
       4. Lemmatization of text
       5. Removing Stopwords
       6. Removing Whitespaces
    Input: Raw string/text
    Output: Clean string/text
    """
    
    x = x.lower()
    x = clean_text(x)
    x = clean_numbers(x)
    x = lemma_text(x)
    x = remove_stopwords(x)
    x = " ".join(x.split())
    x = x.replace("'","")
    x = x.strip()
    return x




         