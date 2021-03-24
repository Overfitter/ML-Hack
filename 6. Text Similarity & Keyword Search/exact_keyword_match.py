# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:57:54 2021

@author: Jitendra Jangid
"""

# General Utilities
import sys, os
import re
import pandas as pd
import numpy as np
np.random.seed(42)
import string
import unicodedata
import spacy
import nltk
nltk.download('stopwords');
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from flashtext import KeywordProcessor

stopwords = set(set(stopwords.words('english')) -set(["or", "no", "not"]))

# class
class text_preprocessing:
    def __init__(self):
        self.nlp = spacy.load('en_core_sci_lg')
        
    def clean_sentence(self, text):
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
        text = text.lower()
        text = unicodedata.normalize("NFKD", text)
        text = self.clean_text(text)
        text = self.remove_punct(text)
        text = self.lemma_text(text)
        text = self.remove_stopwords(text, is_lower_case=True)
        text = " ".join(text.split())
        text = text.replace("'", "")
        text = text.strip()
        return(text) 
    
    def clean_text(self, text):
        """Removing Punctuations from the text"""
        text = str(text)
        for punct in "/-'":
            text = text.replace(punct, ' ')
        for punct in '&':
            text = text.replace(punct, ' ')
        for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
            text = text.replace(punct, ' ')
        return(text)
    
    def clean_numbers(self, text):
        """Removing Numbers (0-9) from text"""
        if bool(re.search(r'\d', text)):
            text = re.sub('[0-9]', ' ', text)
        return(text)

    def remove_punct(self, text):
        clean_text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        while '  ' in clean_text: clean_text = clean_text.replace('  ', ' ')
        return(clean_text)
    
    def lemma_text(self, text):
        """Converting the word into the root word (Lemmatization) using Scispacy Module"""
        s = [token.lemma_ for token in self.nlp(text) if token.lemma_ != '-PRON-']
        output = ' '.join(s)
        return(output)
    
    def remove_stopwords(self, text, is_lower_case=True):
        """Removing stopwords after tokenizing the text using ToktokTokenizer"""
        stopword_list = list(set(list(stopwords) + ['may','also','across','among','beside','however','yet','within']))
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = " ".join(filtered_tokens)
        return(filtered_text)
    
    
class keyword_extract:
    def __init__(self, keyword_dict):
        self.text_clean_device = text_preprocessing()
        self.original_keyword_dict = keyword_dict
        self.cleaned_keyword_dict = [self.text_clean_device.clean_sentence(keyword) for keyword in self.original_keyword_dict]
        self._keyword_map_dic = dict(zip(self.cleaned_keyword_dict, self.original_keyword_dict))
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        self.keyword_processor.add_keywords_from_dict(self.cleaned_keyword_dict)
        
    def _exact_match(self, text):
        reg_joinedpunct = re.compile(r'\d+/[a-zA-Z]+', re.S)
        sentence = text
        sentence = sentence.replace("â\x89¤", "less than equal to")
        sentence = sentence.replace("â\x89¥", "greater than equal to")
        sentence = re.sub("\n|\r", "", re.sub(" +", " ", sentence.strip()))
        sentence = re.sub('([.,;:!?{}()])', r' \1 ', sentence)
        sentence = re.sub('\s{2,}', ' ', sentence)
        sentence = reg_joinedpunct.sub(lambda m: m.group().replace("/", " ", 1), sentence)
        sentence = re.sub(r"(\d+) , *?(\d+)", r"\1,\2", sentence)
        sentence = re.sub(r"(\d+) \. *?(\d+)", r"\1.\2", sentence)
        sentence = re.sub("-", " ", sentence)
        keywords_found_list = sorted(list(set([keyword[0] for keyword in self.keyword_processor.extract_keywords(sentence, span_info=True)])))
        return(keywords_found_list)
           
    def extract_from(self, raw_text):
        clean_text = self.text_clean_device.clean_sentence(raw_text)
        keywords_found_list = self._exact_match(clean_text)
        original_keywords_map_list = [self._keyword_map_dic[keyword] for keyword in keywords_found_list]
        return(original_keywords_map_list)
 

class basic_keyword_extract:
    """Basic keyword extractor class using the keyword processor by Flashtext"""
    def __init__(self, keyword_dict, case_sensitive=False):
        self.keyword_dict = keyword_dict
        self.case_sensitive = case_sensitive
        self.keyword_processor = self._initialize_basic_keyword_processor()
  
    def _initialize_basic_keyword_processor(self):
        kp = KeywordProcessor(case_sensitive=self.case_sensitive)
        kp.add_keywords_from_dict(self.keyword_dict)
        return kp
  
    def extract_from_text(self, raw_text):
        if self.case_sensitive: 
            text = raw_text
        else: 
            text = raw_text.lower()
        clean_text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        while '  ' in clean_text: clean_text = clean_text.replace('  ', ' ')
        clean_extracted_keywords_ls = self.keyword_processor.extract_keywords(clean_text, span_info=True)
        return [x for x,y,z in clean_extracted_keywords_ls]

### Load Keyword dict Dataframe:
#keyword_df = pd.read_excel(r"C:\Users\admin\OneDrive - True North Managers LLP\Desktop\TrueNorth\1. Projects\1. 3A Team Project\1. Research & Analysis\2. Master Data - CMI\3. Docs\2. Nature of Business Summary\Company Industry Keywords.xlsx")
#
#keyword_df.columns = keyword_df.loc[2].tolist()
#keyword_df = keyword_df.iloc[3:,1:].fillna("")
#
#keyword_dict = {k:[] for k in keyword_df.columns.tolist()}
#
#for key in keyword_dict.keys():
#    keywords = list(set(filter(None,keyword_df[key].tolist())))
#    keyword_dict[key] = keywords
#
#keyword_search = basic_keyword_extract(keyword_dict, case_sensitive=False)
#keyword_search.extract_from_text("Pharma is a good industry and construction is also good")
# def clean_text_func(text):
#         text = str(text)
#         text = text.lower()
#         text = re.sub(r'[^\x00-\x7F]+',' ', text)
#         text = text.replace("\n", " ")
#         translator = re.compile('[%s]' % re.escape(string.punctuation))
#         text = translator.sub(' ', text)
#         text = re.sub(' +', ' ', text)
#         text = " ".join(text.split())
#         text = text.strip()
#         return text

# class basic_keyword_extract:
#     """Basic keyword extractor class using the keyword processor by Flashtext"""
#     def __init__(self, keyword_list, case_sensitive=False):
#         self.keyword_list = keyword_list
#         self.case_sensitive = case_sensitive
#         self.keyword_processor = self._initialize_basic_keyword_processor()
   
#     def _initialize_basic_keyword_processor(self):
#         kp = KeywordProcessor(case_sensitive=self.case_sensitive)
#         kp.add_keywords_from_list(self.keyword_list)
#         return kp
#     def extract_from_text(self, raw_text):
#         if self.case_sensitive:
#             text = raw_text
#         else:
#             text = raw_text.lower().strip()
#             clean_text = clean_text_func(text)
            
#         clean_extracted_keywords_ls = self.keyword_processor.extract_keywords(clean_text, span_info=True)
# #         output_ls = list(set([i[0] for i in clean_extracted_keywords_ls]))
#         output_ls = [{'keyword':i[0], 'start_idx':i[1], 'end_idx':i[2]} for i in clean_extracted_keywords_ls]
#         output_ls = list({i['keyword']:i for i in reversed(output_ls)}.values()) ## Remove dups
#         if len(output_ls) == 0:
#             return []
#         else:
#             return output_ls