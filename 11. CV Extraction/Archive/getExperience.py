# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:50:36 2021

@author: Jitendra
"""
## Loading Required Modules

# General Utilities
import numpy as np
import re
import string
import spacy
nlp = spacy.load('en_core_web_sm')

## Text Preprocessing Functions
def clean_ner_text(text):
    """Function to clean NER text"""
    text = str(text)
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    text = text.replace("\n", " ")
    text = re.sub(' +', ' ', text)
    text = " ".join(text.split())
    return text

def clean_special_text(text):
    """Function to clean general text"""
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    text = text.replace("\n", " ")
    translator = re.compile('[%s]' % re.escape(string.punctuation))
    text = translator.sub(' ', text)
    text = re.sub(r'(?<=\b[a-z]) (?=[a-z]\b)', '', text)
    text = re.sub(' +', ' ', text)
    text = " ".join(text.split())
    text = text.strip()
    return text

## Company Names Extraction
def getCompanyNames(text, keyword_processor):
    """Function to extract company names from resumes/cvs"""
    if text == "Fail":
        return "Fail"
    text = clean_special_text(text)
    company_ls = keyword_processor.extract_from_text(text)
    return company_ls

## Title/Designation Extraction
def getJobTitles(text, keyword_processor):
    """Function to extract titles/designations from resumes/cvs"""
    if text == "Fail":
        return "Fail"
    text = clean_special_text(text)
    title_ls = keyword_processor.extract_from_text(text)
    return title_ls

## Time Period Extraction
def getPeriod(text, keyword_processor):
    """Function to extract experience time period from resumes/cvs"""
    if text == "Fail":
        return "Fail"
    text = clean_ner_text(text)
    doc = nlp(text)
    date_ls = []
    if doc.ents: 
        for ent in doc.ents: 
            date_dict = {'period':'', 'start_char':'', 'end_char':''}
            if ent.label_ == "DATE":
                date_dict['period'] = ent.text
                date_dict['start_char'] = ent.start_char
                date_dict['end_char'] = ent.end_char
                if len(keyword_processor.extract_from_text(ent.text)) > 0:
                    date_ls.append(date_dict)
    return date_ls

## Full Experience Extraction
def getExperience(text, company_output, title_output, date_output):
    """
    Function to Link Company, Title & Period based on distances
    
    a) Input: Clean Text, Extracted Companies, Title & Dates
    
    b) Output: Mapped Company, Title & Dates
    """
    if text == "Fail":
        return "Fail"
    
    clean_txt = clean_special_text(text)
    
    out_ls = []
    if (len(company_output) > 0) and (len(title_output) > 0):
        for ls_1 in company_output:
            ls = []
            for ls_2 in title_output:
                _dict = {'company_name':ls_1['keyword'], 'title':ls_2['keyword'], 'distance':'', 
                         'company_start_idx':ls_1['start_idx'], 'company_end_idx':ls_1['end_idx'],
                         'title_start_idx':ls_2['start_idx'], 'title_end_idx':ls_2['end_idx']}
                company_end_idx = ls_1['end_idx']
                title_start_idx = ls_2['start_idx']
                _dict['distance'] = np.abs(company_end_idx-title_start_idx)
                if (company_end_idx > title_start_idx):
                    _dict['words'] = len(clean_txt[title_start_idx:company_end_idx].strip().split())
                else:
                    _dict['words'] = len(clean_txt[company_end_idx:title_start_idx].strip().split())

                if _dict['words'] > 50: ## If # of words between company & title exceed 50 then don't assign [Assumption]
                    _dict['title'] = ''
                ls.append(_dict)
            out_ls.append(min(ls, key=lambda x:x['distance']))

    full_ls = []

    if (len(out_ls) > 0) and (len(date_output) > 0):
        for ls_1 in out_ls:
            ls = []
            for ls_2 in date_output:
                _dict = {'company_name':ls_1['company_name'], 'title':ls_1['title'], 'period':ls_2['period'], 'distance_1':ls_1['distance'], 
                         'distance_2':'', 'words':ls_1['words']}
                company_end_idx = ls_1['company_end_idx']
                date_start_idx = ls_2['start_char']
                _dict['distance_2'] = np.abs(company_end_idx-date_start_idx)

                if (company_end_idx > date_start_idx):
                    _dict['date_words'] = len(clean_txt[date_start_idx:company_end_idx].strip().split())
                else:
                    _dict['date_words'] = len(clean_txt[company_end_idx:date_start_idx].strip().split())

                if _dict['date_words'] > 50: ## If # of words between company & date exceed 50 then don't assign [Assumption]
                    _dict['period'] = ''

                ls.append(_dict)
            full_ls.append(min(ls, key=lambda x:x['distance_2']))
    
    full_ls = sorted(full_ls, key = lambda i: (i['title'], i['words']), reverse=True)
    full_ls = list({i['company_name']:i for i in reversed(full_ls)}.values()) ## Remove dups
    return full_ls
