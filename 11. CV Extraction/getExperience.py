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
    """Function to extract period info from resumes/cvs"""
    if text == "Fail":
        return "Fail"
    date_ls = []
    clean_txt = clean_special_text(text)
    for txt in text.split("\n"):
        try:
            date_dict = {'period': '', 'start_idx': '', 'end_idx': ''}
            search_txt = clean_special_text(txt)
            search_out = keyword_processor.extract_from_text(search_txt)
            if len(search_out) > 0:
                match = (re.search(search_txt, clean_txt))
                date_dict['period'] = search_txt
                date_dict['start_idx'] = match.span()[0]
                date_dict['end_idx'] = match.span()[1]
                date_ls.append(date_dict)
        except Exception as e:
            continue
    
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
            try:
                ls = []
                for ls_2 in title_output:
                    _dict = {'company_name':ls_1['keyword'], 'title':ls_2['keyword'], 'title_distance':'', 
                             'company_start_idx':ls_1['start_idx'], 'company_end_idx':ls_1['end_idx'],
                             'title_start_idx':ls_2['start_idx'], 'title_end_idx':ls_2['end_idx']}
                    company_end_idx = ls_1['end_idx']
                    title_start_idx = ls_2['start_idx']
                    _dict['title_distance'] = np.abs(company_end_idx-title_start_idx)
                    if (company_end_idx > title_start_idx):
                        _dict['title_words'] = len(clean_txt[title_start_idx:company_end_idx].strip().split())
                    else:
                        _dict['title_words'] = len(clean_txt[company_end_idx:title_start_idx].strip().split())

                    if _dict['title_words'] > 50: ## If # of words between company & title exceed 50 then don't assign [Assumption]
                        continue
                    ls.append(_dict)
                out_ls.append(min(ls, key=lambda x:x['title_distance']))
            except Exception as e:
                continue
        out_ls = sorted(out_ls, key = lambda i: (i['company_name'], i['title_words']), reverse=False)
        out_ls = list({i['company_name']:i for i in reversed(out_ls)}.values()) ## Remove dups
    else:
        return []
    
    full_ls = []

    if (len(out_ls) > 0) and (len(date_output) > 0):
        for ls_1 in out_ls:
            try:
                ls = []
                for ls_2 in date_output:
                    _dict = {'company_name':ls_1['company_name'], 'title':ls_1['title'], 'period':ls_2['period'], 'title_distance':ls_1['title_distance'], 
                             'date_distance':'', 'title_words':ls_1['title_words']}
                    company_end_idx = ls_1['company_end_idx']
                    date_start_idx = ls_2['start_idx']
                    _dict['date_distance'] = np.abs(company_end_idx-date_start_idx)

                    if (company_end_idx > date_start_idx):
                        _dict['date_words'] = len(clean_txt[date_start_idx:company_end_idx].strip().split())
                    else:
                        _dict['date_words'] = len(clean_txt[company_end_idx:date_start_idx].strip().split())

                    if _dict['date_words'] > 50: ## If # of words between company & date exceed 50 then don't assign [Assumption]
                        _dict['period'] = ''

                    ls.append(_dict)
                full_ls.append(min(ls, key=lambda x:x['date_distance']))
            except Exception as e:
                continue
    
        other_companies = list(set([i['keyword'] for i in company_output]) - set([i['company_name'] for i in full_ls]))

        other_ls = []
        for comp in other_companies:
            _dict = {'company_name': '', 
                     'title': '',
                     'period': '',
                     'title_distance': '',
                     'date_distance': '',
                     'title_words': '',
                     'date_words': ''}
            _dict['company_name'] = comp
            other_ls.append(_dict)

        final_ls = full_ls + other_ls
        
        return final_ls
    else:
        return out_ls