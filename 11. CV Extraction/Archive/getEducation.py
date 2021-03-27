# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:53:53 2021

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

def getCollege(text, keyword_processor):
    """Function to extract college names from resumes/cvs"""
    if text == "Fail":
        return "Fail"
    text = clean_special_text(text)
    college_ls = keyword_processor.extract_from_text(text)
    return college_ls

def getDegree(text, keyword_processor):
    """Function to extract degree names from resumes/cvs"""
    if text == "Fail":
        return "Fail"
    text = clean_special_text(text)
    degree_ls = keyword_processor.extract_from_text(text)
    return degree_ls

def getEducation(text, college_output, degree_output):
    """
    Function to Link College & Degree based on distances
    
    a) Input: Text, Extracted Colleges & Degree
    
    b) Output: Mapped Colleges, Degree
    """
    if text == "Fail":
        return "Fail"
    
    clean_txt = clean_special_text(text)
    
    out_ls = []
    if (len(college_output) > 0) and (len(degree_output) > 0):
        for ls_1 in college_output:
            ls = []
            for ls_2 in degree_output:
                _dict = {'college':ls_1['keyword'], 'degree':ls_2['keyword'], 'distance':'', 
                         'college_start_idx':ls_1['start_idx'], 'college_end_idx':ls_1['end_idx'],
                         'degree_start_idx':ls_2['start_idx'], 'degree_end_idx':ls_2['end_idx']}
                college_end_idx = ls_1['end_idx']
                degree_start_idx = ls_2['start_idx']
                _dict['distance'] = np.abs(college_end_idx-degree_start_idx)
                if (college_end_idx > degree_start_idx):
                    _dict['words'] = len(clean_txt[degree_start_idx:college_end_idx].strip().split())
                else:
                    _dict['words'] = len(clean_txt[college_end_idx:degree_start_idx].strip().split())

                if _dict['words'] > 30: ## If # of words between college & degree exceed 15 then don't assign [Assumption]
                    _dict['degree'] = ''
                ls.append(_dict)
            out_ls.append(min(ls, key=lambda x:x['distance']))
    out_ls = sorted(out_ls, key = lambda i: (i['degree'], i['words']), reverse=True)
    out_ls = list({i['college']:i for i in reversed(out_ls)}.values()) ## Remove dups
    return out_ls