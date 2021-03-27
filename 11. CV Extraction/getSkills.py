# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:48:02 2021

@author: Jitendra
"""
## Loading Required Modules

# General Utilities
import numpy as np
import re
import string

## Text Preprocessing Functions

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

def getSkills(text, keyword_processor):
    """Function to extract skills from Resume/CVs"""
    text = clean_special_text(text)
    skill_ls = keyword_processor.extract_from_text(text)
    return skill_ls
