import nltk, os, subprocess, code, glob, re, traceback, sys, inspect

def getEmail(inputString):
    email = None
    try:
        # email = ['abhishek.mathur@gmail.com']
        # pattern = re.compile(r'\S*@\S*')
        # matches = pattern.findall(inputString) # Gets all email addresses as a list
        # matches = re.findall(r'[\w\.-]+@[\w\.-]+', inputString)
        matches = re.findall("[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",inputString)
        email = matches
        email = list(set(matches))
    except Exception as e:
        return e

    return email


# print(getEmail("abhishek@gmail.com"))