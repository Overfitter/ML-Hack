import nltk, os, subprocess, code, glob, re, traceback, sys, inspect

def getPhone(inputString):

    number = None
    try:
        pattern = re.compile(r'([+(]?\d+[)\-]?[ \t\r\f\v]*[(]?\d{2,}[()\-]?[ \t\r\f\v]*\d{2,}[()\-]?[ \t\r\f\v]*\d*[ \t\r\f\v]*\d*[ \t\r\f\v]*)')
            # Understanding the above regex
            # +91 or (91) -> [+(]? \d+ -?
            # Metacharacters have to be escaped with \ outside of character classes; inside only hyphen has to be escaped
            # hyphen has to be escaped inside the character class if you're not incidication a range
            # General number formats are 123 456 7890 or 12345 67890 or 1234567890 or 123-456-7890, hence 3 or more digits
            # Amendment to above - some also have (0000) 00 00 00 kind of format
            # \s* is any whitespace character - careful, use [ \t\r\f\v]* instead since newlines are trouble
        match = pattern.findall(inputString)
        # match = [re.sub(r'\s', '', el) for el in match]
            # Get rid of random whitespaces - helps with getting rid of 6 digits or fewer (e.g. pin codes) strings
        # substitute the characters we don't want just for the purpose of checking
        match = [re.sub(r'[,.]', '', el) for el in match if len(re.sub(r'[()\-.,\s+]', '', el))>6]
            # Taking care of years, eg. 2001-2004 etc.
        match = [re.sub(r'\D$', '', el).strip() for el in match]
            # $ matches end of string. This takes care of random trailing non-digit characters. \D is non-digit characters
        match = [el for el in match if len(re.sub(r'\D','',el)) <= 15]
            # Remove number strings that are greater than 15 digits
        try:
            for el in list(match):
                # Create a copy of the list since you're iterating over it
                if len(el.split('-')) > 3: continue # Year format YYYY-MM-DD
                for x in el.split("-"):
                    try:
                        # Error catching is necessary because of possibility of stray non-number characters
                        # if int(re.sub(r'\D', '', x.strip())) in range(1900, 2100):
                        if x.strip()[-4:].isdigit():
                            if int(x.strip()[-4:]) in range(1900, 2100):
                                # Don't combine the two if statements to avoid a type conversion error
                                match.remove(el)
                    except:
                        pass
        except:
            pass
        number = list(set(match))
    except:
        pass

    return number