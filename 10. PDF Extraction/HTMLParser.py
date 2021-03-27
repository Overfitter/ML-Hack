"""
HTML parser
"""

from html.parser import HTMLParser
from html.entities import name2codepoint
from unidecode import unidecode

class MyHTMLParser(HTMLParser):
    """
    wrapper to default HTML parser in python writes text to a file.
    Divides html document into following parts:
    1) Strat tag
    2) End tag
    3) Data
    4) Comment

    Writes a tripple pipe (|||) separated file to given directory
    with two columns "tag_type" and "tag_value"

    Parameters
    ----------
    file_name: str
    	text file name to write output of HTML parsed file
    
    Returns
    -------
    None
    """
    def __init__(self, file_name):
        HTMLParser.__init__(self)
        self.file_name = file_name
        
    def handle_starttag(self, tag, attrs):
        print("Start tag ||| ", unidecode(tag), file=open(self.file_name, "a+"))
        for attr in attrs:
            print("attr ||| ", tuple(unidecode(x) for x in attr), file=open(self.file_name, "a+"))

    def handle_endtag(self, tag):
        print("End tag ||| ", unidecode(tag), file=open(self.file_name, "a+"))

    def handle_data(self, data):
        print("Data ||| ", unidecode(data), file=open(self.file_name, "a+"))

    def handle_comment(self, data):
        print("Comment ||| ", unidecode(data), file=open(self.file_name, "a+"))

    def handle_entityref(self, name):
        c = chr(name2codepoint[name])
        print("Named ent ||| ", unidecode(c), file=open(self.file_name, "a+"))

    def handle_charref(self, name):
        if name.startswith('x'):
            c = chr(int(name[1:], 16))
        else:
            c = chr(int(name))
        print("Num ent ||| ", unidecode(c), file=open(self.file_name, "a+"))

    def handle_decl(self, data):
        print("Decl ||| ", unidecode(data), file=open(self.file_name, "a+"))
