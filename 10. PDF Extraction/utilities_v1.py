"""
This python script ccontains a set of helper functions 
to parse a pdf document.
"""
######################## IMPORTING PACKAGES #######################
import re
import os
#from io import StringIO, BytesIO
import warnings
warnings.filterwarnings("ignore")
import gc
import pickle

import pandas as pd
import numpy as np
import string
import fitz
import camelot
from fuzzywuzzy import fuzz 
#from unidecode import unidecode
from HTMLParser import MyHTMLParser

#from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
#from pdfminer.converter import TextConverter, XMLConverter, HTMLConverter, PDFPageAggregator
#from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTImage
#from pdfminer.pdfpage import PDFPage
#from pdfminer.pdfparser import PDFParser
#from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines


#########################################################################################################################################
#########################################################################################################################################

class newStack:
    def __init__(self):
        self.stack = list()
        self.stack_count = 0
    
    def push(self, data):
        self.stack.append(data)
        self.stack_count+=1
    
    def pop(self):
        temp_data = self.stack.pop()
        self.stack_count-=1
        return(temp_data)
    
    def emptyStack(self):
        self.stack = list()
        self.stack_count = 0
        
    def isEmpty(self):
        if self.stack_count==0: return(True)
        else: return(False)
        
class newQueue:
    def __init__(self):
        self.queue = list()
        self.queue_count = 0
    
    def push(self, data):
        self.queue.insert(0, data)
        self.queue_count+=1
        
    def pop(self):
        temp_data = self.queue.pop()
        self.queue_count-=1
        return(temp_data)
        
    def emptyQueue(self):
        self.queue = list()
        self.queue_count = 0
        
    def isEmpty(self):
        if self.queue_count==0: return(True)
        else: return(False)
             


class TOC_extract:
    def __init__(self, path):
        self.path = path
        self.toc = self.get_toc()

    def __getSection(self, x):
    	#pattern=re.compile(r'[(\d+\.)| (Figure\s?\d+)| (Table\s?\d+)]+',re.I)
    	pattern=re.compile(r'^[(\d+\.)]+',re.I)
    	match=re.findall(pattern,x)
    	if len(match)==0:
            return None
    	return match[0]
 
    
    def __replaceSection(self, x):
        return(re.sub(r'^[(\d+\.)]+', "", x))

    def __extractSection(self, df):
        df['section'] = [self.__getSection(x) for x in df['contents']]
        df['contents'] = [self.__replaceSection(x) for x in df['contents']]
        return(df)
    
    def get_toc(self):
        doc = fitz.open(self.path)
        toc = doc.getToC()
        if(len(toc))==0:
            toc = None
        else:
            toc = pd.DataFrame(toc)
            toc.drop(columns=0, inplace=True)
            toc.rename(columns={1:'contents', 2:'page_no'}, inplace=True)
            toc = self.__extractSection(toc)
        return(toc)

'''
class PDF_converter:
    """
    Converts pdf into following formats:
    1) HTML (only basic tags like br, div , style etc..)
    2) XML (not reliable)
    3) Text with escape characters 

    Parameters
    ----------
    pdf_file_path: str
        path of the pdf file to be converted
    format_: str
        "text", "html", "xml"
    codec: str
        'utf-8', 'latin-1' etc.
    password: str
        only if pdf is passwod protected
    save_converted_file: boolean
        whether to save output file or not
    save_converted_file_path: str
        directory to save the converted file

    Returns
    -------
    str:
        text with html markups or plain text with escape characters
        depending on the format_
    """
    def __init__(self, pdf_file_path, format_='html', codec='utf-8', password='', save_converted_file=True, save_converted_file_dir=None):
        self.pdf_file_path = pdf_file_path
        self.format_ = format_.lower().strip()
        self.codec = codec.lower().strip()
        self.password = password
        self.save_converted_file = save_converted_file
        self.save_converted_file_dir = save_converted_file_dir if(save_converted_file_dir) else '\\'.join(self.pdf_file_path.split('\\')[:-1])
        self.file_name = self.pdf_file_path.split('\\')[-1]
        self.converted_output = self.convert_pdf()
        
    def __save_output(self, text):
        """
        Saves an TXT/HTML/XML to a file after decoding characters using unidecode
    
        Parameters
        ----------
        text: str
            HTML text
    
        Returns
        -------
        None
        """
        if self.format_ == 'text': file_extension = '.txt'
        elif self.format_ == 'html': file_extension = '.html'
        elif self.format_ == 'xml': file_extension = '.xml'
        else: raise ValueError('provide format, either text, html or xml!')
        
        with open(os.path.join(self.save_converted_file_dir, self.file_name.rstrip('.pdf')+file_extension), 'w') as f:
            f.write(unidecode(text))
            
    def convert_pdf(self):
        rsrcmgr = PDFResourceManager()
        retstr = BytesIO()
        
        laparams = LAParams()
        if self.format_ == 'text':
            device = TextConverter(rsrcmgr, retstr, codec=self.codec, laparams=laparams)
        elif self.format_ == 'html':
            device = HTMLConverter(rsrcmgr, retstr, codec=self.codec, laparams=laparams)
        elif self.format_ == 'xml':
            device = XMLConverter(rsrcmgr, retstr, codec=self.codec, laparams=laparams)
        else:
            raise ValueError('provide format, either text, html or xml!')
        
        fp = open(self.pdf_file_path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        maxpages = 0
        caching = True
        pagenos=set()
        pages = PDFPage.get_pages(fp, pagenos, maxpages=maxpages, caching=caching, check_extractable=True, password=self.password)
        
        for page in pages:
            interpreter.process_page(page)
        
        converted_doc = retstr.getvalue().decode()
        fp.close()
        device.close()
        retstr.close()
        if self.save_converted_file: self.__save_output(converted_doc)
        return(converted_doc)
'''        
class PDF_converter:
    def __init__(self, pdf_file_path, password='', save_converted_file=True, save_converted_file_dir=None):
        self.pdf_file_path = pdf_file_path
        self.password = password
        self.save_converted_file = save_converted_file
        #self.save_converted_file_dir = save_converted_file_dir if(save_converted_file_dir) else '\\'.join(self.pdf_file_path.split('\\')[:-1])
        self.save_converted_file_dir = save_converted_file_dir if(save_converted_file_dir) else os.path.split(self.pdf_file_path)[0]
        #self.file_name = self.pdf_file_path.split('\\')[-1]
        self.file_name = os.path.split(self.pdf_file_path)[-1]
        self.prot_name= self.file_name.rstrip('.pdf').strip()
        self.converted_output = self.convert_pdf()
        
    def __save_output(self, file):
        file_extension = '.html'
        #temp_file = open(os.path.join(self.save_converted_file_dir, self.file_name.rstrip('.pdf')+file_extension), 'w', encoding='utf-8')
        with open(os.path.join(self.save_converted_file_dir, self.prot_name+file_extension), 'w', encoding='utf-8') as temp_file:
            temp_file.writelines(file)
        #temp_file.close()
        
    def __correct_page_number(self, txt, i):
        pattern = r'id="page(\d+)"'
        rep = 'id="page'+str(i)+'"'
        return(re.sub(pattern, rep, txt))
    
    def convert_pdf(self):
        doc = fitz.open(self.pdf_file_path)
        html_list = []
        page_counter = 0
        for page in doc:
            page_counter+=1
            html_text = page.getText("html")
            html_text = self.__correct_page_number(html_text, page_counter)
            html_list.append(html_text)
            gc.collect()
        if self.save_converted_file:
            self.__save_output(html_list)
        return(html_list)


'''
class HTML_to_DF_converter:
    
    def __get_page_no(self, txt):
        """
        gives page number from data column from keyword "('name', '1')"
    
        Parameters
        ----------
        txt: str
            applied to a tag value column
    
        Returns
        ------- 
        int:
            page number if regex match is found else NaN
        """
        if txt is not None:
            txt = re.findall("('name', '[0-9]*')", txt)
            number = np.nan if len(txt) == 0 else int("".join(re.findall("[0-9]*", txt[0])))
            return(number)
        return(np.nan)
    
    def __get_font_size(self, txt):
        """
        Function to be applied to the tag_value column of HTML parsed dataframe
        for extracting font size 
    
        Parameters
        ----------
        txt: str
            attribute text of HTML markup for extracting font-size
        
        Returns
        -------
        int:
            font size from style attribute of HTML markup
        """
        if txt is not None:
            font_size_txt = re.findall("font-size:[0-9]*px", txt)            
            font_size = np.nan if len(font_size_txt) == 0 else int("".join(re.findall("[0-9]*", font_size_txt[0])))
            return(font_size)
        return(np.nan)

    def __get_font_style(self, txt):
        """
        Extracts font style from html attributes
    
        Parameters:
        ----------
        txt: str
            html attribute 
        
        Returns
        -------
        str
            font style like Arial etc. and NaN if not style is present in attribute
        """
        if txt is not None:
            font_fam_txt = re.findall("font\-family: b'.*'", txt)            
            font_fam = np.nan if len(font_fam_txt) == 0 else font_fam_txt[0].replace("font-family: b", "")
            return(font_fam)
        return(np.nan)

    def __get_italic_flag(self, txt):
        """
        Extracts italic flag from html attributes
    
        Parameters:
        ----------
        txt: str
            html attribute 
        
        Returns
        -------
        bool
            True if italic, False otherwise
        """
        return(bool(re.search("Italic", txt)))

    def __get_bold_flag(self, txt):
        """
        Extracts bold flag from html attributes
    
        Parameters:
        ----------
        txt: str
            html attribute 
        
        Returns
        -------
        bool
            True if bold, False otherwise
        """
        return(bool(re.search("Bold", txt)))
        
    def __html_to_df(self, file_name, delim='\|\|\|'):
        """
        Converts the file written by "MyHTMLParser" into dataframe with two columns
        "tag_type" and "tag_value"
    
        Parameters
        ----------
        file_name: str
            path of file written by MyHTMLParser
        delim: str 
            separator for converting parsed HTML file to dataframe
    
        Returns
        -------
        dataframe:
            columns - ["tag_type", "tag_value"]
        """
        df = pd.read_table(file_name, delimiter= delim, header=None, skip_blank_lines=True)
        df.rename(columns={0:"tag_type", 1:"tag_value"}, inplace=True)  
        return(df)
     
    def parse_html(self, html_file, save_parsed_file_path):
        html_parser = MyHTMLParser(save_parsed_file_path)
        html_parser.feed(html_file)
        
    def create_protocol_dataframe(self, file_path):
        prot_df = self.__html_to_df(file_path)
        prot_df["tag_type"] = prot_df["tag_type"].str.strip()
        prot_df["tag_value"] = prot_df["tag_value"].str.strip()
        prot_df = prot_df.dropna().reset_index(drop=True)

        prot_df["font_size"] = prot_df["tag_value"].astype('str').apply(lambda x: self.__get_font_size(x))
        prot_df["font_style"] = prot_df["tag_value"].astype('str').apply(lambda x: self.__get_font_style(x))
        prot_df["page_no"] = prot_df["tag_value"].astype('str').apply(lambda x: self.__get_page_no(x))

        prot_df["font_size"].fillna(method='ffill', inplace=True)
        prot_df["font_style"].fillna(method='ffill', inplace=True)
        prot_df["page_no"].fillna(method='ffill', inplace=True)

        prot_df["font_size"].fillna(0, inplace=True)
        prot_df["font_style"].fillna("unknown", inplace=True)
        prot_df["page_no"].fillna(0, inplace=True)

        prot_df["page_no"] = prot_df["page_no"].apply(lambda x: int(x))

        prot_df["is_bold"] = prot_df["font_style"].apply(lambda x: self.__get_bold_flag(x))
        prot_df["is_italic"] = prot_df["font_style"].apply(lambda x: self.__get_italic_flag(x))
        return(prot_df)
'''

class HTML_to_DF_converter:
  
    def __init__(self):
        self.page_number_pattern = r"\('id',\s?'page(\d+)'\)"
        self.font_size_pattern = r'font-size:(.*)pt'
        self.font_fam_pattern = r'font-family:(.*);'
    
    def __get_page_no(self, txt):
        """
        gives page number from data column from keyword "('name', '1')"
    
        Parameters
        ----------
        txt: str
            applied to a tag value column
    
        Returns
        ------- 
        int:
            page number if regex match is found else NaN
        """
        if txt is not None:
            pattern = self.page_number_pattern
            page_no_ls = re.findall(pattern, txt)
            page_no = np.nan if len(page_no_ls) == 0 else int("".join(re.findall("[0-9]*", page_no_ls[0])))
            return(page_no)
        return(np.nan)
    
    def __get_font_size(self, txt):
        """
        Function to be applied to the tag_value column of HTML parsed dataframe
        for extracting font size 
    
        Parameters
        ----------
        txt: str
            attribute text of HTML markup for extracting font-size
        
        Returns
        -------
        int:
            font size from style attribute of HTML markup
        """
        if txt is not None:
            pattern = self.font_size_pattern
            font_size_ls = re.findall(pattern, txt)            
            font_size = np.nan if len(font_size_ls) == 0 else int(float("".join(re.findall("[0-9.]*", font_size_ls[0]))))
            return(font_size)
        return(np.nan)

    def __get_font_style(self, txt):
        """
        Extracts font style from html attributes
    
        Parameters:
        ----------
        txt: str
            html attribute 
        
        Returns
        -------
        str
            font style like Arial etc. and NaN if not style is present in attribute
        """
        if txt is not None:
            pattern = self.font_fam_pattern
            font_fam_ls = re.findall(pattern, txt)            
            font_fam = np.nan if len(font_fam_ls) == 0 else str(font_fam_ls[0]).strip()
            return(font_fam)
        return(np.nan)

    def __get_italic_flag(self, txt):
        """
        Extracts italic flag from html attributes
    
        Parameters:
        ----------
        txt: str
            html attribute 
        
        Returns
        -------
        bool
            True if italic, False otherwise
        """
        return(bool(re.search("Italic", txt)))

    def __get_bold_flag(self, txt):
        """
        Extracts bold flag from html attributes
    
        Parameters:
        ----------
        txt: str
            html attribute 
        
        Returns
        -------
        bool
            True if bold, False otherwise
        """
        return(bool(re.search("Bold", txt)))
        
    def html_to_df(self, file_name, delim='\|\|\|'):
        """
        Converts the file written by "MyHTMLParser" into dataframe with two columns
        "tag_type" and "tag_value"
    
        Parameters
        ----------
        file_name: str
            path of file written by MyHTMLParser
        delim: str 
            separator for converting parsed HTML file to dataframe
    
        Returns
        -------
        dataframe:
            columns - ["tag_type", "tag_value"]
        """
        df = pd.read_table(file_name, delimiter= delim, header=None, skip_blank_lines=True)
        df.rename(columns={0:"tag_type", 1:"tag_value"}, inplace=True)  
        return(df)
     
    def parse_html(self, html_file, save_parsed_file_path):
        html_parser = MyHTMLParser(save_parsed_file_path)
        html_parser.feed(html_file)
        
    def create_protocol_dataframe(self, file_path):
        prot_df = self.html_to_df(file_path)
        prot_df["tag_type"] = prot_df["tag_type"].str.strip()
        prot_df["tag_value"] = prot_df["tag_value"].str.strip()
        prot_df = prot_df.dropna().reset_index(drop=True)

        prot_df["font_size"] = prot_df["tag_value"].astype('str').apply(lambda x: self.__get_font_size(x))
        prot_df["font_style"] = prot_df["tag_value"].astype('str').apply(lambda x: self.__get_font_style(x))
        prot_df["page_no"] = prot_df["tag_value"].astype('str').apply(lambda x: self.__get_page_no(x))

        prot_df["font_size"].fillna(method='ffill', inplace=True)
        prot_df["font_style"].fillna(method='ffill', inplace=True)
        prot_df["page_no"].fillna(method='ffill', inplace=True)

        prot_df["font_size"].fillna(0, inplace=True)
        prot_df["font_style"].fillna("unknown", inplace=True)
        prot_df["page_no"].fillna(0, inplace=True)

        prot_df["page_no"] = prot_df["page_no"].apply(lambda x: int(x))

        prot_df["is_bold"] = prot_df["font_style"].apply(lambda x: self.__get_bold_flag(x))
        prot_df["is_italic"] = prot_df["font_style"].apply(lambda x: self.__get_italic_flag(x))
        
        prot_df = prot_df.loc[prot_df['tag_type'].isin(['Data', 'data', 'Start tag', 'End tag', 'attr'])].reset_index(drop=True)
        
        return(prot_df)
        
        
'''        
class ContentExtractor:
    def __init__(self, prot_data, prot_name,toc_to_ignore_starts_with=('list', 'table', 'attachment','figure')):
        self.prot_name=prot_name
        self.df=prot_data[prot_name]['prot_df']
        self.df_only_data=self.df[self.df['tag_type']=='Data']
        self.df_bold_data=self.df_only_data[self.df_only_data['is_bold']==True]
        self.toc=prot_data[prot_name]['toc']
        self.toc_after_ignore=self.toc[[not self._check_if_starts_with(x,toc_to_ignore_starts_with) 
                                        for x in self.toc['contents']]].reset_index(drop=True)
        self.all_contents=self._getAllContents()
        self.all_contents_clubbed={}
        self.__stackList=[]
        self._update_all_content_with_section_none()
        #self.all_contents_clubbed.update(self._getClubbedContent_list_based(0,len(self.toc_after_ignore)-1))
        
    def _replaceSection(self, x): ######
        return re.sub(r'^[(\d+\.)]+',"",x)
        
    def _check_if_starts_with(self,string, tuple_to_check):
        return string.strip().lower().startswith(tuple_to_check)
    
    def _leveshteinIndex(self, x, list_to_match): ######
        ratios=[fuzz.ratio(x,y) for y in list_to_match]
        ind=ratios.index(max(ratios))
        return ind
    
    def _find_section_index(self, df,section_1,section_2,isLast=False): ######
        df=df.reset_index()
        if not isLast:
            return df['index'].values[self._leveshteinIndex(self._replaceSection(section_1.strip()),df['tag_value'])],\
                   df['index'].values[self._leveshteinIndex(self._replaceSection(section_2.strip()),df['tag_value'])]
        else:
            return df['index'].values[self._leveshteinIndex(self._replaceSection(section_1.strip()),df['tag_value'])]
        
    def _get_content_before_next_heading(self,index):  ######
        if index<len(self.toc_after_ignore)-1:
            section_data=self.df_only_data[(self.df_only_data['page_no']>=self.toc_after_ignore['page_no'].values[index]) 
                                           &(self.df_only_data['page_no']<=self.toc_after_ignore['page_no'].values[index+1])
                                          ].reset_index(drop =True)
            first_sec_index,second_sec_index=self._find_section_index(
                                                                    section_data[section_data['is_bold']==True],
                                                                    self.toc_after_ignore['contents'].values[index],
                                                                    self.toc_after_ignore['contents'].values[index+1]
                                                                    )
            trimmed_section_data=section_data.iloc[first_sec_index+1:second_sec_index,:]
            return " ".join(list(trimmed_section_data['tag_value'].values))
        else :
            section_data=self.df_only_data[(self.df_only_data['page_no']>=self.toc_after_ignore['page_no'].values[index])]
            sec_index=self._find_section_index(section_data[section_data['is_bold']==True],
                                               self.toc_after_ignore['contents'].values[index],
                                               self.toc_after_ignore['contents'].values[index],isLast=True)
            trimmed_section_data=section_data.iloc[sec_index+1:,:]
            return " ".join(list(trimmed_section_data['tag_value'].values))
        
    def _getAllContents(self): ######
        content={}
        for i in range(len(self.toc_after_ignore)):
            try:
                content.update({self.toc_after_ignore['contents'].values[i]: self._get_content_before_next_heading(i)})
            except IndexError as e:
                print("match not found for protocol {} and section {}".format(
                    self.prot_name,self.toc_after_ignore['contents'].values[i]))
        return content
    
    def _getNextSectionIndex(self,sec):
        #TODO
        return
        
    def _update_all_content_with_section_none(self):
        for i in range(len(self.toc_after_ignore)):
            if self.toc_after_ignore['section'].values[i] is None:
                self.all_contents_clubbed.update({self.toc_after_ignore['contents'].values[i]:
                                                      self._get_content_before_next_heading(i)})
        
    def _getClubbedContent(self,index):
        if index==len(self.toc_after_ignore):
            return
        if self.toc_after_ignore['section'].values[index] is None:
            self.all_contents_clubbed.update({self.toc_after_ignore['contents'].values[index]:
                                              self._get_content_before_next_heading(index)})
            self._getClubbedContent(index+1)
        self.__stackList.append(self.toc_after_ignore['section'].values[index])
        currSection=self.__stackList.pop()
        #TODO
        return
    
    def _getClubbedContent_dict_based(self):
        for i in range(self.toc_after_ignore):
            if self.toc_after_ignore['section'].values[i] is None:
                if not 0 in self.all_contents_clubbed.keys():
                    self.all_contents_clubbed.update({0:{self.toc_after_ignore['contents'].values[i]:
                                                         self._get_content_before_next_heading(i)}})
                else:
                    self.all_contents_clubbed[0].update({self.toc_after_ignore['contents'].values[i]:
                                                         self._get_content_before_next_heading(i)})
            else:
                #TODO
                return
    
    def _getListOfSiblings(self, index,lastIndex):
        sections=[]
        #print("index={}, last_index={}".format(index,lastIndex))
        sections_with_index=[]
        sec=self.toc_after_ignore['section'].astype(str).values[index]
        len_to_check=len(sec.split("."))-1
        sections.append(sec)
        for s in self.toc_after_ignore['section'].values[index:lastIndex+1]:
            if not s is None:
                #print(s[:len(sec)])
                sib=".".join(s.split(".")[:len_to_check])+"."
                #sections.add(".".join(s.split(".")[:len_to_check])+".")
                if not sib in sections:
                    sections.append(sib)
                #print(sections)
        #print(sections)        
            
        for s in sections:
            #print(s)
            sections_with_index.append((s,self.toc_after_ignore['section'].tolist().index(s)))
        #print(sections_with_index)
        return sections_with_index
        
    
    def _getClubbedContent_list_based(self,start_index, last_index):
        if start_index>=last_index:
            return {self.toc_after_ignore['contents'].values[start_index]:
                            {'main_content':self._get_content_before_next_heading(start_index)}}
        while self.toc_after_ignore['section'].values[start_index] is None:
            start_index+=1
        
        while self.toc_after_ignore['section'].values[last_index] is None:
            last_index-=1
            
        #print("start_index={}, last_index={}".format(start_index,last_index))
        content={}
        siblingList=self._getListOfSiblings(start_index,last_index)
        #print("Sibling List")
        #print(siblingList)
        for i in range(len(siblingList)):
            if i<len(siblingList)-1:
#                 content.update({self.toc_after_ignore['contents'].values[siblingList[i][1]]:
#                             {'main_content':self._get_content_before_next_heading(siblingList[i][1])}
#                                 .update(self._getClubbedContent_list_based(siblingList[i][1]+1,siblingList[i+1][1]-1))})
                content.update({self.toc_after_ignore['contents'].values[siblingList[i][1]]:
                                {'main_content':self._get_content_before_next_heading(siblingList[i][1])}})
                if siblingList[i][1]+1!=siblingList[i+1][1]:
                    if siblingList[i][1]+1<len(self.toc_after_ignore):
                        content[self.toc_after_ignore['contents'].values[siblingList[i][1]]].update(
                                    self._getClubbedContent_list_based(siblingList[i][1]+1,siblingList[i+1][1]-1))

                                
                #print(content[self.toc_after_ignore['contents'].values[siblingList[i][1]]])
            else:
                content.update({self.toc_after_ignore['contents'].values[siblingList[i][1]]:
                                {'main_content':self._get_content_before_next_heading(siblingList[i][1])}})
                if (siblingList[i][1]+1<last_index)&(siblingList[i][1]+1<len(self.toc_after_ignore)):
                    content[self.toc_after_ignore['contents'].values[siblingList[i][1]]].update(
                                self._getClubbedContent_list_based(siblingList[i][1]+1,last_index))

        #print(content)
        return content
'''

class ContentExtractor:
    def __init__(self, prot_df, toc, ignore_section_prefix=('list', 'attachment','figure'), matching_threshold=90):
        self.threshold_same = matching_threshold
        self.threshold_part = 95
        self.prot_df = prot_df.reset_index()
        self.raw_toc = toc
        self.toc = self.raw_toc[[not self._check_if_starts_with(x, ignore_section_prefix) for x in self.raw_toc['contents']]].reset_index(drop=True)
        self.toc['contents'] = self.toc['contents'].apply(lambda x: x.replace('"',"'"))
        self.unstructured_data_dic = self.unstructured_section_extraction()
        self.structured_data_dic = self.structured_section_extraction()
        
    def _check_if_starts_with(self, phrase, tuple_to_check):
        return phrase.strip().lower().startswith(tuple_to_check)
    
    def __isSame(self, phrase1, phrase2):
        THRESHOLD = self.threshold_same
        match_flag = False
        phrase1, phrase2 = str(phrase1).lower().strip(), str(phrase2).lower().strip()
        if fuzz.ratio(phrase1, phrase2) >= THRESHOLD:
            match_flag = True
        else:
            match_flag = False
        return(match_flag)
        
    def __isParent(self, parent, child):
        if len(parent)<len(child):
            if parent == child[:-1]:
                return(True)
            else:
                return(False)
        else:
            return(False)
            
    def __isSubPart(self, phrase1, phrase2):
        ''' phrase2 is sub-part od phrase1'''
        THRESHOLD = self.threshold_part
        part_flag = False
        phrase1, phrase2 = str(phrase1).lower().strip(), str(phrase2).lower().strip()
        if fuzz.partial_ratio(phrase1, phrase2) >= THRESHOLD:
            part_flag = True
        else:
            part_flag = False
        return(part_flag)
    
    def __StartWith(self, phrase1, phrase2):
        ''' Phrase1 starts with phrase2'''
        phrase1 = list(map(str.strip, phrase1.split(' ')))
        phrase2 = list(map(str.strip, phrase2.split(' ')))
        phrase1 = ' '.join(phrase1[:len(phrase2)])
        phrase2 = ' '.join(phrase2)
        return(self.__isSame(phrase1, phrase2))
        
    def __extract_from_df(self, start_page, end_page, start_title, next_title, isLastTitle=False):
        is_start_title_found = False
        is_next_title_found = False
        data_start_idx = -1
        data_end_idx = -1
        output_data = ''
        page_no = list(range(start_page, self.prot_df['page_no'].max()+1)) if(isLastTitle) else list(range(start_page, end_page+1))
        temp_df = self.prot_df.loc[(self.prot_df['page_no'].isin(page_no))&(self.prot_df['tag_type']=='Data')&(self.prot_df['is_bold']==True)].reset_index(drop=True)
        temp_df['checkFlag'] = 1
        row_idx = 0
        while row_idx < temp_df.shape[0]:
            if re.sub('[^A-Za-z]+', '', temp_df.loc[row_idx, 'tag_value']).lower()=='table' and row_idx+1<temp_df.shape[0] and temp_df.loc[row_idx, 'page_no']==temp_df.loc[row_idx+1, 'page_no']:
                temp_df.loc[row_idx, 'tag_value'] = temp_df.loc[row_idx, 'tag_value'] + ' ' + temp_df.loc[row_idx+1, 'tag_value']
                temp_df.loc[row_idx, 'index'] = temp_df.loc[row_idx+1, 'index']
                temp_df.loc[row_idx+1, 'checkFlag'] = 0
                row_idx+=2
            else:
                row_idx+=1
        temp_df = temp_df.loc[temp_df['checkFlag']==1].reset_index(drop=True)
        
        row_idx = 0
        while row_idx < temp_df.shape[0] and not(is_next_title_found):
            #print(row_idx)
            if not(is_start_title_found) and (self.__isSame(start_title, temp_df.loc[row_idx, 'tag_value'])):
                data_start_idx = temp_df.loc[row_idx, 'index']+1
                is_start_title_found = True
                row_idx+=1
            elif not(is_start_title_found) and (self.__StartWith(start_title, temp_df.loc[row_idx, 'tag_value'])):
                temp_continue_flag = True
                temp_str = ' '.join(map(str.strip, temp_df.loc[row_idx, 'tag_value'].split(' ')))
                row_idx+=1
                temp_row_idx = row_idx
                while row_idx < temp_df.shape[0] and not(is_start_title_found) and temp_continue_flag:
                    temp_str+=' '+' '.join(map(str.strip, temp_df.loc[row_idx, 'tag_value'].split(' ')))
                    if self.__isSubPart(start_title, temp_str):
                        if (len(temp_str.split(' '))>=len(start_title.split(' '))-1) and (self.__isSame(start_title, temp_str)):
                            data_start_idx = temp_df.loc[row_idx, 'index']+1
                            is_start_title_found = True
                            break
                        else:
                            row_idx+=1
                    else:
                        row_idx = temp_row_idx
                        temp_continue_flag = False      
            elif not(isLastTitle) and is_start_title_found and not(is_next_title_found) and (self.__isSame(next_title, temp_df.loc[row_idx, 'tag_value'])):
                data_end_idx = temp_df.loc[row_idx, 'index']-2
                is_next_title_found = True
                break
            elif not(isLastTitle) and is_start_title_found and not(is_next_title_found) and (self.__StartWith(next_title, temp_df.loc[row_idx, 'tag_value'])):
                temp_continue_flag = True
                temp_str = ' '.join(map(str.strip, temp_df.loc[row_idx, 'tag_value'].split(' ')))
                row_idx+=1
                temp_row_idx = row_idx
                while row_idx < temp_df.shape[0] and not(is_next_title_found) and temp_continue_flag:
                    temp_str+=' '+' '.join(map(str.strip, temp_df.loc[row_idx, 'tag_value'].split(' ')))
                    if self.__isSubPart(next_title, temp_str):
                        if (len(temp_str.split(' '))>=len(next_title.split(' '))-1) and (self.__isSame(next_title, temp_str)):
                            data_end_idx = temp_df.loc[temp_row_idx-1, 'index']-2
                            is_next_title_found = True
                            break
                        else:
                            row_idx+=1
                    else:
                        row_idx = temp_row_idx
                        temp_continue_flag = False
            else:
                row_idx+=1
                
        if not(is_start_title_found):
            output_data = ''
        elif is_start_title_found and not(is_next_title_found):
            if isLastTitle:
                output_data = ' '.join(map(lambda x: str(x).strip(), self.prot_df.loc[(self.prot_df['index'].isin(list(range(data_start_idx, data_end_idx+1))))&(self.prot_df['tag_type']=='Data')]['tag_value'].tolist()))
            else:
                output_data = ''
        elif is_start_title_found and is_next_title_found:
            output_data = ' '.join(map(lambda x: str(x).strip(), self.prot_df.loc[(self.prot_df['index'].isin(list(range(data_start_idx, data_end_idx))))&(self.prot_df['tag_type']=='Data')]['tag_value'].tolist()))
        #print('')
        #print(start_page, end_page, start_title, next_title, isLastTitle)
        #print(is_start_title_found, is_next_title_found, data_start_idx, data_end_idx)
        return(output_data)
            
    def unstructured_section_extraction(self):
        data_dic = dict()
        for row_idx in range(self.toc.shape[0]):
            start_page = int(self.toc.loc[row_idx, 'page_no'])
            end_page = None if(row_idx==self.toc.shape[0]-1) else int(self.toc.loc[row_idx+1, 'page_no'])
            start_title = self.toc.loc[row_idx, 'contents'].strip()
            next_title = None if(row_idx==self.toc.shape[0]-1) else self.toc.loc[row_idx+1, 'contents'].strip()
            data_dic[start_title] = self.__extract_from_df(start_page, end_page, start_title, next_title, isLastTitle = True if(row_idx==self.toc.shape[0]-1) else False)
        return(data_dic)
        
    
    def structured_section_extraction(self):
        data_dic = dict()   #  WARNING : << DONT CHANGE THIS VARIABLE NAME EVER >>
        #section = self.toc['section'].tolist()
        sec_stack = newStack()
        key_stack = newStack()
        #idx = 0
        #while idx < len(section):
        row_idx = 0
        while row_idx < self.toc.shape[0]:
            #curr_sec = list(map(int, section[idx].rstrip('.').split('.')))
            curr_sec = list(map(int, self.toc.loc[row_idx, 'section'].rstrip('.').split('.')))
            key = self.toc.loc[row_idx, 'contents'].strip()
            key_value = self.unstructured_data_dic[key].replace('"', "'").replace(chr(0),'')
            if sec_stack.isEmpty():
                sec_stack.push(curr_sec)
                key_stack.push(key)
                curr_parent = curr_sec
                #key = '.'.join(map(str,curr_sec))
                data_dic[key] = {'main_content': key_value}
            else:
                if self.__isParent(curr_parent, curr_sec):
                    #key = self.toc.loc[row_idx, 'contents']
                    exec_cmd = 'data_dic'
                    for elem in key_stack.stack:
                        #exec_cmd += '["' + '.'.join(map(str,elem)) + '"]'
                        exec_cmd += '["' + elem + '"]'
                    #exec_cmd += '["' + '.'.join(map(str,curr_sec)) + '"] = {"main_content":"yoyo"}'
                    ##exec_cmd += '["' + key + '"] = {"main_content":"'+ key_value +'"}'
                    exec_cmd += '["' + key + '"] = {"main_content":key_value}'
                    #print(exec_cmd)
                    exec(exec_cmd)
                    sec_stack.push(curr_sec)
                    key_stack.push(key)
                    curr_parent = curr_sec
                else:
                    while not(sec_stack.isEmpty()) and not(self.__isParent(curr_parent, curr_sec)):
                        curr_parent = sec_stack.pop()
                        key_parent = key_stack.pop()
                    if self.__isParent(curr_parent, curr_sec):
                        sec_stack.push(curr_parent)
                        key_stack.push(key_parent)
                    if sec_stack.isEmpty():
                        sec_stack.push(curr_sec)
                        key_stack.push(key)
                        curr_parent = curr_sec
                        #key = '.'.join(map(str,curr_sec))
                        data_dic[key] = {'main_content': key_value}
                    else:
                        exec_cmd = 'data_dic'
                        for elem in key_stack.stack:
                            #exec_cmd += '["' + '.'.join(map(str,elem)) + '"]'
                            exec_cmd += '["' + elem + '"]'
                        #exec_cmd += '["' + '.'.join(map(str,curr_sec)) + '"] = {"main_content":"yoyo"}'
                        ##exec_cmd += '["' + key + '"] = {"main_content":"'+ key_value +'"}'
                        exec_cmd += '["' + key + '"] = {"main_content":key_value}'
                        exec(exec_cmd)
                        
                        sec_stack.push(curr_sec)
                        key_stack.push(key)
                        curr_parent = curr_sec
            row_idx+=1
        return(data_dic)
        
        
class extractData:
    def __init__(self):
        self.threshold_isSame = 85
        self.threshold_isSubPart = 98
        
    def __isSame(self, phrase1, phrase2):
        THRESHOLD = self.threshold_isSame
        phrase1 = re.sub('[^0-9a-zA-Z]+', ' ',str(phrase1)).lower().strip()
        phrase2 = re.sub('[^0-9a-zA-Z]+', ' ',str(phrase2)).lower().strip()
        return(True if(fuzz.token_sort_ratio(phrase1, phrase2)) >= THRESHOLD else False)
        
    def __isSubPart(self, phrase1, phrase2):
        '''Is phrase2 sub-part od phrase1?'''
        THRESHOLD = self.threshold_isSubPart
        phrase1 = re.sub('[^0-9a-zA-Z]+', ' ',str(phrase1)).lower().strip()
        phrase2 = re.sub('[^0-9a-zA-Z]+', ' ',str(phrase2)).lower().strip()
        return(True if(fuzz.partial_ratio(phrase1, phrase2)) >= THRESHOLD else False)
        
    def __StartWith(self, phrase1, phrase2):
        ''' Phrase1 starts with phrase2'''
        right_margin = 2
        same_flag = False
        phrase1 = list(map(str.strip, phrase1.split(' ')))
        phrase2 = list(map(str.strip, phrase2.split(' ')))
        if len(phrase2) > len(phrase1):
            same_flag = False
        else:
            while len(phrase1)<len(phrase2)+right_margin:
                right_margin-=1
            phrase1 = ' '.join(phrase1[:len(phrase2)+right_margin])
            phrase2 = ' '.join(phrase2)
            same_flag = self.__isSubPart(phrase1, phrase2)
        return(same_flag)
        
    def __StartWith_for_multiple(self, phrase1, phrase2_dict_list):
        ''' Phrase1 starts with phrase2 dict_list'''
        right_margin = 2
        same_flag = False
        phrase1 = list(map(str.strip, phrase1.split(' ')))
        found_flag = False
        matched_keyword = None
        dict_keys_ls = list(phrase2_dict_list.keys())
        for idx in range(len(dict_keys_ls)):
            temp_keyword_list = [dict_keys_ls[idx]] + phrase2_dict_list[dict_keys_ls[idx]]
            for phrase2 in temp_keyword_list:
                phrase2 = list(map(str.strip, phrase2.split(' ')))
                if len(phrase2) > len(phrase1):
                    same_flag = False
                else:
                    while len(phrase1)<len(phrase2)+right_margin:
                        right_margin-=1
                    temp_phrase1 = ' '.join(phrase1[:len(phrase2)+right_margin])
                    temp_phrase2 = ' '.join(phrase2)
                    same_flag = self.__isSubPart(temp_phrase1, temp_phrase2)
                if same_flag:
                    matched_keyword = dict_keys_ls[idx]
                    found_flag = True
                    break
            if found_flag:
                break
        return(matched_keyword)
               
    def __agg_data_from_dic(self, dic):
        values = [[value] if(isinstance(value, str)) else self.__agg_data_from_dic(value) for value in dic.values()]
        temp_ls = [res for value in values for res in value]
        return(temp_ls)
        
    def from_raw_text_dic(self, file_dir, file_name, toc, raw_text_structured, keyword_dic_list, alternative_keyword, sub_section_split_keyword_dict_list, default_keyword):
        delimiter = '|||'
        filename = file_name.rstrip('.pdf').strip()
        try:
            key_queue = newQueue()
            temp_key_list = list()
            extracted_dic = dict()
            extracted_dic['file_name'] = filename
            if isinstance(keyword_dic_list, dict):
                for key in keyword_dic_list.keys():
                    key_queue.push(key)
                while not(key_queue.isEmpty()):
                    parent_key = key_queue.pop()
                    temp_dic = keyword_dic_list
                    for sub_key in parent_key.split(delimiter):
                        temp_dic = temp_dic[sub_key]
                    if isinstance(temp_dic, dict):
                        for key in temp_dic.keys():
                            key_queue.push(parent_key+delimiter+key)
                    elif isinstance(temp_dic, list):
                        for key in temp_dic:
                            temp_key_list.append(parent_key+delimiter+key)
                    else:
                        raise('dict or list required')
                for key in temp_key_list:
                    temp_dic = raw_text_structured
                    temp_ls = key.split(delimiter)
                    for idx in range(len(temp_ls)):
                        if idx==len(temp_ls)-1:
                            extracted_dic[temp_ls[idx]] = ''
                            if len(temp_dic.keys())>1:
                                for original_key in temp_dic.keys():
                                    if self.__isSubPart(original_key, temp_ls[idx]):
                                        extracted_dic[temp_ls[idx]] += ' '+' '.join(self.__agg_data_from_dic(temp_dic[original_key]))
                        else:
                            temp_dic = temp_dic[temp_ls[idx]]
            elif isinstance(keyword_dic_list, list):
                temp_delimiter = ' | '
                for key in keyword_dic_list:
                    extracted_dic[key] = temp_delimiter.join(self.__agg_data_from_dic(raw_text_structured[key]))
            else:
                raise('dict or list required')
            extracted_dic['table_present'] = False
        except:
            pdf_file = os.path.join(file_dir, filename+'.pdf')
            temp_toc = toc.copy(deep=True)
            temp_toc['check_flag'] = temp_toc['contents'].apply(lambda x: 1 if(self.__isSame(x, alternative_keyword)) else 0)
            #print('h1')
            if temp_toc.loc[temp_toc['check_flag']==1].shape[0] > 0:
                start_page_no = int(temp_toc.loc[temp_toc['check_flag']==1].reset_index().iloc[0]['page_no'])
                #print('h2', start_page_no)
                start_sec_num_ls = list(map(int, temp_toc.loc[temp_toc['check_flag']==1].reset_index().iloc[0]['section'].split('.')[:-1]))
                #print('h3', start_sec_num_ls)
                next_sec_num_ls = [start_sec_num_ls[idx]+1 if(idx==len(start_sec_num_ls)-1) else start_sec_num_ls[idx] for idx in range(len(start_sec_num_ls))]
                #print('h4', next_sec_num_ls)
                next_sec_num = '.'.join(map(str, next_sec_num_ls))+'.'
                #print('h5', next_sec_num)
                if next_sec_num not in temp_toc['section'].tolist():
                    #print('h6')
                    found_flag = False
                    while not(found_flag):
                        try:
                            next_sec_num_ls = [next_sec_num_ls[idx]+1 if(idx==len(next_sec_num_ls[:-1])-1) else next_sec_num_ls[idx] for idx in range(len(next_sec_num_ls[:-1]))]
                            next_sec_num = '.'.join(map(str, next_sec_num_ls))+'.'
                            #print('h7', next_sec_num)
                        except:
                            #print('h8')
                            break
                        if next_sec_num in temp_toc['section'].tolist():
                            next_page_no = int(temp_toc.loc[temp_toc['section']==next_sec_num].reset_index().iloc[0]['page_no'])
                            found_flag = True
                            #print('h9', next_page_no)
                else:
                    next_page_no = int(temp_toc.loc[temp_toc['section']==next_sec_num].reset_index().iloc[0]['page_no'])
            
            page_no = ','.join(map(str, list(range(start_page_no, next_page_no+1))))
            #print(page_no)
            extracted_dic = self.from_table(filename, pdf_file, page_no, sub_section_split_keyword_dict_list, default_keyword)
        
        return(extracted_dic)
        
    def from_table(self, file_name, pdf_file, page_no, sub_section_split_keyword_dict_list, default_keyword):
        #default_keyword = sub_section_split_keyword_dict_list.keys()[0]
        extracted_dic = dict()
        extracted_dic['file_name'] = file_name
        for keyword in sub_section_split_keyword_dict_list.keys():
            extracted_dic[keyword] = ''
        table_list = camelot.read_pdf(pdf_file, pages=page_no)
        current_keyword = None
        continue_flag = False
        for table in table_list:
            temp_df = table.df.copy(deep=True)
            if temp_df.shape[1] < 2:
                continue
            for row_idx in range(temp_df.shape[0]):
                current_keyword = self.__StartWith_for_multiple(temp_df.iloc[row_idx,0], sub_section_split_keyword_dict_list)
                if current_keyword:
                    temp_keyword1 = current_keyword
                    continue_flag = True
                elif not(continue_flag):
                    current_keyword = default_keyword
                else:
                    current_keyword = temp_keyword1
                extracted_dic[current_keyword] += ' '+str(temp_df.iloc[row_idx,1]).strip()
        extracted_dic['table_present'] = True
        return(extracted_dic)
        
####################################    FUNCTIONS WITHOUT CLASS   ####################################

def impute_nan_section(df, column='section'):
    parent_section = None
    new_child_section_counter = 0
    #previous_section_none = False
    for row_idx in range(df.shape[0]):
        if df.loc[row_idx, column] == None:
            if row_idx==0:
                parent_section = '0.'
                df.loc[row_idx, column] = parent_section
            else:
                new_child_section_counter+=1
                df.loc[row_idx, column] = str(parent_section) + str(new_child_section_counter) + '.'        
        else:
            new_child_section_counter = 0
            parent_section = df.loc[row_idx, column]
    return(df)
    
    
def section_correction(df, column='section'):
    max_section_number = -1
    for row_idx in range(df.shape[0]):
        current_section_ls = df.loc[row_idx, column].split('.')
        current_parent_section = int(current_section_ls[0])
        if current_parent_section > max_section_number:
            max_section_number = current_parent_section
        elif current_parent_section < max_section_number:
            if len(current_section_ls)==2:
                max_section_number+=1
            current_section_ls[0] = str(max_section_number)
            df.loc[row_idx, column] = '.'.join(current_section_ls)
    return(df)
    
#########################################################################################################################################
#########################################################################################################################################
'''
    def __extract_from_df(self, start_page, end_page, start_title, next_title, isLastTitle=False):
        is_start_title_found = False
        is_next_title_found = False
        data_start_idx = -1
        data_end_idx = -1
        output_data = ''
        page_no = list(range(start_page, self.prot_df['page_no'].max()+1)) if(isLastTitle) else list(range(start_page, end_page+1))
        temp_df = self.prot_df.loc[(self.prot_df['page_no'].isin(page_no))&(self.prot_df['tag_type']=='Data')&(self.prot_df['is_bold']==True)].reset_index(drop=True)
        temp_df['checkFlag'] = 1
        row_idx = 0
        while row_idx < temp_df.shape[0]:
            if re.sub('[^A-Za-z]+', '', temp_df.loc[row_idx, 'tag_value']).lower()=='table' and row_idx+1<temp_df.shape[0] and temp_df.loc[row_idx, 'page_no']==temp_df.loc[row_idx+1, 'page_no']:
                temp_df.loc[row_idx, 'tag_value'] = temp_df.loc[row_idx, 'tag_value'] + ' ' + temp_df.loc[row_idx+1, 'tag_value']
                temp_df.loc[row_idx, 'index'] = temp_df.loc[row_idx+1, 'index']
                temp_df.loc[row_idx+1, 'checkFlag'] = 0
                row_idx+=2
            else:
                row_idx+=1
        temp_df = temp_df.loc[temp_df['checkFlag']==1].reset_index(drop=True)
        for row_idx in range(temp_df.shape[0]):
            if not(is_start_title_found) and (self.__isSame(start_title, temp_df.loc[row_idx, 'tag_value'])):
                data_start_idx = temp_df.loc[row_idx, 'index']+1
                is_start_title_found = True
            elif not(isLastTitle) and is_start_title_found and not(is_next_title_found) and (self.__isSame(next_title, temp_df.loc[row_idx, 'tag_value'])):
                data_end_idx = temp_df.loc[row_idx, 'index']-2
                is_next_title_found = True
                break
        if not(is_start_title_found):
            output_data = ''
        elif is_start_title_found and not(is_next_title_found):
            if isLastTitle:
                output_data = ' '.join(map(lambda x: str(x).strip(), self.prot_df.loc[(self.prot_df['index'].isin(list(range(data_start_idx, data_end_idx+1))))&(self.prot_df['tag_type']=='Data')]['tag_value'].tolist()))
            else:
                output_data = ''
        elif is_start_title_found and is_next_title_found:
            output_data = ' '.join(map(lambda x: str(x).strip(), self.prot_df.loc[(self.prot_df['index'].isin(list(range(data_start_idx, data_end_idx+1))))&(self.prot_df['tag_type']=='Data')]['tag_value'].tolist()))
        #print('')
        #print(start_page, end_page, start_title, next_title, isLastTitle)
        #print(is_start_title_found, is_next_title_found, data_start_idx, data_end_idx)
        return(output_data)
'''
#########################################################################################################################################
#########################################################################################################################################

if __name__=='__main__':
    PATH = r'C:/mihir/ADS/ZS/3_zs_BMS_SOAP_Ankush/Mihir/SOAP_pilot_code'
    data_dic = pickle.load(open(r'dataDic_v5.pickle', 'rb'))
    #prot_name = 'ca209742-revprot01'
    prot_name = 'ca012004-prot'
    temp = data_dic[prot_name]
    #prot_name = 'ca209742-prot'
    prot_df = data_dic[prot_name]['prot_df']
    toc = data_dic[prot_name]['toc']
    extraction_device = ContentExtractor(prot_df, toc)
    rud = extraction_device.unstructured_data_dic
    rsd = extraction_device.structured_data_dic