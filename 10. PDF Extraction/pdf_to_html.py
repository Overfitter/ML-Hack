
from utilities_v1 import PDF_converter, TOC_extract
import pandas as pd
from utilities_v1 import HTML_to_DF_converter
import utilities_v1 as util
import re
import codecs
import os

def pdf_to_html_function(raw_prot, RAW_DATA_PATH, HTML_PATH):
    try:
        # protocol name
        prot_name = raw_prot.rstrip('.pdf').rstrip('.PDF').strip()
        print(prot_name)
        # converting PDF -> HTML
        trial_html = PDF_converter(pdf_file_path=os.path.join(RAW_DATA_PATH, raw_prot),
                                   save_converted_file=True,
                                   save_converted_file_dir=HTML_PATH).converted_output
    except:
        print('error while converting the HTML files')

def html_to_text(raw_prot, HTML_PATH, TEXT_PATH):

    # protocol name
    prot_name = raw_prot.rstrip('.pdf').rstrip('.PDF').strip()
    print(prot_name)

    # initialize HTML -> DataFrame class
    html2df_converter_device = HTML_to_DF_converter()
    # HTML -> txt
    html2df_converter_device.parse_html(codecs.open(os.path.join(HTML_PATH, prot_name + '.html')).read(),
                                        os.path.join(TEXT_PATH, prot_name + '.txt'))


def text_to_dataframe(raw_prot, TEXT_PATH, prot_df_PATH):
    prot_name = raw_prot.rstrip('.pdf').rstrip('.PDF').strip()
    print(prot_name)
    html2df_converter_device = HTML_to_DF_converter()
    prot_df = html2df_converter_device.create_protocol_dataframe(os.path.join(TEXT_PATH, prot_name + '.txt'))
    # save prot_df
    prot_df.to_csv(os.path.join(prot_df_PATH, 'prot_df_' + prot_name + '.csv'), index=False)


def pdf_to_toc(raw_prot, RAW_DATA_PATH, toc_df_PATH):
    try:
        print(raw_prot)
        print(RAW_DATA_PATH)
        prot_name = raw_prot.rstrip('.pdf').rstrip('.PDF').strip()
        print(prot_name)
        toc = TOC_extract(os.path.join(RAW_DATA_PATH, raw_prot)).toc
        if toc is None:
            data=pd.DataFrame()
            data.to_csv(os.path.join(toc_df_PATH, 'toc_df_' + prot_name + '.csv'), index=False)
            return
        toc['section'] = toc['section'].apply(lambda x: re.sub(r'\.*$', '', x.strip()) + '.' if (x) else None)
        toc['contents'] = toc['contents'].apply(lambda x: x.strip())
        toc = util.impute_nan_section(toc)
        toc = util.section_correction(toc)
        toc['contents'] = toc['contents'].apply(lambda x: x.replace('"', "'"))
        # saving toc
        toc.to_csv(os.path.join(toc_df_PATH, 'toc_df_' + prot_name + '.csv'), index=False)
    except:
        data = pd.DataFrame()
        data.to_csv(os.path.join(toc_df_PATH, 'toc_df_' + prot_name + '.csv'), index=False)
        return



def extract_pdf_file(raw_prot, RAW_DATA_PATH, HTML_PATH, TEXT_PATH, prot_df_PATH, toc_df_PATH):
    # error
    error = None
    try:
        # protocol name
        prot_name = raw_prot.rstrip('.pdf').rstrip('.PDF').strip()
        print(prot_name)
        # converting PDF -> HTML
        trial_html = PDF_converter(pdf_file_path=os.path.join(RAW_DATA_PATH, raw_prot),
                                   save_converted_file=True,
                                   save_converted_file_dir=HTML_PATH).converted_output
        # initialize HTML -> DataFrame class
        html2df_converter_device = HTML_to_DF_converter()
        # HTML -> txt
        html2df_converter_device.parse_html(codecs.open(os.path.join(HTML_PATH, prot_name + '.html')).read(),
                                            os.path.join(TEXT_PATH, prot_name + '.txt'))
        # txt -> DataFrame
        prot_df = html2df_converter_device.create_protocol_dataframe(os.path.join(TEXT_PATH, prot_name + '.txt'))
        # save prot_df
        prot_df.to_csv(os.path.join(prot_df_PATH, 'prot_df_' + prot_name + '.csv'), index=False)
    except Exception as e:
        # error message
        prot_df = None
        print(e)
        error = 'Error: PDF upload/conversion Fail'
        print(e)

    if not (error):
        if 'data' not in [tag.lower().strip() for tag in prot_df['tag_type'].unique()]:
            error = 'Error: Scanned PDF'
    toc = None
    if not (error):
        # extract TOC (table of content)
        toc = TOC_extract(os.path.join(RAW_DATA_PATH, raw_prot)).toc
        try:
            toc['section'] = toc['section'].apply(lambda x: re.sub(r'\.*$', '', x.strip()) + '.' if (x) else None)
            toc['contents'] = toc['contents'].apply(lambda x: x.strip())
            toc = util.impute_nan_section(toc)
            toc = util.section_correction(toc)
            toc['contents'] = toc['contents'].apply(lambda x: x.replace('"', "'"))
            # saving toc
            toc.to_csv(os.path.join(toc_df_PATH, 'toc_df_' + prot_name + '.csv'), index=False)
        except:
            toc = None
            # error = 'Error: TOC Missing'
            error = 'Error: TOC Extraction Fail'
    # return toc and prot_df
    return prot_df, toc, error


