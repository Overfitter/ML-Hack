# -#- coding: utf-8 -#-
"""
Created on Mon Dec 16 14:06:32 2019

@author: jj18826
"""

# =============================================================================
# Import required libraries/utilities:
# =============================================================================

# General Utilities
import pandas as pd
import numpy as np
import os
import re
import itertools

# Nltk
from nltk import ngrams

# Pandas parameters
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
pd.set_option('display.max_colwidth', -1)

# Other utilities:
from utilities import *


class NGramMatch():
    def __init__(self, user_procedure_frequency_path, source_score_path, mapping_flag):
        """
        """
        super().__init__()
        self.USER_PROCEDURE_FREQUENCY_PATH = user_procedure_frequency_path
        self.source_SCORE_DATA_PATH = source_score_path
        self.mapping_flag = mapping_flag
    
    def __get_user_procedure_list(self, df, col_name):
        """
        Read procedure frequency dataframe and returns unique procedures list
        Parametes
        --------
        user_input_path: Dataframe
                       Procedure Frequency dataframe
            
        Returns
        -------
        dataframe:
            Return unique raw procedures list
        """
        ## unique procedure list entered by user
        user_procedure_list = list(filter(None, df[col_name].fillna("").unique()))
        
        return user_procedure_list
    
    def __union(self, list_1, list_2):
        """Taking out Union of two lists (list_1 & list_2"""
        return list(set().union(list_1, list_2))
    
    def __filter_cleaned_universe_data(self, data):
        """
        """
        # Creating id_universe Column: 
        data["id_universe"] = data.index
        
        ########---------------------- Filters on universe_procedures_data -------------------##############
        
        # Taking only source procedures which have P Score Data scores
        data.mapped_to_survey_question = data.mapped_to_survey_question.str.replace("Doesn't map to question segments", "")
        
        # Filter out Map Procedures data:
        data = data[["id_universe", "source_procedures"]]
        
        # Text cleaning on map_procedures_data & universe_procedures_data -
        data["source_procedures_cleaned"] = data["source_procedures"].apply(lambda x: clean_sentence(x))
        
        return data
        
    
    def __cleaned_map_data(self, data):
        """
        """
        # Creating id_map & id_universe Column: 
        data["id_map"] = data.index
    
        # Text cleaning on map_procedures_data: 
        data["procedures_cleaned"] = data["procedures"].apply(lambda x: clean_sentence(x))
        
        return data
    
        
    def __ngram_entities_dict(self, map_procedures_df, universe_procedures_df, map_entity_column, universe_entity_column, n_gram):
        
        """
        Purpose: Creating Dictionary of map_id/universe_id (key) & entities ngrams(values) 
                 for given value of n_gram (2, 3...)
        Input: source Procedures dataframe, Universe/CPT Procedures Dataframe, map_entity_column (medical/non-medical source proc)
               universe_entity_column (medical/non-medical), n_gram
        Output: Returns dictionary ngram_map_entity_dict & ngram_universe_entity_dict 
                (map_id/universe_id (key) & entities ngrams (values))
        """
        
        # Map procedure & ngram dictionary
        map_sentence_list = [x.split() for x in map_procedures_df[map_entity_column]]
        ngram_list = [list(ngrams(map_proc, n_gram)) for map_proc in map_sentence_list]
    
        ngram_list = ["|".join([",".join([str(y) for y in x]) for x in k]) for k in ngram_list]
    
        ngram_map_entity_dict = {k: v for k, v in zip(map_procedures_df['id_map'], ngram_list)}
        
        # Universe procedure & ngram dictionary
        map_sentence_list = [x.split() for x in universe_procedures_df[universe_entity_column]]
        ngram_list = [list(ngrams(map_proc, n_gram)) for map_proc in map_sentence_list]
    
        ngram_list = ["|".join([",".join([str(y) for y in x]) for x in k]) for k in ngram_list]
    
        ngram_universe_entity_dict = {k: v for k, v in zip(universe_procedures_df['id_universe'], ngram_list)}
        
        
        return ngram_map_entity_dict, ngram_universe_entity_dict
    
    def __matching(self, map_proc_word_dataframe, universe_proc_word_dataframe, map_proc_dictionary, 
                 universe_proc_dictionary):
        
        """
        Purpose: This function match the unigrams/bigrams/trigrams of map procedures data & universe procedures data 
                 and it is a subpart of exact_match_bw_dataframes function. It takes following inputs:
        Input: 
            1. map_proc_word_dataframe - Dictionary of map_id & source procedures unigram/bigram/trigram
            2. universe_proc_word_dataframe - Dictionary of universe_id & CPT Procedures unigram/bigram/trigram
            3. map_proc_dictionary - Dictionary of map_id & source Procedures
            4. universe_proc_dictionary - Dictionary of universe_id & CPT Procedures
            5. universe_code_dictionary - Dictionary of universe_id & CPT Codes
        
        Output: Returns - Matched Unigram/Bigram/Trigram Dataframes
        """
        
        # Simple processing on mapping & universe procedure data (pivot up)
        map_proc_word_dataframe.columns = ["doc_id", "word"]
        map_proc_word_dataframe["doc_id"] = map_proc_word_dataframe["doc_id"].astype(str)
        map_proc_word_dataframe["doc_id"] = "map_" + map_proc_word_dataframe["doc_id"]
    
        universe_proc_word_dataframe.columns = ["doc_id", "word"]
        universe_proc_word_dataframe["doc_id"] = universe_proc_word_dataframe["doc_id"].astype(str)
        universe_proc_word_dataframe["doc_id"] = "universe_" + universe_proc_word_dataframe["doc_id"]
    
        df_word_map = map_proc_word_dataframe.groupby(['word'])['doc_id'].apply(list).reset_index()
    
        df_word_universe = universe_proc_word_dataframe.groupby(['word'])['doc_id'].apply(list).reset_index()
    
        # Creating Unigram (key) & map_id (value) dictionary
        df_word_map_dict = {k: v for k, v in zip(df_word_map['word'], df_word_map['doc_id'])}
    
        # Creating Unigram (key) & universe_id (value) dictionary
        df_word_universe_dict = {k: v for k, v in zip(df_word_universe['word'], df_word_universe['doc_id'])}
    
        # Intializing empty list to store all the matched words between df_word_map_dict & df_word_universe_dict
        all_list = []
    
        for k, _map in df_word_map_dict.items():
            try:
                _universe = df_word_universe_dict[k]
                all_comb = list(itertools.product(_map, _universe))
                df = pd.DataFrame(all_comb, columns=["map_id", "universe_id"])
                df['word'] = k
                all_list.append(df)
            except Exception as e:
                continue
    
        # Concating all the dataframes which is stored in all_list
        if len(all_list) > 0:
            df = pd.concat(all_list, axis=0)
            df.reset_index(drop=True, inplace=True)
    
            # Calculating word count of unigrams (ngrams)
            dfwc = df.groupby(['map_id', 'universe_id']).agg(lambda x: "|".join(set(x))).reset_index()
            dfwc["word_count"] = dfwc["word"].apply(lambda x: len(x.split("|")))
            
            # Sorting on word_count column
            dfwc.sort_values("word_count", ascending=False, inplace = True)
        
            final_dfwc = dfwc.copy() 
        
            final_dfwc.sort_values(['map_id', 'word_count'], ascending=False, inplace = True)
        
            final_dfwc.columns = ["id_map", "id_universe", "word", "word_count"]
        
            # Taking out only map & universe index 
            final_dfwc["id_map"] = [int(x.split("_")[1]) for x in final_dfwc["id_map"]]
            final_dfwc["id_universe"] = [int(x.split("_")[1]) for x in final_dfwc["id_universe"]]
        
            final_dfwc["source_procedures"] = final_dfwc["id_map"].map(map_proc_dictionary)
            final_dfwc["concept_name"] = final_dfwc["id_universe"].map(universe_proc_dictionary)
        
            return final_dfwc
        else:
            return pd.DataFrame()
    
    def __exact_match_bw_dataframes(self, map_procedures_df, universe_procedures_df, 
                                  map_entity_column, universe_entity_column,
                                  map_proc_dictionary, universe_proc_dictionary, 
                                  map_entity_dictionary, universe_entity_dictionary, n_gram):
        """
        Purpose: This function provides the final matched - ngrams dataframe & It uses "__matching" function
                 which we defined earlier after calculating Jaccard scores
        Input: 
            1. map_proc_word_df - Dictionary of map_id & source procedures unigram/bigram/trigram
            2. universe_proc_word_df - Dictionary of universe_id & CPT Procedures unigram/bigram/trigram
            3. map_proc_dictionary - Dictionary of map_id & source Procedures
            4. universe_proc_dictionary - Dictionary of universe_id & CPT Procedures
            5. universe_code_dictionary - Dictionary of universe_id & CPT Codes
        
        Output: Returns - Final Matched Unigram/Bigram/Trigram Dataframes with Jaccard Scores
        """
        if n_gram ==1:
            
            # Creating mapping procedure data for unigrams (pivot down data) entities
            map_proc_word_df = pd.DataFrame([[row[1], x] for row in map_procedures_df[["id_map", map_entity_column]].itertuples() for x in row[2].split()],
                    columns = ["id_map", "source_procedures_cleaned_words"])
    
            # Creating universe procedure data for unigrams (pivot down data) entities
            universe_proc_word_df = pd.DataFrame([[row[1], x] for row in universe_procedures_df[["id_universe", universe_entity_column]].itertuples() for x in row[2].split()],
                        columns = ["id_universe", "concept_name_cleaned_words"])
    
            # Calling __matching function to get final_dfwc dataframe (__matching of ngrams)
                    
            final_dfwc = self.__matching(map_proc_word_df, universe_proc_word_df, map_proc_dictionary, 
                                  universe_proc_dictionary)
            
            if final_dfwc.shape[0] > 0:
            
                final_dfwc["source_unigram_entities"] = final_dfwc["id_map"].map(map_entity_dictionary)
        
                final_dfwc["universe_unigram_entities"] = final_dfwc["id_universe"].map(universe_entity_dictionary)
        
                # Take Union of unigram entities -
                final_dfwc["unigram_union"] = final_dfwc.apply(lambda row: "|".join(self.__union(row["source_unigram_entities"].split(" "), 
                                                  row["universe_unigram_entities"].split(" "))), axis = 1)
        
                # Calculating Jaccard score for unigrams
                final_dfwc["unigram_jaccard_score"] = final_dfwc.apply(lambda row: len(row["word"].split("|"))/len(row["unigram_union"].split("|")), axis = 1)
                            
            else:
                final_dfwc = pd.DataFrame()
            
        else:
            map_sentence_list = [x.split() for x in map_procedures_df[map_entity_column]]
            map_procedures_df['ngram_map'] = [list(ngrams(map_proc, n_gram)) for map_proc in map_sentence_list]
    
            ngram_map_entity_dictionary = {k: v for k, v in zip(map_procedures_df['id_map'], map_procedures_df['ngram_map'])}
                    
            map_proc_word_df = pd.DataFrame([[row["id_map"], ",".join(x)] for i, row in map_procedures_df.iterrows() for x in row['ngram_map']], columns= ["id_map", "words"])
            
            universe_sentence_list = [x.split() for x in universe_procedures_df[universe_entity_column]]
            universe_procedures_df['ngram_universe'] = [list(ngrams(universe_proc, n_gram)) for universe_proc in universe_sentence_list]
    
            ngram_universe_entity_dictionary = {k: v for k, v in zip(universe_procedures_df['id_universe'], universe_procedures_df['ngram_universe'])}
            
            universe_proc_word_df = pd.DataFrame([[row["id_universe"], ",".join(x)] for i, row in universe_procedures_df.iterrows() for x in row['ngram_universe']], columns= ["id_universe", "words"])
    
            # Calling __matching function to get final_dfwc dataframe (__matching of ngrams)
                    
            final_dfwc = self.__matching(map_proc_word_df, universe_proc_word_df, map_proc_dictionary, 
                                   universe_proc_dictionary)
            
            if final_dfwc.shape[0] > 0:
            
                final_dfwc["source_ngram_entities"] = final_dfwc["id_map"].map(ngram_map_entity_dictionary)
        
                final_dfwc["universe_ngram_entities"] = final_dfwc["id_universe"].map(ngram_universe_entity_dictionary)
        
                final_dfwc['source_ngram_entities'] = ["|".join([",".join([str(y) for y in x]) for x in k]) for k in final_dfwc['source_ngram_entities']]
                final_dfwc['universe_ngram_entities'] = ["|".join([",".join([str(y) for y in x]) for x in k]) for k in final_dfwc['universe_ngram_entities']]
        
                # Take Union of N-gram entities -
                final_dfwc["ngram_union"] = final_dfwc.apply(lambda row: "|".join(self.__union(row["source_ngram_entities"].split("|"), 
                                                  row["universe_ngram_entities"].split("|"))), axis = 1)
        
                # Calculating Jaccard score for N-grams
                final_dfwc["ngram_jaccard_score"] = final_dfwc.apply(lambda row: len(row["word"].split("|"))/len(row["ngram_union"].split("|")),axis = 1)
                            
            else:
                final_dfwc = pd.DataFrame()
        
        return final_dfwc        
    
    def __create_output_data(self, unigram_df, bigram_df, trigram_df):
        """
        """
        if unigram_df.shape[0] > 0:
            ## Change column names for unigram, bigram & trigram dataframe
            unigram_df.columns = ['id_map', 'id_universe', 'unigram', 'unigram_count', 'procedures',
                               'source_procedures', 'janssen_unigram_entities',
                               'universe_unigram_entities', 'unigram_union',
                               'unigram_jaccard_score']
        else:
            unigram_df = pd.DataFrame(columns = ['id_map', 'id_universe', 'unigram', 'unigram_count', 'procedures',
                               'source_procedures', 'janssen_unigram_entities',
                               'universe_unigram_entities', 'unigram_union',
                               'unigram_jaccard_score'])
                
        if bigram_df.shape[0] > 0:
            bigram_df.columns = ['id_map', 'id_universe', 'bigram', 'bigram_count', 'procedures',
                               'source_procedures', 'janssen_bigram_entities',
                               'universe_bigram_entities', 'bigram_union',
                               'bigram_jaccard_score']
        else:
            bigram_df = pd.DataFrame(columns = ['id_map', 'id_universe', 'bigram', 'bigram_count', 'procedures',
                               'source_procedures', 'janssen_bigram_entities',
                               'universe_bigram_entities', 'bigram_union',
                               'bigram_jaccard_score'])
                
        if trigram_df.shape[0] > 0:
            trigram_df.columns = ['id_map', 'id_universe', 'trigram', 'trigram_count', 'procedures',
                               'source_procedures', 'janssen_trigram_entities',
                               'universe_trigram_entities', 'trigram_union',
                               'trigram_jaccard_score']
        else:
            trigram_df = pd.DataFrame(columns = ['id_map', 'id_universe', 'trigram', 'trigram_count', 'procedures',
                               'source_procedures', 'janssen_trigram_entities',
                               'universe_trigram_entities', 'trigram_union',
                               'trigram_jaccard_score'])
                
        merge_df_list = [trigram_df, bigram_df, unigram_df]
        df_merge = merge_df_list[0]
        for _df in merge_df_list[1:]:
            df_merge = df_merge.merge(_df, how='right', on=["id_map", "procedures", "id_universe", "source_procedures"])
            
        # Change the column sequence
        df_merge = df_merge[['id_map', 'procedures', 'id_universe', 'source_procedures', 
                           'trigram', 'trigram_count',
                           'janssen_trigram_entities',
                           'universe_trigram_entities', 'trigram_union',
                           'bigram', 'bigram_count',
                           'janssen_bigram_entities', 'universe_bigram_entities',
                           'bigram_union', 'unigram', 'unigram_count',
                           'janssen_unigram_entities', 'universe_unigram_entities',
                           'unigram_union', 'trigram_jaccard_score', 'bigram_jaccard_score', 'unigram_jaccard_score']]
        
        ## Taking top 3 recommendations:
        final_df = df_merge.sort_values(["id_map", 'trigram_jaccard_score', 'bigram_jaccard_score', 'unigram_jaccard_score'], ascending=False).groupby("id_map").head(3)
        
        # Filling NA with 0
        final_df.fillna(0, inplace = True)
        
        return final_df
    
    def get_source_procedures_mapping(self):
        """
        """
        
        print("#######------------- Loading Unique Procedures from User Input Config File-----------------#########")
        proc_frequency_df = pd.read_csv(self.USER_PROCEDURE_FREQUENCY_PATH, 
                                         index_col = None,
                                         encoding = "ISO-8859-1").drop_duplicates()

        proc_frequency_df.columns = proc_frequency_df.columns.str.lower().str.replace(" ","_")        
        
        # Create map_procedures_data dataframe from user input procedure - frequency data (proc_frequency_df):
        map_procedures_data = pd.DataFrame(self.__get_user_procedure_list(proc_frequency_df, "study_procedures"), columns = ["procedures"])
        
        map_procedures_data["procedures"] = map_procedures_data["procedures"].str.lower().str.strip()

        # Load source procedures data:
        universe_procedures_data = pd.read_csv(self.source_score_DATA_PATH, index_col = None, encoding = "ISO-8859-1").fillna("").drop_duplicates("source_procedures")
        
        universe_procedures_data.columns = universe_procedures_data.columns.str.lower()

        if self.mapping_flag=="p_score":
            ## P Score Data Data
            universe_procedures_data = universe_procedures_data[~universe_procedures_data['p_score'].isin(["", " "])].reset_index(drop=True)
        else:
            ## S Score Data Data
            universe_procedures_data = universe_procedures_data[~universe_procedures_data['s_score'].isin(["", " "])].reset_index(drop=True)
        
        print("#######------------- Checking the shape of map & universe data-----------------#########")
        print("Map Procedures Data Shape (User Input Procedures) : ", map_procedures_data.shape)
        print("Universe Procedures Data Shape (source score - Procedures) : ", universe_procedures_data.shape)
        
        # Get Cleaned & Filtered Universe/map data:
        map_procedures_data = self.__cleaned_map_data(map_procedures_data)
        universe_procedures_data = self.__filter_cleaned_universe_data(universe_procedures_data)
        
        ############------------------------Creating various dictionary----------------------###############
        
        # Mapping id to procedure dictionary
        map_proc_dict = {k: v for k, v in zip(map_procedures_data['id_map'], map_procedures_data['procedures'])}
        
        # Universe id to procedure dictionary
        universe_proc_dict = {k: v for k, v in zip(universe_procedures_data['id_universe'], universe_procedures_data['source_procedures'])}
        
        # Mapping id to all entities dictionary
        map_entity_dict = {k: v for k, v in zip(map_procedures_data['id_map'], map_procedures_data['procedures_cleaned'])}
        
        # Universe id to all entities dictionary
        universe_entity_dict = {k: v for k, v in zip(universe_procedures_data['id_universe'], universe_procedures_data['source_procedures_cleaned'])}
        
        
        #### Ngrams & entities dictionary preparation ####
        # Pass medical entities columns if you want to match only medical entities
        bigram_map_entity_dict,  bigram_universe_entity_dict = self.__ngram_entities_dict(map_procedures_data,
                                                                                    universe_procedures_data,
                                                                                    "procedures_cleaned", 
                                                                                    "source_procedures_cleaned", 
                                                                                    2)
        
        trigram_map_entity_dict, trigram_universe_entity_dict = self.__ngram_entities_dict(map_procedures_data,
                                                                                    universe_procedures_data, 
                                                                                    "procedures_cleaned", 
                                                                                    "source_procedures_cleaned",
                                                                                    3)
        
        print("######------------------ Creating Unigram Matches ----------------#######")
        # Create Unigram matches dataframe
        unigram_df = self.__exact_match_bw_dataframes(map_procedures_data, universe_procedures_data, 
                                               "procedures_cleaned", "source_procedures_cleaned",
                                              map_proc_dict, universe_proc_dict,  
                                               map_entity_dict, universe_entity_dict, 1)
        
        print("######------------------ Creating Bigram Matches ----------------#######")
        # Create Bigram matches dataframe
        bigram_df = self.__exact_match_bw_dataframes(map_procedures_data, universe_procedures_data, 
                                              "procedures_cleaned", "source_procedures_cleaned",
                                              map_proc_dict, universe_proc_dict,
                                              map_entity_dict, universe_entity_dict, 2)
        
        print("######------------------ Creating Trigram Matches ----------------#######")
        # Create Trigram matches dataframe
        trigram_df = self.__exact_match_bw_dataframes(map_procedures_data, universe_procedures_data, 
                                               "procedures_cleaned", "source_procedures_cleaned",
                                               map_proc_dict, universe_proc_dict, 
                                               map_entity_dict, universe_entity_dict, 3)
        
        print("######------------------ Creating Output Data ----------------#######")
        output_df = self.__create_output_data(unigram_df, bigram_df, trigram_df)
        
         # Including trigram & bigram  entities - 
        output_df["janssen_trigram_entities"] = output_df["id_map"].map(trigram_map_entity_dict)
        output_df["universe_trigram_entities"] = output_df["id_universe"].map(trigram_universe_entity_dict)
        
        output_df["janssen_bigram_entities"] = output_df["id_map"].map(bigram_map_entity_dict)
        output_df["universe_bigram_entities"] = output_df["id_universe"].map(bigram_universe_entity_dict)

        # Get ranking
        output_df["ranking"] = 1

        output_df["ranking"] = output_df.groupby(["id_map"])["ranking"].cumsum().values.tolist()
        
        return output_df

#if __name__ == '__main__':
#    
#    
#    ngram_cls = NGramMatch(user_procedure_frequency_path, source_score_path)
#    
#    mapping_df = ngram_cls.get_source_procedures_mapping()
#    
#    
    
    
