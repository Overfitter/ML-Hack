import nltk, os, subprocess, code, glob, re, traceback, sys, inspect
from io import BytesIO

def is_english_word(word,english_words):
    return word.lower() in english_words

def preprocess(document):
    try:
    # Try to get rid of special characters
        try:
            document = document.decode('ascii', 'ignore')
        except:
            document = document.encode('ascii', 'ignore')
            
        document = re.sub('[^a-zA-Z0-9\n]', ' ', document)
        lines = [el.strip() for el in document.split("\n") if len(el) > 0]  # Splitting on the basis of newlines 
        lines = [nltk.word_tokenize(el) for el in lines]            # Tokenize the individual lines
        lines = [nltk.pos_tag(el) for el in lines]  # Tag them
        

        sentences = nltk.sent_tokenize(document)    # Split/Tokenize into sentences (List of strings)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]    # Split/Tokenize sentences into words (List of lists of strings)
        tokens = sentences
        sentences = [nltk.pos_tag(sent) for sent in sentences]    # Tag the tokens - list of lists of tuples - each tuple is (<word>, <tag>)
        
        dummy = []
        for el in tokens:
            dummy += el
        tokens = dummy
        
        # tokens - words extracted from the doc, lines - split only based on newlines (may have more than one sentence)
        # sentences - split on the basis of rules of grammar
        
        return tokens, lines, sentences
    except Exception as e:
        print(e)

def getName(text_string,indianNames,english_words):
    if text_string == "Fail":
        return "Fail"
    otherNameHits = []
    nameHits = []
    name = None
    try:
        # tokens, lines, sentences = self.preprocess(text_string)
#         print(text_string)
        tokens, lines, sentences = preprocess(text_string)
        # Try a regex chunk parser
        # grammar = r'NAME: {<NN.*><NN.*>|<NN.*><NN.*><NN.*>}'
        grammar = r'NAME: {<JJ|NN.*><NN.*><NN.*>*}'
        # Noun phrase chunk is made out of two or three tags of type NN. (ie NN, NNP etc.) - typical of a name. {2,3} won't work, hence the syntax
        # Note the correction to the rule. Change has been made later.
        chunkParser = nltk.RegexpParser(grammar)
        all_chunked_tokens = []
        j=0
        for tagged_tokens in lines:
            # Creates a parse tree
            if len(tagged_tokens) == 0: continue # Prevent it from printing warnings
            chunked_tokens = chunkParser.parse(tagged_tokens)
            all_chunked_tokens.append(chunked_tokens)
            for subtree in chunked_tokens.subtrees():
                #  or subtree.label() == 'S' include in if condition if required
                # print(subtree.leaves(),subtree.label())
                if subtree.label() == 'NAME':
#                     print(subtree.leaves(),subtree.label())
                    s = sum([1 if leaf[1] not in ['NN','NNP','NNS','JJ'] else 0 for ind, leaf in enumerate(subtree.leaves())])
                    if s>0:
                        break
                    else:
                        a = sum([1 if is_english_word(leaf[0].lower(),english_words) else 0 for ind, leaf in enumerate(subtree.leaves())])
                        b = sum([1 if leaf[1] not in ['NN','NNP','NNS','JJ'] else 0 for ind, leaf in enumerate(subtree.leaves())])
#                         print(a,b)
                        if a>0 or b>0:
                            break
                        else:
#                             print(subtree.leaves())
                            if sum([1 if leaf[0].lower() in indianNames else 0 for ind, leaf in enumerate(subtree.leaves())])>0:
#                                 print(subtree.leaves())
                                for ind, leaf in enumerate(subtree.leaves()):
    #                                 print(leaf)
            #                         if leaf[0].lower() in indianNames and 'NN' in leaf[1]:
#                                     if 'NN' in leaf[1]:
        #                                     print(leaf)
        #                             print(subtree.leaves()[ind:ind+3])
                                    # Case insensitive matching, as indianNames have names in lowercase
                                    # Take only noun-tagged tokens
                                    # Surname is not in the name list, hence if match is achieved add all noun-type tokens
                                    # Pick upto 3 noun entities
                                    hit = " ".join([el[0] for el in subtree.leaves()[ind:ind+3]])
                                    # Check for the presence of commas, colons, digits - usually markers of non-named entities
        #                             if re.compile(r'[\d,:]').search(hit): continue
                                    nameHits.append(hit)
                                    # Need to iterate through rest of the leaves because of possible mis-matches
                j+=1
                if j>20:
                    break
        if j>20 and len(nameHits)==0:
            all_chunked_tokens = []
            for tagged_tokens in lines:
                # Creates a parse tree
                if len(tagged_tokens) == 0: continue # Prevent it from printing warnings
                chunked_tokens = chunkParser.parse(tagged_tokens)
                all_chunked_tokens.append(chunked_tokens)
                for subtree in chunked_tokens.subtrees():
                    #  or subtree.label() == 'S' include in if condition if required
#                     print(subtree.leaves())
                    if subtree.label() == 'NAME':
#                         print(subtree.leaves())
                        s = sum([1 if leaf[1] not in ['NN','NNP','NNS','JJ'] else 0 for ind, leaf in enumerate(subtree.leaves())])
                        if s>0:
                            break
                        else:
                            a = sum([1 if is_english_word(leaf[0].lower(),english_words) else 0 for ind, leaf in enumerate(subtree.leaves())])
                            b = sum([1 if leaf[1] not in ['NN','NNP','NNS','JJ'] else 0 for ind, leaf in enumerate(subtree.leaves())])
                            
                            if a>0 or b>0:
                                break
                            else:
#                                 print(subtree.leaves())
                                for ind, leaf in enumerate(subtree.leaves()):
#                                     print(leaf)
            #                         if leaf[0].lower() in indianNames and 'NN' in leaf[1]:
                                    if 'NNP' in leaf[1]:
            #                                     print(leaf)
            #                             print(subtree.leaves()[ind:ind+3])

                                        # Pick upto 3 noun entities
                                        hit = " ".join([el[0] for el in subtree.leaves()[ind:ind+3]])
                                        # Check for the presence of commas, colons, digits - usually markers of non-named entities
            #                             if re.compile(r'[\d,:]').search(hit): continue
                                        nameHits.append(hit)
                                        # Need to iterate through rest of the leaves because of possible mis-matches
                
                
        # Going for the first name hit
        if len(nameHits) > 0:
            nameHits = [re.sub(r'[^a-zA-Z \-]', '', el).strip() for el in nameHits]
#             print(nameHits)
            name = " ".join([el[0].upper()+el[1:].lower() for el in nameHits[0].split() if len(el)>0])
            otherNameHits = nameHits[1:]

    except Exception as e:
        print(traceback.format_exc())
        print(e)
    return name
#     return name, otherNameHits




