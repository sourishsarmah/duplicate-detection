# -*- coding: utf-8 -*-
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = str(parent_key) + sep + str(k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


print('Downloading NLP resources . . .')
nltk.download('punkt')


def stem_tokens(tokens):
    """
    Trimms the tokens to their root.
    """
    stemmer = nltk.stem.porter.PorterStemmer()
    return [stemmer.stem(item) for item in tokens]


def normalize(text):
    """
    Remove punctuation, converts all words to lowercase and stems the words.  
    """
    removePunctuation = dict((ord(char), None) for char in string.punctuation)
    return stem_tokens(nltk.word_tokenize(text.lower().translate(removePunctuation)))


def preprocessText(text) :
    """
    Tokenize words.  
    """
    text = ''.join(i for i in text if not i.isdigit())
    tokens = normalize(text)
    return tokens 

def processTextrow(data, *args):
    for col in args:
        row = data[str(col)]
        data[str(col)] = []
        if type(row) is not float:
            tokens = preprocessText(row)
            data[str(col)] = list(tokens)
    return data

def processSize(data): 
    for i, row in enumerate(data['size']):
        row = str(row)
        if row.isdigit():
            if row == '6' or row == '32':
                row = 'XS'
            elif row == '8' or row == '10' or row == '34' or row == '36':
                row = 'S'
            elif row == '12' or row == '14' or row == '38' or row == '40':
                row = 'M'
            elif row == '16' or row == '18' or row == '42' or row == '44':
                row = 'L'
            elif row == '20' or row == '22' or row == '46' or row == '48':
                row = 'XL'
            elif row == '24' or row == '50':
                row = '2XL'
            elif row == '26' or row == '52':
                row = '3XL'
            elif row == '28' or row == '54':
                row = '4XL'
            
            data['size'][i] = row
    return data
        

def divideSubCats(data, col):
    subcats = list(data[str(col)].unique())
    df = {}
    for i, subcat in enumerate(subcats):
        if type(subcat) is not float:
            df[i] = data[data[col] == subcat]
            df[i].drop(columns=str(col), inplace=True)
    return df
