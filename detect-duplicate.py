# -*- coding: utf-8 -*-

import pandas as pd
from preprocess import processTextrow, processSize, divideSubCats, flatten, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import json


def detectduplicate(r1, r2, threshold=100):

    content1 = str(r1['title']) + str(r1['detailedSpecsStr'])
    content2 = str(r2['title']) + str(r2['detailedSpecsStr'])
    vectorizer = TfidfVectorizer(
        tokenizer=normalize, stop_words='english', vocabulary=None)

    tfidf = vectorizer.fit_transform([content1, content2])
    confidence = ((tfidf * tfidf.T).A)[0, 1] * 100

    if confidence >= threshold:
        return True, confidence
    else:
        return False, confidence


if __name__ == "__main__":

    data = pd.read_csv('datafile.csv')
    data = data.loc[:, ['productId', 'title', 'imageUrlStr', 'productUrl', 'productBrand',
                        'size', 'color', 'detailedSpecsStr', 'sellerName', 'sleeve', 'neck']]

    data = processSize(data)

    data = divideSubCats(data, 'sleeve')
    subframes = {}
    for i, key in enumerate(data.keys()):
        subframes[i] = divideSubCats(data[key], 'neck')

    subframes = flatten(subframes)
    df = {}
    for i, key in enumerate(subframes.keys()):
        df[i] = subframes[key]
    subframes = df
    del df
    
    sf={}
    for i, key in enumerate(subframes.keys()):
            sf[i] = divideSubCats(subframes[key], 'size')
    sf = flatten(sf)
    df = {}
    for i, key in enumerate(sf.keys()):
        df[i] = sf[key]
    sf = df
    del df
    subframes = sf
    del sf
    
    
    duplicates = defaultdict(list)
    
    for key in subframes.keys():
        df = subframes[key]
        l = len(df) 
        i = 0
        while(l > 1):
            item = df.iloc[i]
            j = 0
            while(j < l):
                if item['productId'] != df.iloc[j]['productId']:
                    result, _ = detectduplicate(item, df.iloc[j])
                    if result == True:
                        duplicates[item['productId']].append(df.iloc[j]['productId'])
                        df = df[df.productId != df.iloc[j]['productId']]
                        l = l-1
                        continue
                j = j+1
            df = df[df.productId != item['productId']]
            l = l-1
    with open('duplicate50000.txt', 'w') as file:
        file.write(json.dumps(duplicates, indent=2))
