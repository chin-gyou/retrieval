#!/usr/bin/python
# -*- coding: utf-8 -*-
import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize

from whoosh.index import create_in,open_dir
from whoosh.fields import *
from whoosh.qparser import *
from whoosh.scoring import *
import os
stop_words = stopwords.words('english')
def build_index(file, index_dir="database"):
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    schema = Schema(context=TEXT(stored=True), reply=TEXT(stored=True))
    ix = create_in(index_dir, schema)
    writer = ix.writer()
    lines = open(file).readlines()
    for i,line in enumerate(lines):
        if i%1000==0:
            print(i)
        turns = [u'']+[u'']+line.decode('utf-8').split('__eou__')
        for j in range(len(turns)-3):
            writer.add_document(context=turns[j].strip()+' '+turns[j+1].strip()+' '+turns[j+2].strip(),reply=turns[j+3])
    writer.commit()

# scoring functions for retieving the first-round candidates
class ScoreModel(WeightingModel):
    def scorer(self, searcher, fieldname, text, qf=1):
        #maxweight = searcher.term_info(fieldname, text).max_weight()
        return MyScorer(searcher, fieldname, text)

    def supports_block_quality(self):
        return True

class MyScorer(WeightLengthScorer):
    def __init__(self, searcher, fieldname, text):
        parent = searcher.get_parent()
        self.idf = parent.idf(fieldname, text)
        self.setup(searcher, fieldname, text)

    def _score(self, weight, length):
        return weight*self.idf/(length+1.0)

def search(contexts,score_model,index_dir="database",field="context",num_res=1000):
    ix = open_dir(index_dir)
    qp = QueryParser(field, schema=ix.schema,group=OrGroup)
    q = qp.parse(contexts)
    with ix.searcher(weighting=score_model) as s:
        results = s.search(q,limit=num_res)
        return [(results[i]['context'],results[i]['reply']) for i in range(results.scored_length())]

# scoring function for retrieval
# tf*idf/doc_length
def scorer(searcher,fieldname,text,matcher):
    parent = searcher.get_parent()
    idf = parent.idf(fieldname,text)
    doc_len = searcher.doc_field_length(matcher.id(),fieldname,1)
    tf = matcher.weight()
    return idf*tf/(doc_len+1.0)

# fuzzy ratio
def fuzzy_ratio(c1, c2):
    return fuzz.QRatio(c1, c2), fuzz.WRatio(c1, c2), fuzz.partial_ratio(c1, c2)

def cosine_distance(c1, c2):
    return cosine(sent2vec(c1), sent2vec(c2))    

def jaccard_distance(c1, c2):
    return jaccard(sent2vec(c1), sent2vec(c2))

def euclidean_distance(c1, c2):
    return euclidean(sent2vec(c1), sent2vec(c2))

# sentence to vector
def sent2vec(s):
    words = s.split()
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return  np.nan_to_num(v / np.sqrt((v ** 2).sum()))

# second-round reranking
def rerank_context(context, q_r_pairs, score_func):
    ranked = sorted(q_r_pairs, key = lambda x: score_func(x[0], context), reverse = True)
    return ranked

# second-round scoring function
def score_second(c1, c2):
    #return sum(fuzzy_ratio(c1, c2))
    return cosine_distance(c1, c2) + jaccard_distance(c1, c2) + euclidean_distance(c1, c2)

# return the final results
def rank(context, index_dir = 'database', limit=10):
    first_round_r = search(context, ScoreModel(), index_dir)
    second_round_r = rerank_context(context, first_round_r, score_second)
    return second_round_r[:limit]

if __name__ == "__main__":
    #build_index("data/train.txt")
    results=rank("Would you mind waiting a while ? Well , how long will it be ? I'm not sure . But I'll get a table ready as fast as I can .")
    #results=search("The kitchen stinks . right ?",scorer,num_res=5)
    print(results)
    #print(results[0].encode('utf-8'))
