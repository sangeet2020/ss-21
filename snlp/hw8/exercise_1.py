from collections import Counter
from pathlib import Path
import nltk
from nltk import RegexpTokenizer
# nltk.download('reuters')
# nltk.download('stopwords')
from nltk.corpus import reuters, stopwords

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

import operator
import re
import math
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
plt.rcParams["figure.figsize"] = (20,5)
import pdb

def plot_category_frequencies(category_frequencies: Counter):
    
    fig, ax = plt.subplots()
    frq_list = list(category_frequencies.values())
    cat_list = list(category_frequencies.keys())
    cat_pos = np.arange(len(cat_list))
    ax.loglog(cat_pos, frq_list)
    ax.set_title("Frequency Curve with log scaling")
    ax.set_xlabel("category")
    ax.set_ylabel("log(Frequency)")
    plt.grid()
    


def plot_pmis(category: str, most_common: List[str], pmis: List[float]):
    fig, ax = plt.subplots()
    # pos = np.arange(len(most_common))
    ax.bar(most_common, pmis)
    ax.set_title("PMI for the category: " + category)
    ax.set_xlabel("most common wor  ds")
    ax.set_ylabel("PMI")
    # plt.xticks(rotation=60, ha='right')
    # plt.xticks(pos, most_common)
    plt.grid()


def plot_dfs(terms: List[str], dfs: List[int]):
    fig, ax = plt.subplots()
    # pos = np.arange(len(terms))
    ax.bar(terms, dfs)
    ax.set_title("Document frequency of top 10 words")
    ax.set_xlabel("words")
    ax.set_ylabel("DF")
    # plt.xticks(rotation=60, ha='right')
    # plt.xticks(pos, terms)
    plt.grid()


class Document:
    def __init__(self, id:str, text: str, categories: str, stop_words: List[str]):
        self.id = id
        # assume only 1 category per document for simplicity
        self.category = categories[0]
        self.stop_words = stop_words
        self.tokens = self.preprocessing(text)

    def preprocessing(self, text):
        text = text.lower()
        text = re.sub(r'[0-9]', '', text)
        tokenizer = RegexpTokenizer("[\w']+")
        tokens = tokenizer.tokenize(text)
        return [word for word in tokens if not word in self.stop_words]



class Corpus:
    def __init__(self, documents: List[Document], categories: List[str]):
        self.documents = documents
        self.categories = set(categories)

    def df(self, term: str, category=None) -> int:
        """
        :param category: None if df is calculated over all categories, 
        else one of the reuters.categories
        """
        
        freq_dict = self.category_frequencies()
        
        if category:
            counter = self.term_frequencies(category)
            df = dict(counter)[term] / freq_dict[category]
        else:
            counter = self.term_frequencies()
            df = dict(counter)[term] / sum(freq_dict.values())
        
        return df
    
    def pmi(self, category: str, terms: List[str]) -> float:
        all_tokens_freq = self.term_frequencies()
        tokens = list(all_tokens_freq.keys())
        N = sum(all_tokens_freq.values())     # vocab size of housing
        categories = self.categories

        freq_dict = dict.fromkeys(categories)
        for cat in categories:
            counter = self.term_frequencies(cat)
            freq_dict[cat] = counter
        
        pmis = []        
        for term in terms:
            P_w_c =  freq_dict[category][term]/ N
            counts = 0
            for cat in categories:
                counts += freq_dict[cat][term]
            P_w =  counts / N
            P_c = sum(freq_dict[category].values()) / N
            pmi = math.log2(P_w_c / (P_w * P_c))
            pmis.append(pmi)
        return pmis

    def term_frequencies(self, category=None) -> Counter:
        if category:
            tokens_nested_list = [document.tokens for document in self.documents if document.category==category]
        else:
            tokens_nested_list = [document.tokens for document in self.documents]
            
        tokens =  [item for sublist in tokens_nested_list for item in sublist]
        return Counter(tokens)


    def category_frequencies(self):
        """calculate the absolute frequencies 
        of each category in the whole corpus

        Returns:
            dict: frequency of categories in a dictionary
        """
        categories = [document.category for document in self.documents]
        freq_dict = Counter(categories)
        freq_dict = dict(sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True))
        return freq_dict


from nltk.corpus import reuters, stopwords
stop_words = stopwords.words('english')

corpus = Corpus(
    documents=[Document(fileid, reuters.raw(fileid), reuters.categories(fileid), stop_words=stop_words) for fileid in reuters.fileids()],
    categories=reuters.categories()
)

most_common_housing = corpus.term_frequencies('housing').most_common(10)
most_common_housing = [term for term, _ in most_common_housing]

dfs_housing = [corpus.df(term, category='housing') for term in most_common_housing]
dfs_all = [corpus.df(term) for term in most_common_housing]

pmis = corpus.pmi('housing', most_common_housing)