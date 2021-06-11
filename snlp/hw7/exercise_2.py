import string
import re
from collections import Counter
from typing import List, Tuple
import nltk
import pdb
from nltk.tokenize.treebank import TreebankWordDetokenizer

#TODO: Implement
def preprocess(text) -> List:
    '''
    params: text-text corpus
    return: tokens in the text
    '''
    file_content = text.lower()
    file_content = file_content.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(file_content)
    return tokens


class KneserNey:
    def __init__(self, tokens: List[str], N: int, d: float):
        '''
        params: tokens - text corpus tokens
        N - highest order of the n-grams
        d - discounting paramater
        '''
        self.d = d

    def get_n_grams(self, tokens: List[str], n: int) -> List[Tuple[str]]:
        if n == 2:
            tokens.append(tokens[0])
        elif n == 3:
            tokens.extend([tokens[0], tokens[1]])
            
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngrams_list =  [tuple(ngram) for ngram in ngrams]
        return ngrams_list

    def get_params(self, trigram) -> None:
        w1, w2, w3 = tuple(trigram.split())
        params = {
            "Nw1_w2_w3"         :  self.get_Nw1_w2_w3(w1, w2, w3),
            "Nw1_w2"            :  self.get_Nw1_w2(w1, w2),
            "Nplus_dot_w2_w3"   :  self.get_Nplus_dot_w2_w3(w2, w3),
            "Nplus_dot_w2_dot"  :  self.get_Nplus_dot_w2_dot(w2),
            "Nplus_dot_w3"      :  self.get_Nplus_dot_w3(w3),
            "Nplus_dot_dot"     :  self.get_Nplus_dot_dot(),
            "Nplus_w1_w2_dot"   :  self.get_Nplus_w1_w2_dot(w1, w2),
            "Nplus_w2_dot"      :  self.get_Nplus_w2_dot(w2),    
        }
        params["lambda_w1_w2"] = (self.d / params["Nw1_w2"]) * params["Nplus_w1_w2_dot"]
        params["lambda_w2"] = (self.d / self.V["unigrams"].get(w2)) * params["Nplus_w2_dot"]        
        print("V= ", self.V_counts["unigrams"])
        return params
        
    
    def global_counter(self, tokens):
        self.unigrams = tokens
        self.bigrams = self.get_n_grams(tokens, 2)
        self.trigrams = self.get_n_grams(tokens, 3)
        
        # Get a dict with combined counts of unique- unigrams, bigrams, trigrams
        self.global_dict = nltk.FreqDist(self.unigrams + self.bigrams + self.trigrams)
        self.V = {
            "unigrams": nltk.FreqDist(self.unigrams),
            "bigrams": nltk.FreqDist(self.bigrams),
            "trigrams": nltk.FreqDist(self.trigrams)
        }
        self.V_counts = {
            "unigrams": len(self.V["unigrams"]),
            "bigrams": len(self.V["bigrams"]),
            "trigrams": len(self.V["trigrams"])
        }
    
    def get_Nw1_w2_w3(self, w1, w2, w3):
        Nw1_w2_w3 = self.V["trigrams"].get((w1, w2, w3))
        if Nw1_w2_w3 != None:
            return Nw1_w2_w3
        else:
            return 0    
        
    def get_Nw1_w2(self, w1, w2):
        Nw1_w2 = self.V["bigrams"].get((w1, w2))
        if Nw1_w2 != None:
            return Nw1_w2
        else:
            return 0

    def get_Nplus_dot_w2_w3(self, w2, w3):
        Nplus_dot_w2_w3 = 0
        for trigram, _ in self.V["trigrams"].items():
            if (trigram[1],trigram[2]) == (w2, w3):
                Nplus_dot_w2_w3 +=1
        
        return Nplus_dot_w2_w3
        
    
    def get_Nplus_dot_w2_dot(self, w2):
        Nplus_dot_w2_dot = 0
        for trigram, _ in self.V["trigrams"].items():
            if trigram[1] == w2:
                Nplus_dot_w2_dot +=1
        return Nplus_dot_w2_dot
    
    def get_Nplus_dot_w3(self, w3):
        Nplus_dot_w3 = 0
        for bigram, _ in self.V["bigrams"].items():
            if bigram[1] == w3:
                Nplus_dot_w3 +=1
        
        return Nplus_dot_w3
    
    def get_Nplus_dot_dot(self):
        Nplus_dot_dot = self.V_counts["bigrams"]
        return Nplus_dot_dot
    
    def get_Nplus_w1_w2_dot(self, w1, w2):
        Nplus_w1_w2_dot = 0
        for trigram, _ in self.V["trigrams"].items():
            if (trigram[0], trigram[1]) == (w1, w2):
                Nplus_w1_w2_dot +=1
        
        return Nplus_w1_w2_dot
    
    def get_Nplus_w2_dot(self, w2):
        Nplus_w2_dot = 0
        for bigram, _ in self.V["bigrams"].items():
            if bigram[0] == w2:
                Nplus_w2_dot +=1
        return Nplus_w2_dot