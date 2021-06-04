import pdb
import numpy as np
import collections
from copy import deepcopy
from typing import List
from collections import Counter

from numpy.lib.financial import ppmt


class LanguageModel:
    
    def __init__(self, train_tokens: List[str], test_tokens: List[str], N: int, alpha: float, epsilon=1.e-10):
        """ 
        :param train_tokens: list of tokens from the train section of your corpus
        :param test_tokens: list of tokens from the test section of your corpus
        :param N: n of the highest-order n-gram model
        :param alpha: pseudo count for lidstone smoothing
        :param epsilon: threshold for probability mass loss, defaults to 1.e-10
        """
        self.train_tokens = train_tokens
        self.test_tokens = test_tokens
        self.N = N
        self.alpha = alpha
        

    def perplexity(self):
        """ returns the perplexity of the language model for n-grams with n=n """
        self.train_ngrams = self.get_ngrams(self.train_tokens, self.N)
        self.test_ngrams = self.get_ngrams(self.test_tokens, self.N)
        self.V = len(Counter(self.train_ngrams))
        
        rel_freq = self.relative_freq(self.test_ngrams)
        smoothed_probs = self.lidstone_smoothing(self.test_ngrams)
        summ = 0
        for test_ngram, rf in rel_freq.items():
            summ += rf * np.log2(smoothed_probs[test_ngram])
        pp = 2**(-summ)
        return pp

    
    def get_ngrams(self, tokens, N):
        if N == 2:
            tokens.append(tokens[0])
        elif N == 3:
            tokens.extend([tokens[0], tokens[1]])
        
        ngrams = zip(*[tokens[i:] for i in range(N)])
        ngrams_list =  [" ".join(ngram) for ngram in ngrams]
        return ngrams_list

    
    def get_ngrams_counts(self):
        train_ngram_counts = Counter(self.train_ngrams)
        test_ngram_counts = Counter(self.test_ngrams)
        
        for k,v in test_ngram_counts.items():
            if k not in train_ngram_counts:
                train_ngram_counts[k] = 0
                
        return train_ngram_counts
        
    def history_counts(self, tokens):
        # pdb.set_trace()        
        N = self.N - 1
        history = self.get_ngrams(tokens, N)
        history_counts = Counter(history)
        return history_counts
    
    def lidstone_smoothing(self, ngrams):
        if self.N != 1:
            history_counts = self.history_counts(self.train_tokens)
        ngram_counts = self.get_ngrams_counts()
        
        if self.N == 1:
            V = len(ngram_counts)
        v_prev = self.history_counts(self.test_tokens)
        
        p_w = collections.defaultdict(float)
        for ngram in ngrams:
            # pdb.set_trace()
            history = ' '.join(ngram.split()[:self.N-1])
            if len(history) == 0:
                history_count = len(self.train_ngrams)
            else:
                history_count = history_counts[history]        
                V = v_prev[history]
            p_w[ngram] = ( ngram_counts[ngram] + self.alpha ) / ( history_count + self.alpha*V )
        # pdb.set_trace()    
        return p_w
    
    def relative_freq(self, ngrams):
        ngram_counts = Counter(ngrams)
        rel_freq = collections.defaultdict(float)
        for token, counts in ngram_counts.items():
            rel_freq[token] = counts / len(ngrams)
        # pdb.set_trace()
        return rel_freq
            

# from importlib import reload
# import os
# import exercise_2
# import exercise_1

# exercise_1 = reload(exercise_1)
# exercise_2 = reload(exercise_2)


# corpora = {} # To save the respective corpora
# # TODO: Add a loop over each file
# for filename in os.listdir('data/'):
#     with open(os.path.join('data/', filename)) as f:
#         text = f.read()
#         pp = exercise_1.preprocess(text) #TODO: preprocess text
#         train, test = exercise_1.train_test_split_data(pp, test_size=0.3) #TODO: split data
#         #TODO: Add respective splits to the corpora dict
#         lang = filename.split('.')[1]
#         corpora[lang] = (train, test)

# N = 3
# PPs = {}
# for lang, (train, test) in corpora.items():
#     pp_list = []
#     for i in range(1,N+1):
#         LM = LanguageModel(train, test, N=i, alpha=1)
#         pp_list.append(LM.perplexity())
#     PPs[lang] = pp_list
# # pdb.set_trace()
# exercise_2.plot_pp(PPs)