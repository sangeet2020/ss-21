import pdb
import numpy as np
import collections
from typing import List
from collections import Counter


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
        self.train_ngram_counts = Counter(self.train_ngrams)
        self.test_ngram_counts = Counter(self.test_ngrams)
        
        for k,v in self.test_ngram_counts.items():
            if k not in self.train_ngram_counts:
                self.train_ngram_counts[k] = 0
                
        return self.train_ngram_counts
        
    def history_counts(self, tokens):
        # pdb.set_trace()        
        N = self.N - 1
        history = self.get_ngrams(tokens, N)
        history_counts = Counter(history)
        return history_counts
    
    def compute_V(self):
        N = 3
        ngram_counts = []
        for n in range(1, N+1):
            ngrams = self.get_ngrams(self.train_tokens, n)
            ngram_counts.append(Counter(ngrams))
            
        V = set(ngram_counts[0].keys())
        
        V = V.union([test_token for test_token in self.test_tokens])
        return V
        
    def lidstone_smoothing(self, ngrams):
        if self.N != 1:
            history_counts = self.history_counts(self.train_tokens)
        ngram_counts = self.get_ngrams_counts()
        V = len(self.compute_V())
        
        p_w = collections.defaultdict(float)
        for ngram in ngrams:
            # pdb.set_trace()
            history = ' '.join(ngram.split()[:self.N-1])
            if len(history) == 0:
                history_count = len(self.train_ngrams)
            else:
                history_count = history_counts[history]        
                
            # pdb.set_trace()
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