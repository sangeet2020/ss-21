from collections import Counter
from copy import deepcopy
import numpy as np
from typing import List
import warnings
import collections

warnings.filterwarnings("ignore")

class LanguageModel:
    
    def __init__(self, train_tokens: List[str], test_tokens: List[str], N: int, alpha: float, epsilon=1.e-10):
        """ 
        :param train_tokens: list of tokens from the train section of your corpus
        :param test_tokens: list of tokens from the test section of your corpus
        :param N: n of the highest-order n-gram model
        :param alpha: pseudo count for lidstone smoothing
        :param epsilon: threshold for probability mass loss, defaults to 1.e-10
        """
        self.N = N
        self.alpha = alpha
        self.train_tokens = train_tokens 
        self.test_tokens = test_tokens
        self.epsilon = epsilon

    def perplexity(self, n: int):
        """ returns the perplexity of the language model for n-grams with n=n """
        # LM - conditional probability
        lm = self.find_ngram_probs(self.train_tokens, n)
        rfs = self.relative_frequencies(self.test_tokens, n)

        summ = 0
        for ngram, rf in rfs.items():
            summ += rf*np.log2(lm[ngram] + self.epsilon)
        pp = 2**(-summ)

        return pp

    def lidstone_smoothing(self, curr_train_tokens, ngram_test_list, alpha: float, n):
        """ applies lidstone smoothing on train counts

        :param alpha: the pseudo count
        :return: the smoothed counts
        """
        # raise NotImplementedError
        # actual count
        train_count = collections.Counter(curr_train_tokens)

        for k in train_count.keys():
            train_count[k] += alpha

        # print(len(train_count))
        # adding missing ngram from test and assign alpha count
        for w in ngram_test_list:
            train_count.setdefault(w, alpha) # if not present set value to alpha

        # print(len(train_count))
        return train_count

    def get_conditional_freq(self, tokens, ngrams_list, n, N):
        """Compute frequncy of the conditioned words

        Args:
            tokens (list): list of all tokens from the corpus
            ngrams_list (list): ngrams list
            n ([type]): unigram/bigram/trigram
            N ([type]): vocabulary size

        Returns:
            [list]: list of frequcy of conditioned words
        """
        tokens = tokens[:N]
        cond_freq = collections.defaultdict(int)
        for ngram in ngrams_list:
            cond = ' '.join(ngram.split()[:n-1])
            cond_freq[cond] += 1
        return cond_freq

    def find_ngram_probs(self, tokens, n) -> dict:
        """
        : param tokens: Pass the tokens to calculate frequencies
        param model: the identifier of the model ('unigram, 'bigram', 'trigram')
        You may modify the remaining function signature as per your requirements

        : return: n-grams and their respective probabilities
        """
        N = len(tokens)

        if n > 1:
            tokens.extend(tokens[:n-1])
        
        # Generate ngrams for train
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngrams_list =  [" ".join(ngram) for ngram in ngrams]
        
        # generate ngrams for test
        ngrams_test = zip(*[self.test_tokens[i:] for i in range(n)])
        ngrams_test_list =  [" ".join(ngram) for ngram in ngrams_test]

        # Get freq of each ngram
        # ngram_freq = collections.Counter(ngrams_list)
        ngram_freq = self.lidstone_smoothing( ngrams_list, ngrams_test_list, self.alpha, n)
        
        if n != 0:
            # Get freq of conditioned sequences
            cond_freq = self.get_conditional_freq(tokens, ngrams_list, n, N)
        
        ngram_prob = collections.defaultdict(float)
        for ngram, freq in ngram_freq.items():
            v = len(ngram_freq)
            # Compute normalized frequency
            if n == 1:
                ngram_prob[ngram] = freq / (N + self.alpha * v) 
            else:
                cond = ' '.join(ngram.split()[:n-1])
                ngram_prob[ngram] = freq / (cond_freq[cond] + self.alpha * v)
        
        return ngram_prob

    def relative_frequencies(self, tokens:List, n) -> dict:
        """
        Should compute the relative n-gram frequencies of the test set of the corpus.
        :param tokens: the tokenized test set
        :param model: the identifier of the model ('unigram, 'bigram', 'trigram')
        :return: a dictionary with the ngrams as keys and the relative frequencies as values
        """
        N = len(tokens)
        # if model == 'unigram':
        #     n = 1
        # elif model == 'bigram':
        #     n = 2
        #     tokens.append(tokens[0])
        # elif model == 'trigram':
        #     n = 3
        #     tokens.extend([tokens[0], tokens[1]])

        if n > 1:
            tokens.extend(tokens[:n-1])
        
        # Generate ngrams
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngrams_list =  [" ".join(ngram) for ngram in ngrams]
        
        # Get freq of each ngram
        ngram_rel_freq = collections.OrderedDict()
        ngram_freq = collections.Counter(ngrams_list)
        for ngram, freq in ngram_freq.items():
            ngram_rel_freq[ngram] = freq/len(ngrams_list)
        
        return ngram_rel_freq      