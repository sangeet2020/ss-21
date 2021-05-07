import numpy as np
from typing import Dict, List
import pdb
import random
import collections
import matplotlib.pyplot as plt
import seaborn
import matplotlib as mpl
seaborn.set()
mpl.rcParams['figure.dpi'] = 120
plt.rcParams["figure.figsize"] = (20,5)

def train_test_split(corpus:List, test_size:float) -> (List, List):
    """
    Should split a corpus into a train set of size len(corpus)*(1-test_size)
    and a test set of size len(corpus)*test_size.
    :param text: the corpus, i. e. a list of strings
    :param test_size: the size of the training corpus
    :return: the train and test set of the corpus
    """
    random.shuffle(corpus)
    train = corpus[int(0.0*len(corpus)):int(0.9*len(corpus))]
    test = corpus[int(0.9*len(corpus)):int(1.0*len(corpus))]
    
    return train, test


def relative_frequencies(tokens:List, model='unigram') -> dict:
    """
    Should compute the relative n-gram frequencies of the test set of the corpus.
    :param tokens: the tokenized test set
    :param model: the identifier of the model ('unigram, 'bigram', 'trigram')
    :return: a dictionary with the ngrams as keys and the relative frequencies as values
    """
    N = len(tokens)
    if model == 'unigram':
        n = 1
    elif model == 'bigram':
        n = 2
        tokens.append(tokens[0])
    elif model == 'trigram':
        n = 3
        tokens.extend([tokens[0], tokens[1]])
    
    # Generate ngrams
    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngrams_list =  [" ".join(ngram) for ngram in ngrams]
    
    # Get freq of each ngram
    ngram_rel_freq = collections.OrderedDict()
    ngram_freq = collections.Counter(ngrams_list)
    for ngram, freq in ngram_freq.items():
        ngram_rel_freq[ngram] = freq/len(ngrams_list)
    
    return ngram_rel_freq


def pp(lm:Dict, rfs:Dict) -> float:
    """
    Should calculate the perplexity score of a language model given the relative
    frequencies derived from a test set.
    :param lm: the language model (from exercise 2)
    :param rfs: the relative frequencies
    :return: a perplexity score
    """
    summ = 0
    for ngram, rf in rfs.items():
        summ += rf*np.log2(lm[ngram])
    pp = 2**(-summ)
    
    return pp


def plot_pps(pps:List) -> None:
    """
    Should plot perplexity value vs. language model
    :param pps: a list of perplexity scores
    :return:
    """
    x_labs = ['unigram', 'bigram', 'trigram']
    _, ax = plt.subplots()
    ax.bar(x_labs, pps)
    ax.set_xlabel("language model")
    ax.set_ylabel("perplexity score")
    plt.show()
    
