import pdb
from typing import List
import matplotlib.pyplot as plt

def train_test_split_data(text: List[str], test_size=0.2):
    """ Splits the input corpus in a train and a test set
    :param text: input corpus
    :param test_size: size of the test set, in fractions of the original corpus
    :return: train and test set
    """
    k = int(len(text) * (1 - test_size))
    return text[:k], text[k:]

def k_validation_folds(text: List[str], k_folds=5):
    """ Splits a corpus into k_folds cross-validation folds
    :param text: input corpus
    :param k_folds: number of cross-validation folds
    :return: the cross-validation folds
    """ 
    k_len = int(len(text)/k_folds)
    cv_folds = list()
    for i in range(k_folds):
        cv_folds.append(text[i*k_len:(i+1)*k_len])
    
    return cv_folds

def plot_pp_vs_alpha(pps: List[float], alphas: List[float]):
    """ Plots n-gram perplexity vs alpha
    :param pps: list of perplexity scores
    :param alphas: list of alphas
    """
    plt.plot(alphas, pps)
    plt.xlabel('Alpha values')
    plt.ylabel('Perplexity')
    plt.title("Alpha vs Perplexity")
    plt.show()

def plot_PPS_vs_alpha(pps: dict(), alphas: List[float]):
    """ Plots n-gram perplexity vs alpha
    :param pps: dict of perplexity scores for ngrams
    :param alphas: list of alphas
    """
    for n, pp in pps.items():
        if n == 1:
            plt.plot(alphas, pp, label='unigram')
        elif n == 2:
            plt.plot(alphas, pp, label='bigram')
        
    plt.xlabel('Alpha values')
    plt.ylabel('Perplexity')
    plt.title("Alpha vs Perplexity")
    plt.legend()
    plt.show()
    