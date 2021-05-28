from typing import Dict, List
import matplotlib.pyplot as plt

def plot_pp(pps: Dict):
    """ Plots perplexity vs n for all languages in the corpus

    :param pps: dictionary with langs as keys and lists of perplexity scores as values
    """
    x = range(1, len(pps['en'])+1)
    plt.loglog(x, pps['en'], 'r')
    plt.loglog(x, pps['fi'], 'g')
    plt.loglog(x, pps['ru'], 'b')
    plt.loglog(x, pps['ta'], 'k')
    # legend
    plt.legend(['English', 'Finnish', 'Russian', 'Tamil'])
    plt.xlabel('N (size of ngram)')
    plt.ylabel('Perplexity')
    # plot
    plt.show()


def plot_pp_vs_alpha(pps: List[float], alphas: List[float]):
    """ Plots n-gram perplexity vs alpha
    :param pps: list of perplexity scores
    :param alphas: list of alphas
    """

    plt.plot(alphas, pps)
    # legend
    # plt.legend(['English', 'Finnish', 'Russian', 'Tamil'])
    plt.xlabel('Alpha values K=100 (0.0, 0.01,...,0.99,1.0)')
    plt.ylabel('Perplexity')
    # plot
    plt.show()