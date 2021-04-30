import matplotlib as mpl
import nltk
import operator
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
mpl.rcParams['figure.dpi'] = 100


def analysis(name, data):
    """
    Plot Zipfian distribution of data + true Zipfian distribution. Compute MSE.

    :param name: title of the graph
    :param data: list of data
    """

    frq_dict = nltk.FreqDist(data)
    frq_dict = dict(
        sorted(frq_dict.items(), key=operator.itemgetter(1), reverse=True))

    frq_list = list(frq_dict.values())
    word_list = list(frq_dict.keys())
    word_pos = range(1, len(word_list)+1)  # Since as log(0) is undefined.

    # Ideal Zipfian
    m = frq_list[0]
    ideal_zipf = [m*1/x for x in word_pos]
    # import pdb; pdb.set_trace()

    # Compute mean squared error
    mse_np = np.sum(np.square(np.array(ideal_zipf) -
                              np.array(frq_list)))/len(ideal_zipf)
    print('MSE for %s: %.10f' % (name, mse_np))

    _, ax = plt.subplots()
    plt.figure(num=None, figsize=(9.5, 6), dpi=300,
               facecolor='r', edgecolor='k')

    ax.plot(word_pos, ideal_zipf, 'g', label='Ideal Zipfian')
    ax.loglog(word_pos, frq_list, 'r+', label=str(name))
    ax.set_ylabel("frequency")
    ax.set_xlabel("rank")
    ax.set_title("Frequency Curve with log scaling: "+str(name))
    ax.legend()
