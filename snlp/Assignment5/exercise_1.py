from typing import List, Dict
#Add imports
import string
import random
import math
import matplotlib.pyplot as plt

def preprocess(text) -> List:
    file_content = text.lower()
    # Strip punctuations. 
    # Reference: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    file_content = file_content.translate(
        str.maketrans('', '', string.punctuation))
    
    tokens_list = file_content.replace('”',' ').replace('“',' ').replace('’',' ').replace('–',' ').split()
    return tokens_list

def train_test_split_data(text:List, test_size:float=0.1):
    shuffle_list = random.sample(text, len(text))
    train = shuffle_list[:math.ceil(len(text)*(1-test_size))] # 70% 
    test = shuffle_list[math.floor(len(text) * (1-test_size)):] # 30%
    return train, test


def get_oov_rates(train:List, test:List) -> List:
    # sort the list in descending order
    train.sort(key=train.count, reverse=True)
    vocab_lens = list(range(1000, 16000, 1000)) # generate 1k - 15k
    oov_rates = list() # to save oov rates

    for l in vocab_lens:
        vocab = train[:l] # extract vocab
        oov_rate = len(set(test) - set(vocab))/len(test)
        oov_rates.append(oov_rate)

    return oov_rates


def plot_oov_rates(oov_rates:Dict) -> None:
    x = range(1000, 16000, 1000) # generate 1k - 15k
    # plotting oov, different color for each language
    plt.loglog(x, oov_rates['en'], 'r')
    plt.loglog(x, oov_rates['fi'], 'g')
    plt.loglog(x, oov_rates['ru'], 'b')
    plt.loglog(x, oov_rates['ta'], 'k')
    # legend
    plt.legend(['English', 'Finnish', 'Russian', 'Tamil'])
    plt.xlabel('Size of vocabulary')
    plt.ylabel('OOV: Rates in percentage')
    # plot
    plt.show()