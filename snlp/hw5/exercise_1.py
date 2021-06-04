
import string
import random
import re
from collections import Counter, OrderedDict
import math
from typing import List, Dict
import matplotlib.pyplot as plt
import pdb
import seaborn
import matplotlib as mpl
seaborn.set()
mpl.rcParams['figure.dpi'] = 120
plt.rcParams["figure.figsize"] = (9,6)

def preprocess(text) -> List:
    file_content = text.lower()
    file_content = file_content.translate(str.maketrans('', '', string.punctuation))
    tokens_list = file_content.split()
    return tokens_list

def train_test_split_data(text:List, test_size:float=0.1):
    k = int(len(text) * (1 - test_size))
    return text[:k], text[k:]


def get_oov_rates(train:List, test:List) -> List:
    train_counts = OrderedDict(Counter(train).most_common())
    top_tokens = [k for k in list(train_counts)[:15000]]
    oov_rates = []

    for l in range(1000, 16000, 1000):
        vocab = top_tokens[:l]
        unseen_tokens = [tok for tok in test if tok not in vocab]
        oov_rate = len(unseen_tokens)/len(test)
        oov_rates.append(oov_rate)
    return oov_rates


def plot_oov_rates(oov_rates:Dict) -> None:
    fig, ax = plt.subplots()
    for k, v in oov_rates.items():
        plt.loglog(range(1000, 16000, 1000), v, label=k)
        ax.set_xlabel("vocab size")
        ax.set_ylabel("OOV rate")
    plt.legend()
    plt.show()
    

from importlib import reload
import os
import exercise_1
exercise_1 = reload(exercise_1)

# Walk through the data directory and read all the corpora
# For each corpus, read the text, preprocess it and create the train test split for each language

corpora = {} # To save the respective corpora

# Add a loop over each file
for filename in os.listdir('data/'):
    with open(os.path.join('data/', filename)) as f:
        text = f.read()
        tokens = exercise_1.preprocess(text) #preprocess text
        
        train, test = exercise_1.train_test_split_data(tokens, test_size=0.3) # split data
        # Add respective splits to the corpora dict
        lang = filename.split('.')[1]
        corpora[lang] = (train, test)

oov_rates = {}
for lang, (train, test) in corpora.items():
    train = ["a", "b", "b", "c", "d", "d", "d"]
    test = ["a", "b", "z", "x"]
    oov_rates[lang] = exercise_1.get_oov_rates(train, test)