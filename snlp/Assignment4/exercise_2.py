from Bio import SeqIO
from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


def sample_records(genome_loc: Path, genome_red_loc: Path, num_records: int):
    """ Samples n reads from a fasta file and saves them to a new file.

    :param genome_loc: path to the unreduced file
    :param genome_red_loc: path to the reduced file
    :num_records: number of reads to sample
    """
    sequence = SeqIO.parse(genome_loc.absolute(), 'fasta')
    sequence = list(sequence)

    genome_red_loc.touch()
    handle = genome_red_loc.open('w')
    
    for _ in range(num_records):
        index = np.random.randint(len(sequence))
        SeqIO.write(sequence[index], handle, 'fasta')
    

def get_k_mers(genome_red_loc: Path, k: int) -> List[str]:
    """ Samples k-mers from a fasta file (preferrably the reduced one).
        See also https://en.wikipedia.org/wiki/K-mer
    :param genome_loc: path to the fasta file
    :param k: length of of the n-mer
    :return: a list of n-mers
    """
    # Read
    sequence = SeqIO.parse(genome_red_loc, 'fasta')
    # sequence = list(sequence)
    sequence = list(sequence)[:10]
    # Combine
    text = ''
    for s in sequence:
        text += str(s.seq)
    text = text.upper()
    # make k-mers
    kmers_list = []
    for i in range(len(text)):
        if i+k >= len(text):
            break
        kmers_list.append(text[i:i+k])

    return kmers_list


def get_k_mers_24(genome_red_loc: Path, k: int, tandem_repeats=False) -> List[str]:
    """ Samples k-mers from a fasta file (preferrably the reduced one), but this time 
        only for tandem repeat regions or non tandem repeat regions.

    :param genome_loc: path to the fasta file
    :param k: length of of the n-mer
    :param tandem_repeats: get only tandem repeats or non-tandem repeats
    :return: a list of n-mers
    """


def k_mer_statistics(genome_red_loc: Path, K: int, delta=1.e-10) -> Tuple:
    """ Calculates relative k-mer frequencies and conditional k-mer probabilities 
        on the provided fasta file.

    :param genome_red_loc: path to the fasta file
    :param K: upper bound of the k of k-mers
    :param delta: threshold for probability mass loss, defaults to 1.e-10
    :return: lists of relative frequencies and conditional probabilities
    """
    Kmer_rel_freq = {}
    Kmer_cond_prob = {}

    for i in range(1,K+1):
        kmers_list = get_k_mers(genome_red_loc, i)
        # print(Kmer_cond_prob)
        # relative frequency
        rel_freq = {}
        for kmer in kmers_list:
            rel_freq[kmer] = kmers_list.count(kmer)/len(kmers_list)

        Kmer_rel_freq[i] = rel_freq # all rel freq for one k-mer

        if i == 1: # condition prob is equal to relative frequency K=1
            cond_prob = rel_freq
            Kmer_cond_prob[i] = cond_prob
        else:

            # conditional probability
            cond_prob = {}
            for kmer in kmers_list:
                cond_prob[kmer] = rel_freq[kmer]/Kmer_rel_freq[i-1][kmer[:-1]]

            Kmer_cond_prob[i] = cond_prob # all cond prob for one k-mer

    return Kmer_rel_freq, Kmer_cond_prob

    


def k_mer_statistics_24(genome_red_loc: Path, K: int, tandem_repeats=False, delta=1.e-10) -> Tuple:
    """ Calculates relative k-mer frequencies and conditional k-mer probabilities 
        on the provided fasta file, but this time only for tandem repeat regions 
        or non tandem repeat regions.

    :param genome_red_loc: path to the fasta file
    :param K: upper bound of the k of k-mers
    :param tandem_repeats: get only tandem repeats or non-tandem repeats
    :param delta: threshold for probability mass loss, defaults to 1.e-10
    :return: lists of relative frequencies and conditional probabilities
    """


def conditional_entropy(rel_freqs: Dict, cond_probs: Dict) -> float:
    """ Calculates the conditional entropy of a corpus given by relative k-mer frequencies
        and conditional k-mer probabilities

    :param rel_freqs: (a dictionary of) relative frequencies
    :param cond_probs: (a dictionary of) conditional probabilities
    :return: the conditional entropy of the corpus
    """
    con_ent = 0
    for k in rel_freqs.keys():
        con_ent += -rel_freqs[k]*np.log2(cond_probs[k])

    return con_ent


def plot_k_mers(rel_freqs, n=10, k=5):
    """ Plots n most frequent k-mers vs. their frequency.

    :param rel_freqs: the list of relative frequency dicts
    :param n: the number of most frequent k-mers to plot
    :param k: the k of k-mers
    """
    fig, ax = plt.subplots(k)

    for i in range(1, k+1):
        # for each k
        rel_freq = rel_freqs[i]

        kmers_sorted = {key: val for key, val in sorted(rel_freq.items(), key=lambda item: item[1], reverse=True)}
        
        #  Pick-out the most frequent top n kmers
        kmers = list(kmers_sorted.keys())[:n]
        freq = list(kmers_sorted.values())[:n]

        ax[i-1].loglog(kmers, freq)
        ax[i-1].set_xlabel("kmers")
        ax[i-1].set_ylabel("frequency")
        ax[i-1].set_title("K="+str(i))
        

    # plt.xticks(rotation=60, ha='right')   
    plt.show()
        


def plot_conditional_entropies(H_ks:List[float]):
    """ Plots conditional entropy vs. k-mer length

    :param H_ks: the conditional entropy scores
    """
    plt.plot(range(1,len(H_ks)+1), H_ks)
    plt.xlabel("k-mer length")
    plt.ylabel("Conditional entropy")
    plt.show()