import string
import collections
import matplotlib.pyplot as plt
import seaborn
import matplotlib as mpl
seaborn.set()
mpl.rcParams['figure.dpi'] = 120
plt.rcParams["figure.figsize"] = (20,5)


def tokenize(text):
    """Perform tokenization on the given text.
    Given the restriction to use any tokenizer, 
    remove special characters by replace function
    and split the string.

    Args:
        text (str): input string

    Returns:
        [list]: list of tokenized tokens
    """
    # if use of regex is allowed
    # file_content = re.sub(r'[^a-zA-Z0-9\s]', ' ', file_content)
    
    tokens_list =  text.replace('”',' ').replace('“',' ').replace('’',' ').replace('–',' ').split()
    return tokens_list

def get_conditional_freq(tokens, ngrams_list, n, N):
    """Compute frequncy of the conditioned words

    Args:
        tokens (list): list of all tokens from the corpus
        ngrams_list (list): ngrams list
        n ([type]): unigram/bigram/trigram
        N ([type]): vocabulary size

    Returns:
        [list]: list of frequnecy of conditioned words
    """
    tokens = tokens[:N]
    cond_freq = collections.defaultdict(int)
    for ngram in ngrams_list:
        cond = ' '.join(ngram.split()[:n-1])
        cond_freq[cond] += 1
    return cond_freq


def preprocess(text) -> list:
    # TODO Exercise 2.2.
    """
    : param text: The text input which you must preprocess by
    removing punctuation and special characters, lowercasing,
    and tokenising

    : return: A list of tokens
    """
    file_content = text.lower()
    # Strip punctuations. 
    # Reference: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    file_content = file_content.translate(
        str.maketrans('', '', string.punctuation))
    
    tokens_list = tokenize(file_content)
    return tokens_list

def find_ngram_probs(tokens, model='unigram') -> dict:
    # TODO Exercise 2.2
    """
    : param tokens: Pass the tokens to calculate frequencies
    param model: the identifier of the model ('unigram, 'bigram', 'trigram')
    You may modify the remaining function signature as per your requirements

    : return: n-grams and their respective probabilities
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
    ngram_freq = collections.Counter(ngrams_list)
    
    if n != 0:
        # Get freq of conditioned sequences
        cond_freq = get_conditional_freq(tokens, ngrams_list, n, N)
    
    ngram_prob = collections.defaultdict(float)
    for ngram, freq in ngram_freq.items():
        # Compute normalized frequency
        if n == 1:
            ngram_prob[ngram] = freq / N
        else:
            cond = ' '.join(ngram.split()[:n-1])
            ngram_prob[ngram] = freq / cond_freq[cond]
    
    return ngram_prob


def plot_most_frequent(ngrams, most_frequent=None) -> None:
    # TODO Exercise 2.2
    """
    : param ngrams: The n-grams and their probabilities
    Your function must find the most frequent ngrams and plot their respective probabilities

    You may modify the remaining function signature as per your requirements
    """
    ngrams_sorted = {k: v for k, v in sorted(ngrams.items(), key=lambda item: item[1], reverse=True)}
    
    # get ngram size
    n = len(list(ngrams_sorted.keys())[0].split())
    if most_frequent != None:
        ngrams_sorted = {ngram: prob for ngram, prob in ngrams_sorted.items() if most_frequent == ' '.join(ngram.split()[:n-1])}
                
    #  Pick-out the most frequent top 20 ngrams
    ngrams = list(ngrams_sorted.keys())[:20]
    probs = list(ngrams_sorted.values())[:20]

    _, ax = plt.subplots()
    ax.bar(ngrams, probs)
    ax.set_xlabel("ngrams")
    ax.set_ylabel("probablities")
    plt.xticks(rotation=60, ha='right')   
    plt.show()
    
    return ngrams[0]
