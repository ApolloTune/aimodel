import matplotlib.pyplot as plt
import numpy as np
# Unigram Tablolalama
def setUnigram(top_15):
    plt.rcdefaults()
    fig, ax = plt.subplots()

    ngram = top_15.keys()
    y_pos = np.arange(len(ngram))
    performance = top_15.values()

    ax.barh(y_pos, performance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ngram)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Counts')
    ax.set_title('Top-15 Most Common Unigrams in lyrics')
    plt.show()
def setBigram(top_15):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ngram = top_15.keys()
    y_pos = np.arange(len(ngram))
    performance = top_15.values()

    ax.barh(y_pos, performance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ngram)
    ax.invert_yaxis()
    ax.set_xlabel('Counts')
    ax.set_title('Top-15 Most Common Bigrams in lyrics')
    plt.show()
def setHistogram(text_list, tokenizer):
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t), text_list))
    tokenized_texts_len = list(map(lambda t: len(t), tokenized_texts))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tokenized_texts_len, bins=40)
    ax.set_xlabel("Length of Lyrics Embeddings")
    ax.set_ylabel("Number of Lyrics")
    plt.show()