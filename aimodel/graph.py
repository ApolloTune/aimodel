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
    ax.set_title('Top-15 Most Common Unigrams in Positive Comments')
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
    ax.set_title('Top-15 Most Common Bigrams in Positive Comments')
    plt.show()