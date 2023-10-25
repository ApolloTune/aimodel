import graph
import init
import cleantext
import numpy as np
import pandas as pd
import os
import torch
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from transformers import AutoTokenizer
def createBigram(df_pos):
    token_list = cleantext.text_process(df_pos)
    freqencies = Counter(token_list)
    freqencies_sorted = sorted(freqencies.items(), key=lambda k: k[1], reverse=True)
    top_15 = dict(freqencies_sorted[0:15])
    graph.setUnigram(top_15)

def createUnigram(df_pos):
    token_list = cleantext.text_process(df_pos)
    bigrams = list(ngrams(token_list, 2))
    freqencies_bigrams = Counter(bigrams)
    freqencies_sorted_bigrams = sorted(freqencies_bigrams.items(), key=lambda k: k[1], reverse=True)
    top_15_bigrams = dict(freqencies_sorted_bigrams[0:15])
    graph.setBigram(top_15_bigrams)

# nltk.download('punkt')
# nltk.download('stopwords')

# init.startupInit()


df_train = pd.read_csv("../dataset/train.csv",index_col=[0],encoding="windows-1252")
df_test = pd.read_csv("../dataset/test.csv",index_col=[0],encoding="windows-1252")
# print(df_train.head()) # ilk 5 satırı getirdim
# print(df_train["Label"].value_counts()) # Labelların adedi
# print(df_train.isna().sum()) # Boş değer var mı olmaması gerekir varsa boşları sil
stop_words=stopwords.words("turkish")
stop_words.extend(["bir","film","filmi","filme","filmde","filmden","filmin","kadar","bi","ben"])

exclude = set(string.punctuation) # !, ., ? gibi karakterler

# pozitif duygular bigrams
df_pos = df_train[df_train["Label"] == 1]
# createBigram(df_pos)

# negatif duygular bigrams
df_neg = df_train[df_train["Label"] == 0]
# createBigram(df_neg)

# pozitif duygular unigrams
# createUnigram(df_pos)

# negatif duygular unigrams
# createUnigram(df_neg)
comments = df_train.comment.values
labels = df_train.Label.values
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased", do_lower_case=True)
graph.setHistogram(comments, tokenizer)