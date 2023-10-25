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
from collections import Counter


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
# pozitif duygular
df_pos = df_train[df_train["Label"] == 1] # print(df_pos.head())
token_list = cleantext.text_process(df_pos)
freqencies = Counter(token_list)
freqencies_sorted = sorted(freqencies.items(), key=lambda k:k[1], reverse=True)
top_15 = dict(freqencies_sorted[0:15])
# graph.setUnigram(top_15)


# negatif duygular
df_neg = df_train[df_train["Label"] == 0]
neg_token_list = cleantext.text_process(df_neg)
neg_freqencies = Counter(neg_token_list)
neg_freqencies_sorted = sorted(neg_freqencies.items(), key=lambda k:k[1], reverse=True)
neg_top_15 = dict(neg_freqencies_sorted[0:15])
graph.setUnigram(neg_top_15)
