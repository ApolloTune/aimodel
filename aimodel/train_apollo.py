import graph
import init
import cleantext
import training_model
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


# def createBigram(df_pos):
#     token_list = cleantext.text_process(df_pos)
#     freqencies = Counter(token_list)
#     freqencies_sorted = sorted(freqencies.items(), key=lambda k: k[1], reverse=True)
#     top_15 = dict(freqencies_sorted[0:15])
#     graph.setBigram(top_15)


# def createUnigram(df_pos):
#     token_list = cleantext.text_process(df_pos)
#     bigrams = list(ngrams(token_list, 2))
#     freqencies_bigrams = Counter(bigrams)
#     freqencies_sorted_bigrams = sorted(freqencies_bigrams.items(), key=lambda k: k[1], reverse=True)
#     top_15_bigrams = dict(freqencies_sorted_bigrams[0:15])
#     graph.setUnigram(top_15_bigrams)


# nltk.download('punkt')
# nltk.download('stopwords')

# init.startupInit()


df_train = pd.read_csv("../dataset/train.csv", index_col="id", encoding="utf-8")
df_test = pd.read_csv("../dataset/test.csv", index_col="id", encoding="utf-8")
# print(df_train.head()) # ilk 5 satırı getirdim
# print(df_train["class"].value_counts()) # Classların adedi
# print(df_train.isna().sum()) # Boş değer var mı olmaması gerekir varsa boşları sil
# stop_words = stopwords.words("turkish")
# stop_words.extend(["bir", "film", "filmi", "filme", "filmde", "filmden", "filmin", "kadar", "bi", "ben"])

# exclude = set(string.punctuation)  # !, ., ? gibi karakterler

# Arabesk Şarkıları bigrams
''':Arabesk duygular bigrams
df_arabesk = df_train[df_train["class"] == 0]
createBigram(df_arabesk)
'''
#  Aşk şarkıları bigrams
''':Aşk duygular bigrams
df_ask = df_train[df_train["class"] == 1]
createBigram(df_ask)
'''

# Hareketli Şarkıları bigrams
''':Hareketli duygular bigrams
df_hareket = df_train[df_train["class"] == 2]
createBigram(df_hareket)
'''

#  Motivasyon şarkıları bigrams
''':Motivasyon duygular bigrams
df_motivasyon = df_train[df_train["class"] == 3]
createBigram(df_motivasyon)
'''

# Arabesk Şarkıları unigrams
''':Arabesk Şarkıları unigrams
df_arabesk = df_train[df_train["class"] == 0]
createUnigram(df_arabesk)
'''

# Aşk şarkıları unigrams
''':Aşk şarkıları unigrams
df_ask = df_train[df_train["class"] == 1]
createUnigram(df_ask)
'''

# Hareketli Şarkıları unigrams
''':Hareketli Şarkıları unigrams
df_hareket = df_train[df_train["class"] == 2]
createUnigram(df_hareket)
'''

# Motivasyon şarkıları unigrams
''':Motivasyon şarkıları unigrams
df_motivasyon = df_train[df_train["class"] == 3]
createUnigram(df_motivasyon)
'''

# TRAINING MODEL
''':Training Model'''
lyrics = df_train.lyrics.values
df_class = df_train['class'].values
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased", do_lower_case=True)
# graph.setHistogram(lyrics, tokenizer) # Görüldüğü üzere token sayımız 1700 lere kadar çıkıyor bert maksimum 512 token kabul ediyor
# Burada Kaldım....
indices = tokenizer.batch_encode_plus(list(lyrics), max_length=512, add_special_tokens=True, return_attention_mask=True,
                                      padding='longest', truncation=True)
input_ids = indices["input_ids"]
attention_masks = indices["attention_mask"]
# print(input_ids[0])
# print(lyrics[0])
training_model.set_train_model(input_ids, df_class, attention_masks)

# MODEL PERFORMANCE TEST
# :Model Performance Test
test_lyrics = df_test.lyrics.values
test_df_class = df_test['class'].values
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased", do_lower_case=True)
test_indices = tokenizer.batch_encode_plus(list(test_lyrics), max_length=512, add_special_tokens=True, return_attention_mask=True, padding='longest', truncation=True)
test_input_ids = test_indices["input_ids"]
test_attention_masks = test_indices["attention_mask"]
training_model.model_performance_test(test_input_ids, test_attention_masks, test_df_class)
