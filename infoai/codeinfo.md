```python 
def text_process(df):
    token_list = []
    for i, r in df.iterrows():
        text = ''.join(ch for ch in df["comment"][i] if ch not in exclude and ch != "’")
        tokens = word_tokenize(text)
        tokens = [tok.lower() for tok in tokens if tok not in stop_words]
        token_list.extend(tokens)
    return token_list
```
- bu kod, DataFrame'deki her bir pozitif etiketli yorumu işleyerek, her yorumun temizlenmiş kelimelerini token_list adlı bir liste içinde biriktirir. Bu işlem, metin madenciliği veya doğal dil işleme uygulamalarında yaygın olarak kullanılır.
````python
freqencies = Counter(token_list)
freqencies_sorted=sorted(freqencies.items(), key=lambda k:k[1], reverse=True)
top_15=dict(freqencies_sorted[0:15])
````
- bu kod, token_list içindeki kelimelerin frekanslarını hesaplar, bu frekansları sıralar ve en yüksek frekansa sahip olan ilk 15 kelimeyi top_15 adlı bir sözlüğe atar.
````python
def setHistogram(text_list, tokenizer):
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t), text_list))
    tokenized_texts_len = list(map(lambda t: len(t), tokenized_texts))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tokenized_texts_len, bins=40)
    ax.set_xlabel("Length of Comment Embeddings")
    ax.set_ylabel("Number of Comments")
    return
````
- Bu fonksiyon, metinlerin tokenleştirilmiş hali (yani, her bir metnin kaç token içerdiği) hakkında bir görsel sunar. 
````python
indices = tokenizer.batch_encode_plus(list(comments), max_length=128, add_special_tokens=True, return_attention_mask=True, padding='longest', truncation=True)
input_ids=indices["input_ids"]
attention_masks=indices["attention_mask"]
print(input_ids[0])
print(comments[0])
````  
-  bu kod bir metin listesini BERT modeli için uygun formata dönüştürür ve token ID'leri ile dikkat maskelerini input_ids ve attention_masks değişkenlerinde saklar. Ayrıca, ilk metnin token ID'lerini ve orijinal metni ekrana yazdırır.

