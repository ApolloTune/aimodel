```python 
    for i,r in df_pos.iterrows():
    text=''.join(ch for ch in df_pos["comment"][i] if ch not in exclude and ch != "’")
    tokens=word_tokenize(text)
    tokens=[tok.lower() for tok in tokens if tok not in stop_words]
    token_list.extend(tokens)
```
- bu kod, DataFrame'deki her bir pozitif etiketli yorumu işleyerek, her yorumun temizlenmiş kelimelerini token_list adlı bir liste içinde biriktirir. Bu işlem, metin madenciliği veya doğal dil işleme uygulamalarında yaygın olarak kullanılır.
````python
freqencies = Counter(token_list)
freqencies_sorted=sorted(freqencies.items(), key=lambda k:k[1], reverse=True)
top_15=dict(freqencies_sorted[0:15])
````
- bu kod, token_list içindeki kelimelerin frekanslarını hesaplar, bu frekansları sıralar ve en yüksek frekansa sahip olan ilk 15 kelimeyi top_15 adlı bir sözlüğe atar.




