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
````python
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,labels,random_state=42, test_size=0.2)
train_masks, validation_masks, _, _ = train_test_split(attention_masks,labels,random_state=42,test_size=0.2)
````
-  bu kod parçası, input_ids, labels ve attention_masks veri kümesini eğitim ve doğrulama setlerine ayırarak, bir modelin eğitimi ve doğrulaması için kullanılabilecek veri kümesini oluşturur. Eğitim seti, modelin eğitilmesi için kullanılırken, doğrulama seti, eğitim sırasında modelin performansını değerlendirmek için kullanılır.
- train_test_split veri kümesini iki alt kümeye böler 'train' ve 'validation' olmak üzere
- 'input_ids' veri kümesi, modelin girişlerini temsil eder.
- 'labels' veri kümesi, her örneğin doğru sınıf etikletlerini temsil eder.
- 'random_state=42' parametresi veri kümesesini rastgele bölmek için başlangıç durumu belirtir
- 'test_size=0.2' parametresi veri kümesinin yüzde 20 sini doğrulama işlemine ayırır.
````python
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    validation_labels = torch.tensor(validation_labels, dtype=torch.long)
    train_masks = torch.tensor(train_masks, dtype=torch.long)
    validation_masks = torch.tensor(validation_masks, dtype=torch.long)
````
- bu kod parçası, veri setlerini PyTorch tensorlarına dönüştürerek, PyTorch üzerinde işlem yapılabilmesi için veri setlerini hazır hale getirir. Bu tensorlar daha sonra bir sinir ağı modeline beslenebilir.
````python
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
````
- bu kod parçası, eğitim veri setini mini gruplara ayırarak, modelin eğitimi için kullanılabilecek bir veri yükleyici oluşturur. Bu, ağın her eğitim döneminde belirli bir mini grup veriyle eğitilmesini sağlar.
- 'batch_size=32' Eğitim veri setinin kaçar örnekle bir araya getirileceğini belirten bir sabit değeri ayarlar. Bu durumda, her bir mini grup (batch) 32 örnek içerecek.
- train_data = TensorDataset(train_inputs, train_masks, train_labels): TensorDataset sınıfını kullanarak eğitim veri setini oluşturur. Bu, train_inputs, train_masks ve train_labels tensorlarını birleştirerek bir veri seti oluşturur. Bu veri seti, her bir örneğin girdi tensorları, dikkat maskesi ve etiketlerini içerir.
- train_sampler = RandomSampler(train_data): Eğitim veri setini rastgele örnekleme stratejisiyle karıştırarak bir örnekleme (sampler) oluşturur. Bu, her eğitim döneminde modelin farklı örnekler görmesini sağlar ve ağırlıklı bir şekilde belirli sınıflara odaklanmasını önler.
- train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size): DataLoader sınıfını kullanarak bir veri yükleyici oluşturur. Bu yükleyici, eğitim veri setini mini gruplara ayırır. batch_size parametresi, her bir mini grup (batch) boyutunu belirtir. sampler parametresi, veri setinden örneklemeleri almak için kullanılacak örnekleme stratejisini belirtir.
- Aynı işlemler validasyon seti içinde tekrarlanır.
````python
    config = AutoConfig.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", config=config)
    model.cpu()
````
- Hugging Face'in Transformers kütüphanesini kullanarak, önceden eğitilmiş bir Türkçe BERT modelini yüklüyor ve bu modeli CPU'ya taşıyor.
- config = AutConfig.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2): Bu satır, AutoConfig sınıfını kullanarak bir yapılandırma (configuration) dosyası oluşturuyor. from_pretrained metodu ile "dbmdz/bert-base-turkish-cased" adlı önceden eğitilmiş BERT modelinin yapılandırma bilgilerini yüklüyor. num_labels=2 parametresi, bu modelin sınıflandırma yapacağı sınıf sayısını belirtir. Bu örnekte, 2 sınıflı bir sınıflandırma problemi olduğu belirtiliyor.
- model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", config=config): Bu satır, önceden eğitilmiş BERT modelini yükler. AutoModelForSequenceClassification sınıfı, önceden eğitilmiş bir BERT modelini bir dizi metin sınıflandırma görevi için uyarlar. from_pretrained metoduyla "dbmdz/bert-base-turkish-cased" modeli yüklenir ve yapılandırma bilgileri ile birlikte ayarlanır.
- model.cpu() = Bu satır modeli cpu'ya taşır ve bu modelin cpu üzerinde eğitilmesini ve tahminler yapmasını sağlar.
````python
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5
                      betas=[0.9, 0.999],
                      eps=1e-6  # args.adam_epsilon  - default is 1e-8.
                      )
````
- Bu python kodu BERT modelinin eğitimi için bir AdamW optimizasyon algoritması oluşturur
- model.parameters() = Bu modelin öğrenilebilir parametrelerini alır. Bu parametreler, modelin ağırlık matrislerini ve biasları gibi değişkenlerdir
- lr=2e-5 :  Öğrenme oranı olarak belirtilen değeri kullanır. Bu bir adımın ne kadar büyük olacağını kontrol eder.
- betas=[0.9, 0.999] : Adam optimizer için beta1 ve beta2 parametrelerini belirtir. Bu değerler momentumun ve kara gradyanlarının etkisini dengeleyen hiperparametrelerdir.
- eps=1e-6 : Epsilon değerini belirtir. Bu sıfıra bölünme hatasını önlemek için kullanılır.
````python
    epochs = 5
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=total_steps)
````
- epochs = 5 : Toplam eğitim döngüsü sayısını belirten bir değişken oluşturur. Bu durumda, eğitim 5 döngü sürecek demektir.
- total_steps : Toplam adım sayısını belirtir Bu eğitim sürecinde yapılacak toplam güncelleme sayısını belirtir.
- scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps): Bu satır, bir öğrenme oranı zamanlayıcısı (scheduler) oluşturur. get_linear_schedule_with_warmup fonksiyonu, eğitim sürecinde öğrenme oranını ayarlamak için kullanılır. Bu fonksiyon, ısınma aşaması (warmup) ve ana eğitim aşaması için doğru şekilde ayarlanmış bir zamanlayıcı oluşturur.
````python
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
````
- bu fonksiyon, modelin tahminlerinin doğruluk oranını hesaplayarak modelin performansını değerlendirmek için kullanılır.
````python
def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))
````
- elapsed_rounded = int(round(elapsed)): elapsed değerinin en yakın tam sayıya yuvarlanmasını sağlar. Bu geçen sürenin saniyeler cinsinden bir tam sayıdır
- return str(datetime.timedelta(seconds=elapsed_rounded)): datetime.timedelta sınıfını kullanarak geçen süreyi bir zaman dilimi nesnesine dönüştürür. Geçen süreyi gün, saat, dakika ve saniye olarak okunabilir bir formata getirir.





