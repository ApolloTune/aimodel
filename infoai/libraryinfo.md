1. import numpy as np:  
numpy (Numerical Python), bilimsel hesaplamalar için kullanılan bir Python kütüphanesidir. numpy dizileri ve matematiksel işlemler sağlar. Özellikle, çok boyutlu dizilerle çalışırken son derece etkilidir.
Örneğin, bir matrisi çarpma, transpozunu alma gibi işlemler numpy ile hızlı ve verimli bir şekilde gerçekleştirilebilir.
2. import pandas as pd:  
pandas, veri analizi ve manipülasyonu için kullanılan bir Python kütüphanesidir. Dataframe adı verilen, tablo benzeri yapılarla çalışmayı sağlar. Veri analizi ve temizliği için oldukça popülerdir.
Örneğin, CSV dosyalarından veri okuma, veri gruplama, filtreleme gibi işlemler pandas ile kolayca yapılabilir.
3. import os:  
os, işletim sistemi işlemlerini (dosya işlemleri, dizin oluşturma, dizin değiştirme vb.) gerçekleştirmek için kullanılan bir Python modülüdür.
Örneğin, dosya yolu oluşturma, dosya silme gibi işlemler os modülü ile yapılabilir.
4. import torch:  
torch, PyTorch adlı bir makine öğrenmesi kütüphanesinin Python arayüzüdür. PyTorch, tensör hesaplamalarını ve derin öğrenme modelleri oluşturmayı sağlar.
Özellikle, sinir ağı modelleri oluşturma, eğitme ve değerlendirme işlemlerinde kullanılır.
5. from nltk import word_tokenize:  
word_tokenize, verilen bir metni kelimelerine ayırarak bir liste haline getirir. Örneğin, "Merhaba dünya!" metni ["Merhaba", "dünya", "!"] şeklinde tokenize edilir. Bu, bir metni analiz etmek için ilk adımdır.
6. from nltk.corpus import stopwords:  
stopwords modülü, yaygın olarak kullanılan dildeki "durma kelimeleri"ni içerir. Bu, bir metindeki yaygın, anlamsız kelimeleri (örneğin, "ve", "veya", "bir", "bu" gibi) filtrelemek için kullanılır. Bu, analizin odaklanmasını ve verilerin daha temiz bir şekilde işlenmesini sağlar.
7. from collections import Counter:  
Counter, bir veri koleksiyonunun elemanlarını saymak için kullanılır. Örneğin, bir listedeki her bir öğenin kaç kez geçtiğini belirlemek için kullanılabilir. Bu, metin verilerinin içeriğini daha iyi anlamak için kullanılabilir.
8. from nltk.util import ngrams:  
Natural Language Toolkit (NLTK) kütüphanesinin bir parçasıdır ve metin verileri üzerinde dil modeli oluştururken veya dil analizi yaparken kullanılır. Bu kütüphane, metinleri belirli bir "n" (sıklık) değerine göre bir araya getirerek n-gram'lar oluşturmanıza olanak tanır.
9. from transformers import AutoTokenizer:  
dil modelleri ile çalışırken metin verilerinizi modelin anlayabileceği forma dönüştürmek için kullanılır. Bu, metin verilerinin işlenmesi ve NLP görevlerinin gerçekleştirilmesi için önemli bir adımdır.
10. import matplotlib.pyplot as plt:  
matplotlib genellikle veri analizi ve makine öğrenmesi projelerinde veri keşfi ve sonuçların görselleştirilmesi için kullanılır. Ayrıca, veri bilimi topluluğunda çok yaygın olarak kullanılan bir kütüphanedir.
Histogram, Çubur-Bar, Dağılım, Çizgi grafikleri çizilmesini sağlar.
11. from sklearn.model_selection import train_test_split:  
train_test_split fonksiyonu, veri setini rastgele iki parçaya ayırarak, modelin eğitim verileri üzerinde öğrenmesini ve test verileri üzerinde değerlendirilmesini sağlar. Bu, modelin gerçek dünya verileri üzerinde nasıl performans göstereceğini tahmin etmek için kullanılır.
12. from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler:
- - TensorDataset: torch.Tensor nesnesini birleştirerek veri setini oluşturur.
- - DataLoader: veri kümesini yüklemek ve işlemek için kullanılır. Veriyi mini gruplara ayırarak eğitim döngüsü oluşturur. Ayrıca, veriyi rastgele karıştırma, paralel yükleme ve diğer özellikleri sağlar.
- - RandomSampler: Veri kümesinden rasgele örnekler seçmek için kullanılır. Bu, eğitim sürecinde çeşitlilik eklemek için yaygın olarak kullanılır.
- - SequentialSampler: Veri kümesindeki örnekleri sırayla seçmek için kullanılır. Bu, doğrusal bir sıra ile veri setine erişim sağlar.
13. from transformers import AutoModelForSequenceClassification, AdamW, AutoConfig:  
- - AutoModelForSequenceClassification: Bu, bir dil modelini belirli bir görev için önceden eğitilmiş bir modelle birleştirmek için kullanılır. Özellikle, bu belirli bir metin sınıflandırma görevi için eğitilmiş bir modeli temsil eder. Bu sınıf, önceden eğitilmiş bir dil modelini (örneğin BERT, GPT-2, vb.) belirli bir sınıflandırma görevi için uyarlamak için kullanılır. Örneğin, belirli bir dilde metinlerin pozitif veya negatif olup olmadığını sınıflandırmak gibi.
- - AdamW: Bu, bir optimizasyon algoritması olan AdamW'yi temsil eder. AdamW, eğitim sırasında modelin ağırlıklarını güncellemek için kullanılır. Bu, gradient iniş tabanlı eğitim algoritmalarından biridir.
- - AutoConfig: Bu, bir modelin yapılandırma ayarlarını yüklemek için kullanılır. Özellikle, önceden eğitilmiş bir modelin yapılandırmasını almak için kullanılır. Bu ayarlar, modelin hiperparametreleri, katman sayısı, giriş boyutları vb. gibi önemli bilgileri içerir.



