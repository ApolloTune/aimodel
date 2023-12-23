# Model Description(EN)
This project uses a specially trained language model for tasks such as sentiment analysis on Turkish lyrics. The model is built by tweaking 'bert-base-turkish-cased', a pre-trained BERT model called BERTurk developed using PyTorch and Huggingface Transformers libraries. Since this model is specially trained to better understand Turkish language structure and culture, it aims to produce more effective results on Turkish lyrics.

BERT (Bidirectional Encoder Representations from Transformers) is a language model architecture known for its ability to understand prior vocabulary and context information. 'bert-base-turkish-cased' is pre-trained as a large language model on texts in the Turkish language. By focusing on general language understanding capabilities, this model allows for fine-tuning for specific tasks, such as Turkish lyrics.

The PyTorch and Huggingface Transformers libraries are powerful tools used to facilitate the training and use of the model. The model is designed for use in music-related tasks such as sentiment analysis, and specifically aims to provide optimal performance for understanding and labeling emotional content in Turkish lyrics.

The 10-fold cross-validation method was used for training the model. This method divides the dataset into 10 equal parts and generates 10 different training-test combinations, using each part as test data and the other 9 parts as training data in turn. Each combination helps us to evaluate the overall performance of the model more reliably. Since this process takes place on different subset combinations of the training data, it increases the generalization ability of the model and prevents overfitting. That is, the performance of the model is evaluated by averaging over 10 different test sets, which allows the model to provide a more reliable and overall performance.
# Model Açıklaması(TR)
Bu proje, Türkçe şarkı sözleri üzerinde duygu analizi gibi görevler için özel olarak eğitilmiş bir dil modeli kullanmaktadır. Model, PyTorch ve Huggingface Transformers kütüphaneleri kullanılarak geliştirilen BERTurk isimli önceden eğitilmiş bir BERT modeli olan 'bert-base-turkish-cased' üzerine ince ayar yapılarak oluşturulmuştur. Bu model, Türkçe dil yapısını ve kültürünü daha iyi anlamak üzere özel olarak eğitildiği için Türkçe şarkı sözleri üzerinde daha etkili sonuçlar üretmeyi amaçlamaktadır.

BERT (Bidirectional Encoder Representations from Transformers), önceki kelime ve bağlam bilgilerini anlama yeteneği ile bilinen bir dil modeli mimarisidir. 'bert-base-turkish-cased', Türkçe dilindeki metinler üzerinde geniş bir dil modeli olarak önceden eğitilmiştir. Bu model, genel dil anlama yetenekleri üzerine odaklanarak, Türkçe şarkı sözleri gibi belirli görevler için ince ayar yapılmasına olanak tanır.

PyTorch ve Huggingface Transformers kütüphaneleri, modelin eğitimi ve kullanımını kolaylaştırmak için kullanılan güçlü araçlardır. Bu model, duygu analizi gibi müzikle ilgili görevlerde kullanılmak üzere tasarlanmış olup, özellikle Türkçe şarkı sözleri üzerinde duygusal içeriği anlamak ve etiketlemek için optimal performans sağlamayı hedeflemektedir.

Modelin eğitimi için 10 katlı çapraz doğrulama yöntemi kullanılmıştır. Bu yöntem, veri setini 10 eşit parçaya böler ve her bir parçayı sırayla test verisi olarak kullanırken diğer 9 parçayı eğitim verisi olarak kullanarak 10 farklı eğitim-test kombinasyonu oluşturur. Her bir kombinasyon, modelin genel performansını daha güvenilir bir şekilde değerlendirmemize yardımcı olur. Bu süreç, eğitim verisinin farklı alt küme kombinasyonları üzerinde gerçekleştiği için modelin genelleme yeteneğini arttırır ve aşırı uydurmayı önler. Yani, modelin performansı, 10 farklı test seti üzerinde ortalaması alınarak değerlendirilmiştir, bu da modelin daha güvenilir ve genel bir performans sunmasını sağlar.
# Model Performance
Training Fold 1/10
Accuracy for Fold 1: 0.6875
Training Fold 2/10
Accuracy for Fold 2: 0.8
Training Fold 3/10
Accuracy for Fold 3: 0.95
Training Fold 4/10
Accuracy for Fold 4: 0.975
Training Fold 5/10
Accuracy for Fold 5: 0.9875
Training Fold 6/10
Accuracy for Fold 6: 1.0
Training Fold 7/10
Accuracy for Fold 7: 0.975
Training Fold 8/10
Accuracy for Fold 8: 1.0
Training Fold 9/10
Accuracy for Fold 9: 1.0
Training Fold 9/10
Accuracy for Fold 10: 0.975
Avarege Accuracy: 0.93499999999