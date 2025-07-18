# 🧠 CIFAR-10 Görüntü Sınıflandırması: Optimizasyon ve Kayıp Fonksiyonu Karşılaştırması

Bu proje, [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) veri seti üzerinde bir **Konvolüsyonel Sinir Ağı (CNN)** kullanarak görüntü sınıflandırması yapar. Ayrıca farklı **optimizer** ve **loss function** kombinasyonlarının modelin doğruluk ve başarım metriklerine etkisini analiz eder.

## 🧩 İçerik

- CNN tabanlı model mimarisi
- CIFAR-10 veri kümesi ile görsel sınıflandırma
- 3 farklı optimizer (`adam`, `sgd`, `rmsprop`)
- 2 farklı loss function (`SparseCategoricalCrossentropy`, `categorical_hinge`)
- Eğitim süreci görselleştirmeleri
- Karışıklık matrisi (confusion matrix) ile analiz
- Accuracy, Precision, Recall, F1-score ve Specificity metrikleri

---

## 🔧 Kullanılan Teknolojiler

- Python 3.x
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Seaborn / Matplotlib

---
