# Bu program CIFAR-10 veri kümesi üzerinde çalışan bir görüntü 
# sınıflandırma modeli oluşturur ve çeşitli optimizasyon 
# algoritmaları ile kayıp fonksiyonlarının model performansına 
# etkisini karşılaştırmalı olarak analiz etmeye yardımcı olur.

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# CIFAR-10 veri setini yüklenmesi
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#normalize etme işlemi
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255
#CategoricalCrossentropy kayıp fonk. için etiketlerin one-hot encode işlemi
train_labels_cat = tf.keras.utils.to_categorical(train_labels, 10)
test_labels_cat = tf.keras.utils.to_categorical(test_labels, 10)

#8 katmandan oluşan modelimizi tanımlıyoruz
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation="softmax"))

#modeli derliyoruz
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#modeli eğitiyoruz
batch_size = 128
history = model.fit(train_images, train_labels_cat, batch_size=batch_size, epochs=10, 
                    validation_data=(test_images, test_labels_cat), validation_split=0.1)
#optimizasyon ve kayıp algoritmalarını sıralı bir şekilde
#denenmesi için bir listede tutuyoruz
optimizers = ['adam', 'sgd', 'rmsprop']
losses = [tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'categorical_hinge']

results = {}
#metrikleri hesapladığımız fonksiyon
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    specificity = np.diag(cm) / np.sum(cm, axis=1)
    return accuracy, precision, recall, f1, specificity
#confusion matrix in gösterilmesini sağlayan fonk.
def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #normalizasyon
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    plt.show()
#epok başına modellerin kayıpları ve doğrulukları grafik 
#olarak göstermemizi sağlayan fonksiyon
def plot_history(history, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.xlabel('Epok')
    plt.ylabel('Kayıp')
    plt.title(f'{title} - Kayıp')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epok')
    plt.ylabel('Doğruluk')
    plt.title(f'{title} - Doğruluk')
    plt.legend()

    plt.show()

#her bir optimizasyon algoritmasını ve kayıp fonksiyonunu
#denenmesi için oluşturduğumuz döngü
for optimizer in optimizers:  
    for loss in losses:
        # kayıp fonksiyonuna göre uygun etiketi 
        # kullanmamızı sağlayan şart bloğu
        if isinstance(loss, tf.keras.losses.SparseCategoricalCrossentropy):
            y_train = train_labels
            y_test = test_labels
        else:
            y_train = train_labels_cat
            y_test = test_labels_cat
            
    
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy'])
        history = model.fit(train_images, y_train, batch_size=batch_size, epochs=10, validation_split=0.1,
                            validation_data=(test_images, y_test), verbose=0)
        
        #test tahminleri 
        y_pred = model.predict(test_images).argmax(axis=1)
        y_true = test_labels.flatten()
        
        #modelin performansının hesaplanması
        score = model.evaluate(test_images, y_test, verbose=0)
        print(f"Optimizer: {optimizer}, Loss: {loss_name} Test loss:", score[0])
        print(f"Optimizer: {optimizer}, Loss: {loss_name} Test accuracy:", score[1])
        
        #performans metriklerini hesaplayın
        metrics = calculate_metrics(y_true, y_pred)
        results[(optimizer, loss)] = metrics
        
        #confusion matrix'in gösterilmesi
        loss_name = loss.__class__.__name__ if isinstance(loss, tf.keras.losses.Loss) else loss
        title = f'Confusion Matrix - Optimizer: {optimizer}, Loss: {loss_name}'
        plot_confusion_matrix(y_true, y_pred, labels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], title=title)
        titlee = f'Optimizer: {optimizer}, Loss: {loss_name}'
        
        #eğitim ve doğrulama kaybı/doğruluğu grafiğinin gösterilmesi
        plot_history(history, titlee)

#her kombinasyon için sonuçları yazdırma
for key, value in results.items():
    print(f'Optimizer: {key[0]}, Loss: {key[1]}')
    print(f'Accuracy: {value[0]:.4f}, Precision: {value[1]:.4f}, Recall: {value[2]:.4f}, F1 Score: {value[3]:.4f}, Specificity: {np.mean(value[4]):.4f}\n')
