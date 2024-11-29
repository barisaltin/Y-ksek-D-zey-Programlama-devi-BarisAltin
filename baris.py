import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# 1. Veri Seti Yükleme ve Keşif
DATA_PATH = r'C:\Users\90506\Downloads\digit-recognizer'
train_data_path = os.path.join(DATA_PATH, 'train.csv')
test_data_path = os.path.join(DATA_PATH, 'test.csv')

# Veriyi yükleme
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

print("Eğitim Verisi Şekli:", train_data.shape)
print("Test Verisi Şekli:", test_data.shape)

# Sınıf Dağılımını İnceleme
label_counts = train_data['label'].value_counts()
label_counts.sort_index().plot(kind='bar', title="Sınıf Dağılımı")
plt.show()

# 2. Veri Hazırlığı
X_train = train_data.iloc[:, 1:].values.astype('float32') / 255.0  # Normalizasyon
y_train = train_data.iloc[:, 0].values
X_test = test_data.values.astype('float32') / 255.0

# Görselleri düzleştirme ve yeniden boyutlandırma
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Eğitim ve Doğrulama Verilerini Ayırma
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 3. Model Tasarımı
# Basit MLP Modeli
def create_mlp_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Daha Gelişmiş CNN Modeli
def create_cnn_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Model Seçimi
model_type = "cnn"  # 'mlp' veya 'cnn' seçebilirsiniz.
if model_type == "mlp":
    model = create_mlp_model()
else:
    model = create_cnn_model()

# 4. Model Eğitimi
history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

# 5. Model Değerlendirme
val_predictions = model.predict(X_val).argmax(axis=1)
print("Doğruluk Skoru:")
print(classification_report(y_val, val_predictions))

# Görsellerle Tahmin Analizi
import random

# Rastgele örnekler seçelim
num_samples = 10  # Kaç örnek gösterileceği
random_indices = random.sample(range(X_val.shape[0]), num_samples)

fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
for i, ax in enumerate(axes):
    idx = random_indices[i]
    img = X_val[idx].reshape(28, 28)  # Görseli 28x28 boyutunda göster
    true_label = y_val[idx]
    predicted_label = val_predictions[idx]
    
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(f"T: {true_label}\nP: {predicted_label}", fontsize=12)

plt.suptitle("Tahminler ve Gerçek Etiketler", fontsize=16)
plt.tight_layout()
plt.show()

# 6. Model Kaydetme
MODEL_PATH = './models/'
os.makedirs(MODEL_PATH, exist_ok=True)
model.save(os.path.join(MODEL_PATH, f"{model_type}_model.h5"))

print(f"Model '{model_type}' başarıyla kaydedildi.")

# 7. Sonuçların Görselleştirilmesi
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluk')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kayıp')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()
