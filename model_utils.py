"""
model_utils.py
--------------
Bu dosya, projenin "beyni" olan Evrişimli Sinir Ağı (CNN) mimarisini ve
ortak kullanılan yardımcı fonksiyonları içerir. Hem eğitim (train.py) hem de
tahmin (predict.py, app.py) dosyaları buradaki fonksiyonları çağırır.

Böylece kod tekrarını önler ve "tek bir doğruluk kaynağı" sağlarız.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# CIFAR-10 veri setindeki 10 sınıfın isimleri (sıra önemlidir, etiket 0..9).
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "ucak",      # 0 - airplane
    "otomobil",  # 1 - automobile
    "kus",       # 2 - bird
    "kedi",      # 3 - cat
    "geyik",     # 4 - deer
    "kopek",     # 5 - dog
    "kurbaga",   # 6 - frog
    "at",        # 7 - horse
    "gemi",      # 8 - ship
    "kamyon",    # 9 - truck
]

# Görüntü boyutu (CIFAR-10 görselleri 32x32 piksel, 3 renk kanalı - RGB)
IMG_SIZE = 32
NUM_CLASSES = len(CLASS_NAMES)

# Eğitilmiş modelin kaydedileceği/okunacağı dosya yolu
MODEL_PATH = "cifar10_cnn_model.keras"


def build_model():
    """
    Sıfırdan bir CNN (Evrişimli Sinir Ağı) mimarisi oluşturur ve derler.

    Mimari mantığı:
      - Görüntü piksellerini 0-1 aralığına ölçekleriz (Rescaling).
      - 3 evrişim bloğu (Conv2D + BatchNormalization + MaxPooling) ile
        görüntüdeki kenar, doku ve şekil gibi özellikleri kademeli olarak
        çıkarırız. Derinleştikçe filtre sayısı artar (32 -> 64 -> 128).
      - Dropout katmanları aşırı öğrenmeyi (overfitting) azaltır.
      - Son katmanda 10 nöron + softmax ile her sınıf için olasılık üretiriz.

    Returns:
        Derlenmiş (compiled) tf.keras modeli.
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Pikselleri [0,255] -> [0,1] aralığına normalize et
        layers.Rescaling(1.0 / 255),

        # --- 1. Evrişim Bloğu ---
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # --- 2. Evrişim Bloğu ---
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        # --- 3. Evrişim Bloğu ---
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.40),

        # --- Sınıflandırıcı (Tam Bağlı Katmanlar) ---
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # etiketler tamsayı (0..9)
        metrics=["accuracy"],
    )
    return model


def preprocess_image(pil_image):
    """
    Kullanıcının yüklediği herhangi bir görseli (PIL Image) modele uygun
    hale getirir: RGB'ye çevirir, 32x32 boyutuna küçültür ve modelin
    beklediği (1, 32, 32, 3) şeklindeki diziye dönüştürür.

    Not: Piksel normalizasyonu (0-1) modelin İLK katmanında (Rescaling)
    yapıldığı için burada tekrar bölmüyoruz.

    Args:
        pil_image: PIL.Image nesnesi.

    Returns:
        numpy dizisi, şekil = (1, 32, 32, 3).
    """
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # tek görsel -> batch boyutu ekle
    return arr


def load_trained_model(model_path=MODEL_PATH):
    """
    Diske kaydedilmiş eğitilmiş modeli yükler.

    Args:
        model_path: .keras model dosyasının yolu.

    Returns:
        Yüklenmiş tf.keras modeli.
    """
    return tf.keras.models.load_model(model_path)


def predict_top_k(model, pil_image, k=3):
    """
    Bir görsel için en yüksek olasılıklı k tahmini döndürür.

    Returns:
        [(sinif_adi, olasilik), ...] şeklinde, olasılığa göre azalan liste.
    """
    arr = preprocess_image(pil_image)
    probs = model.predict(arr, verbose=0)[0]  # şekil = (10,)
    top_idx = np.argsort(probs)[::-1][:k]
    return [(CLASS_NAMES[i], float(probs[i])) for i in top_idx]
