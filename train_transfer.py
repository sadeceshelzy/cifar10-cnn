"""
train_transfer.py
-----------------
TRANSFER ÖĞRENME ile görüntü sınıflandırma (kedi-köpek karışıklığını azaltmak için).

Mantık:
  1) ImageNet üzerinde önceden eğitilmiş MobileNetV2 modelini "özellik çıkarıcı"
     olarak kullanırız (ağırlıkları donduruyoruz -> tekrar eğitilmez).
  2) Her CIFAR-10 görselini 96x96'ya büyütüp MobileNetV2'den geçirerek
     1280 boyutlu bir "özellik vektörü" elde ederiz. Bunu BİR KEZ hesaplarız
     (en pahalı adım budur) -> CPU'da bile hızlı eğitim sağlar.
  3) Bu özelliklerin üzerine küçük bir sınıflandırıcı (Dense katmanlar) eğitiriz.
  4) Son olarak; girişi 32x32 alıp içeride büyüten, uçtan uca tek bir model
     birleştirip kaydederiz. Böylece mevcut app.py / predict.py değişmeden çalışır.

Çalıştırma:
    python train_transfer.py

Üretilen çıktılar:
    - cifar10_cnn_model.keras          -> (üzerine yazılır) yeni, daha güçlü model
    - egitim_grafikleri_transfer.png
    - confusion_matrix_transfer.png
    - siniflandirma_raporu_transfer.txt
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from model_utils import CLASS_NAMES, MODEL_PATH

# MobileNetV2'nin kabul ettiği görsel boyutu (32x32'yi buraya büyüteceğiz)
TARGET_SIZE = 96


def build_feature_extractor():
    """
    Girişi ham 32x32 RGB (0-255) görsel alan, içeride 96x96'ya büyütüp
    MobileNetV2 ile 1280 boyutlu özellik vektörü üreten modeli kurar.
    base_model ileride birleştirmede tekrar kullanılacağı için birlikte döner.
    """
    base_model = MobileNetV2(
        input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
        include_top=False,     # kendi sınıflandırıcımızı ekleyeceğiz
        weights="imagenet",    # önceden öğrenilmiş ağırlıklar
        pooling="avg",         # çıktıyı (1280,) vektöre indirger
    )
    base_model.trainable = False  # ağırlıkları dondur

    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Resizing(TARGET_SIZE, TARGET_SIZE)(inputs)
    # MobileNetV2 ön işleme: pikselleri [-1, 1] aralığına getir (x/127.5 - 1)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    features = base_model(x, training=False)

    feature_model = models.Model(inputs, features, name="ozellik_cikarici")
    return feature_model, base_model


def build_classifier_head():
    """Özellik vektörü (1280) üzerine eğitilecek küçük sınıflandırıcı."""
    head = models.Sequential([
        layers.Input(shape=(1280,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ], name="siniflandirici")
    head.compile(optimizer="adam",
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
    return head


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Eğitim")
    axes[0].plot(history.history["val_accuracy"], label="Doğrulama")
    axes[0].set_title("Doğruluk (Transfer Öğrenme)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Doğruluk")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Eğitim")
    axes[1].plot(history.history["val_loss"], label="Doğrulama")
    axes[1].set_title("Kayıp (Transfer Öğrenme)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Kayıp")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("egitim_grafikleri_transfer.png", dpi=150)
    print(">> Grafikler kaydedildi: egitim_grafikleri_transfer.png")


def main():
    # 1) Veri
    print(">> CIFAR-10 yükleniyor...")
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # 2) Özellik çıkarıcıyı kur
    print(">> MobileNetV2 yükleniyor (ilk çalıştırmada ağırlıklar inecek)...")
    feature_model, base_model = build_feature_extractor()

    # 3) Özellikleri BİR KEZ hesapla (en uzun adım; CPU'da birkaç dakika sürebilir)
    print(">> Eğitim görsellerinden özellik çıkarılıyor...")
    train_feats = feature_model.predict(x_train, batch_size=64, verbose=1)
    print(">> Test görsellerinden özellik çıkarılıyor...")
    test_feats = feature_model.predict(x_test, batch_size=64, verbose=1)

    # 4) Sınıflandırıcıyı eğit (özellikler hazır olduğu için çok hızlı)
    head = build_classifier_head()
    print("\n>> Sınıflandırıcı eğitiliyor...")
    history = head.fit(
        train_feats, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=64,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5,
                                 restore_best_weights=True)],
        verbose=1,
    )

    # 5) Uçtan uca modeli birleştir (giriş 32x32 -> tahmin)
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Resizing(TARGET_SIZE, TARGET_SIZE)(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base_model(x, training=False)
    outputs = head(x)
    final_model = models.Model(inputs, outputs, name="transfer_cifar10")
    final_model.save(MODEL_PATH)
    print(f"\n>> Yeni model kaydedildi: {MODEL_PATH}")

    # 6) Değerlendirme
    print("\n>> Test verisi üzerinde değerlendiriliyor...")
    y_pred = np.argmax(head.predict(test_feats, verbose=0), axis=1)
    test_acc = np.mean(y_pred == y_test)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    print(f"\nTest Doğruluğu: {test_acc:.4f}\n")
    print(report)
    with open("siniflandirma_raporu_transfer.txt", "w", encoding="utf-8") as f:
        f.write(f"Test Dogrulugu (Accuracy): {test_acc:.4f}\n\n")
        f.write(report)
    print(">> Rapor kaydedildi: siniflandirma_raporu_transfer.txt")

    # 7) Karışıklık matrisi
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Karışıklık Matrisi (Transfer Öğrenme)")
    plt.xlabel("Tahmin Edilen Sınıf"); plt.ylabel("Gerçek Sınıf")
    plt.xticks(rotation=45); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_transfer.png", dpi=150)
    print(">> Karışıklık matrisi kaydedildi: confusion_matrix_transfer.png")

    plot_history(history)
    print("\n>> Bitti! 'streamlit run app.py' ile yeni modeli deneyebilirsin.")


if __name__ == "__main__":
    main()
