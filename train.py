"""
train.py
--------
CIFAR-10 veri seti üzerinde Evrişimli Sinir Ağı (CNN) modelini eğitir,
değerlendirir ve sonuçları kaydeder.

Çalıştırma:
    python train.py
    python train.py --epochs 50      (epoch sayısını değiştirmek için)

Üretilen çıktılar:
    - cifar10_cnn_model.keras   -> eğitilmiş model
    - egitim_grafikleri.png     -> doğruluk/kayıp eğrileri
    - confusion_matrix.png      -> karışıklık matrisi
    - siniflandirma_raporu.txt  -> precision/recall/f1 metrikleri
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from model_utils import build_model, CLASS_NAMES, MODEL_PATH


def load_data():
    """CIFAR-10 veri setini indirir/yükler ve eğitim-test olarak böler."""
    print(">> CIFAR-10 veri seti yükleniyor (ilk çalıştırmada indirilir)...")
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    # y_train şekli (50000, 1) -> (50000,) yapalım
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    print(f"   Eğitim verisi: {x_train.shape[0]} görsel")
    print(f"   Test verisi  : {x_test.shape[0]} görsel")
    return (x_train, y_train), (x_test, y_test)


def plot_history(history):
    """Eğitim sürecindeki doğruluk ve kayıp eğrilerini grafik olarak kaydeder."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Eğitim")
    axes[0].plot(history.history["val_accuracy"], label="Doğrulama")
    axes[0].set_title("Model Doğruluğu (Accuracy)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Doğruluk")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Eğitim")
    axes[1].plot(history.history["val_loss"], label="Doğrulama")
    axes[1].set_title("Model Kaybı (Loss)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Kayıp")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("egitim_grafikleri.png", dpi=150)
    print(">> Eğitim grafikleri kaydedildi: egitim_grafikleri.png")


def evaluate_model(model, x_test, y_test):
    """Test verisi üzerinde modeli değerlendirir; rapor ve confusion matrix üretir."""
    print("\n>> Model test verisi üzerinde değerlendiriliyor...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"   Test doğruluğu: {test_acc:.4f}  |  Test kaybı: {test_loss:.4f}")

    # Tahminler
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Sınıflandırma raporu (precision / recall / f1-score)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    print("\n" + report)
    with open("siniflandirma_raporu.txt", "w", encoding="utf-8") as f:
        f.write(f"Test Dogrulugu (Accuracy): {test_acc:.4f}\n")
        f.write(f"Test Kaybi (Loss): {test_loss:.4f}\n\n")
        f.write(report)
    print(">> Sınıflandırma raporu kaydedildi: siniflandirma_raporu.txt")

    # Confusion matrix (karışıklık matrisi)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Karışıklık Matrisi (Confusion Matrix)")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print(">> Karışıklık matrisi kaydedildi: confusion_matrix.png")


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN eğitimi")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Eğitim epoch sayısı (varsayılan: 40)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch boyutu (varsayılan: 64)")
    args = parser.parse_args()

    # 1) Veriyi yükle
    (x_train, y_train), (x_test, y_test) = load_data()

    # 2) Veri artırma (data augmentation): görselleri rastgele döndürüp/kaydırarak
    #    modelin genelleme yeteneğini artırır ve ezberlemeyi azaltır.
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # 3) Modeli oluştur (artırma katmanını başa ekleyerek)
    base_model = build_model()
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    outputs = base_model(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print("\n>> Model mimarisi:")
    base_model.summary()

    # 4) Eğitimi iyileştiren callback'ler
    callbacks = [
        # Doğrulama kaybı iyileşmezse erken durdur (zaman kazandırır)
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        # İyileşme durunca öğrenme oranını düşür
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]

    # 5) Eğitim
    print(f"\n>> Eğitim başlıyor ({args.epochs} epoch)...")
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,       # eğitim verisinin %10'u doğrulama için
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # 6) Modeli kaydet
    model.save(MODEL_PATH)
    print(f"\n>> Model kaydedildi: {MODEL_PATH}")

    # 7) Grafikler ve değerlendirme
    plot_history(history)
    evaluate_model(model, x_test, y_test)

    print("\n>> Tüm işlemler tamamlandı! Artık 'streamlit run app.py' ile arayüzü açabilirsin.")


if __name__ == "__main__":
    main()
