"""
demo_tahmin.py
--------------
Eğitilmiş modeli kullanarak CIFAR-10 test setinden rastgele örnekler seçer,
her biri için tahmini + gerçek etiketi + güven oranını gösteren bir özet
görsel üretir. Rapora "modelin çalıştığını kanıtlayan" görsel olarak eklenebilir.

Çalıştırma:
    python demo_tahmin.py
Çıktı:
    tahmin_ornekleri.png
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

from model_utils import load_trained_model, CLASS_NAMES, MODEL_PATH

# Kaç örnek gösterilecek (3 satır x 5 sütun)
ROWS, COLS = 3, 5
N = ROWS * COLS


def main():
    print(">> Model ve test verisi yükleniyor...")
    model = load_trained_model()
    (_, _), (x_test, y_test) = datasets.cifar10.load_data()
    y_test = y_test.flatten()

    # Rastgele N örnek seç (tekrar üretilebilir olsun diye sabit tohum)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_test), size=N, replace=False)

    # Tahminler (model girişi ham 32x32 RGB; normalizasyon model içinde)
    preds = model.predict(x_test[idx], verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    pred_conf = np.max(preds, axis=1)

    fig, axes = plt.subplots(ROWS, COLS, figsize=(13, 8))
    for ax, i, p, c in zip(axes.flat, idx, pred_labels, pred_conf):
        ax.imshow(x_test[i])
        ax.axis("off")
        true_label = y_test[i]
        dogru = (p == true_label)
        renk = "green" if dogru else "red"
        isaret = "DOGRU" if dogru else "YANLIS"
        ax.set_title(
            f"Tahmin: {CLASS_NAMES[p]} (%{c*100:.0f})\n"
            f"Gercek: {CLASS_NAMES[true_label]} [{isaret}]",
            color=renk, fontsize=9,
        )

    dogru_sayisi = int(np.sum(pred_labels == y_test[idx]))
    fig.suptitle(
        f"Model Tahmin Ornekleri  ({dogru_sayisi}/{N} dogru)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("tahmin_ornekleri.png", dpi=150)
    print(f">> Kaydedildi: tahmin_ornekleri.png  ({dogru_sayisi}/{N} dogru tahmin)")


if __name__ == "__main__":
    main()
