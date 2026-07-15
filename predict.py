"""
predict.py
----------
Eğitilmiş model ile komut satırından tek bir görselin sınıfını tahmin eder.
(Arayüz kullanmadan, hızlı test için.)

Çalıştırma:
    python predict.py resim.jpg
"""

import sys
import os
from PIL import Image

from model_utils import load_trained_model, predict_top_k, MODEL_PATH


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python predict.py <resim_dosyasi>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"HATA: '{image_path}' bulunamadı.")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Eğitilmiş model ({MODEL_PATH}) bulunamadı.")
        print("Önce 'python train.py' komutuyla modeli eğitmelisin.")
        sys.exit(1)

    print(">> Model yükleniyor...")
    model = load_trained_model()

    image = Image.open(image_path)
    results = predict_top_k(model, image, k=3)

    print(f"\n'{image_path}' için tahminler:")
    print("-" * 35)
    for i, (label, prob) in enumerate(results, start=1):
        print(f"  {i}. {label:<10} -> %{prob * 100:.2f}")
    print("-" * 35)
    print(f"SONUÇ: Bu görsel büyük olasılıkla '{results[0][0]}'")


if __name__ == "__main__":
    main()
