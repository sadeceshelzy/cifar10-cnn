# CIFAR-10 Görüntü Sınıflandırma (CNN)

Yapay Zeka Uygulamaları dersi projesi — CIFAR-10 veri seti üzerinde evrişimli sinir ağı (CNN)
ve transfer learning ile görüntü sınıflandırma.

## İçerik
- `train.py` — sıfırdan CNN eğitimi
- `train_transfer.py` — transfer learning ile eğitim
- `predict.py` / `demo_tahmin.py` — eğitilmiş modelle tahmin
- `app.py` — tahmin arayüzü (`baslat.bat` ile çalıştırılır)
- `model_utils.py` — ortak model yardımcıları
- `cifar10_cnn_model.keras` — eğitilmiş model
- `confusion_matrix*.png`, `egitim_grafikleri*.png`, `siniflandirma_raporu*.txt` — değerlendirme çıktıları
- `Rapor.docx`, `Efe_Osmanhan_Guducu_Sunum_1.pdf` — proje raporu ve sunum

## Kurulum
```bash
pip install -r requirements.txt
python train.py          # veya train_transfer.py
python app.py
```

## Teknolojiler
Python · TensorFlow / Keras
