"""
app.py
------
Streamlit tabanlı web arayüzü. Kullanıcı bir görsel yükler, model görseli
10 CIFAR-10 sınıfından birine sınıflandırır ve olasılıkları gösterir.

Çalıştırma:
    streamlit run app.py
"""

import os
import streamlit as st
from PIL import Image
import pandas as pd

from model_utils import load_trained_model, predict_top_k, CLASS_NAMES, MODEL_PATH

# ---------------------------------------------------------------------------
# Sayfa ayarları
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Görüntü Sınıflandırma", page_icon="🖼️", layout="centered")


# ---------------------------------------------------------------------------
# Modeli yalnızca BİR KEZ yükle (cache). Her tahminte yeniden yüklenmesini önler.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_trained_model()


def main():
    st.title("🖼️ Görüntü Sınıflandırma Sistemi")
    st.markdown(
        "Bu uygulama, **Evrişimli Sinir Ağı (CNN)** kullanarak yüklediğiniz görseli "
        "aşağıdaki **10 sınıftan** birine sınıflandırır."
    )

    # Sınıf listesini göster
    with st.expander("ℹ️ Modelin tanıyabildiği sınıflar"):
        st.write(", ".join(CLASS_NAMES))
        st.caption(
            "Not: Model 32x32 piksellik CIFAR-10 görselleriyle eğitildi. "
            "En iyi sonuç için bu sınıflara ait net görseller yükleyin."
        )

    model = get_model()
    if model is None:
        st.error(
            f"Eğitilmiş model ({MODEL_PATH}) bulunamadı. "
            "Lütfen önce terminalde `python train.py` komutunu çalıştırın."
        )
        st.stop()

    # Dosya yükleyici
    uploaded = st.file_uploader(
        "Bir görsel seçin (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is not None:
        image = Image.open(uploaded)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Yüklenen Görsel", use_container_width=True)

        with col2:
            with st.spinner("Tahmin yapılıyor..."):
                results = predict_top_k(model, image, k=3)

            best_label, best_prob = results[0]
            st.success(f"**Tahmin: {best_label}**")
            st.metric("Güven Oranı", f"%{best_prob * 100:.1f}")

        # En yüksek 3 tahmini çubuk grafik olarak göster
        st.subheader("En Olası 3 Sınıf")
        df = pd.DataFrame(
            {"Sınıf": [r[0] for r in results],
             "Olasılık": [r[1] for r in results]}
        ).set_index("Sınıf")
        st.bar_chart(df)

        # Detaylı liste
        for label, prob in results:
            st.write(f"- **{label}**: %{prob * 100:.2f}")


if __name__ == "__main__":
    main()
