import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load model
if os.path.exists("model.keras"):
    model = load_model("model.keras")
else:
    st.error("❌ File model.keras tidak ditemukan.")
    st.stop()

# Daftar label prediksi (ganti sesuai output model)
class_labels = ['cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic']

# Judul aplikasi
st.title("Klasifikasi Sampah Berbasis Gambar")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar sampah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)[0]
    confidence = np.max(prediction)
    predicted_class = class_labels[np.argmax(prediction)]

    # Threshold
    if confidence < 0.3:
        st.warning("Model tidak yakin terhadap kelas gambar ini.")
    else:
        st.success(f"Prediksi: {predicted_class} ({confidence*100:.2f}% yakin)")
