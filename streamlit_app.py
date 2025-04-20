import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
import gdown

# === CONFIG ===
GDRIVE_MODEL_ID = "1AbCdEfGhIjKlMnOpQrSt"  # Ganti dengan ID model kamu
MODEL_PATH = "vit_model.pt"
LABELS = ['Marah', 'Senang', 'Netral', 'Sedih', 'Kaget']

# === Fungsi: Unduh model jika belum ada ===
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

# === Fungsi: Preprocessing gambar wajah ===
def preprocess_face(face_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return transform(img).unsqueeze(0)

# === Fungsi: Prediksi ekspresi wajah ===
def predict_expression(model, face_tensor):
    with torch.no_grad():
        output = model(face_tensor)
        pred = output.argmax(dim=1).item()
    return LABELS[pred]

# === UI Streamlit ===
st.set_page_config(layout="centered")
st.title("🎭 Deteksi Ekspresi Wajah Realtime")

run = st.toggle("🎥 Aktifkan Kamera")
model = download_and_load_model()
frame_window = st.image([])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Kamera tidak tersedia.")
        st.stop()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Gagal membaca frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue
            try:
                face_tensor = preprocess_face(face)
                label = predict_expression(model, face_tensor)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                print("Error prediksi:", e)

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    cv2.destroyAllWindows()
else:
    st.info("Aktifkan toggle di atas untuk mulai kamera.")
