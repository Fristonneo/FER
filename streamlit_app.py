import streamlit as st
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import cv2
import numpy as np
import os
import gdown

# === Konfigurasi model ===
GDRIVE_MODEL_ID = "1UmP6NdpNzl7jR9fROOB11bX5o88WoFRV"  # Ganti dengan ID Google Drive kamu
MODEL_PATH = "vit_model.pt"
MODEL_NAME = "google/vit-base-patch32-224-in21k"
CLASS_NAMES = ['Anger', 'Happy', 'Neutral', 'Sad', 'Surprise']

# === Download & load model ===
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# === Preprocessing wajah ===
def preprocess(face_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return transform(face_pil).unsqueeze(0)

# === Streamlit UI ===
st.set_page_config(page_title="Ekspresi Wajah Realtime", layout="centered")
st.title("🎭 Deteksi Ekspresi Wajah Realtime dengan ViT")

run = st.toggle("🎥 Aktifkan Kamera")
frame_window = st.image([])

model = download_and_load_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Kamera tidak tersedia.")
        st.stop()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Tidak bisa membaca frame dari kamera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            input_tensor = preprocess(face_img)

            with torch.no_grad():
                outputs = model(input_tensor).logits
                probs = torch.nn.functional.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs).item()
                pred_label = CLASS_NAMES[pred_idx]
                confidence = probs[0][pred_idx].item()

            label_text = f"{pred_label} ({confidence*100:.1f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    cv2.destroyAllWindows()
else:
    st.info("Klik toggle untuk memulai kamera dan deteksi ekspresi.")
