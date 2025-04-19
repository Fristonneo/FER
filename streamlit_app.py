import streamlit as st
import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import os
import gdown

model_url = 'https://drive.google.com/uc?id=1s3pABCDEFghijkLmnopQRstu'  # Ganti ID
model_path = 'best_model.pt'

if not os.path.exists(model_path):
    with st.spinner("📦 Mengunduh model... tunggu sebentar"):
        gdown.download(model_url, model_path, quiet=False)
        st.success("✅ Model berhasil diunduh!")

# Load model
model_path = 'best_model.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
model.eval()

# Class names (ganti sesuai label aslimu)
class_names = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# Preprocessing image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

st.title("Real-time Ekspresi Wajah Detector 😁😡😢")

run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Camera not found")
        break

    # Convert ke RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, 1).item()
        label = class_names[pred]

    # Tampilkan prediksi
    cv2.putText(img, f"Ekspresi: {label}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    FRAME_WINDOW.image(img)

else:
    st.write('Stopped')
    camera.release()
