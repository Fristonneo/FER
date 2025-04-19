import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import gdown

# --- Unduh model jika belum ada ---
model_url = 'https://drive.google.com/uc?id=1UmP6NdpNzl7jR9fROOB11bX5o88WoFRV'
model_path = 'best_model.pt'

if not os.path.exists(model_path):
    with st.spinner("📦 Mengunduh model... tunggu sebentar..."):
        gdown.download(model_url, model_path, quiet=False)
        st.success("✅ Model berhasil diunduh!")

# --- Load model TorchScript ---
device = torch.device("cpu")
model = torch.jit.load(model_path, map_location=device)
model.eval()

class_names = ['angry', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

st.set_page_config(page_title="Ekspresi Webcam Cloud", layout="centered")
st.title("🧠 Deteksi Ekspresi Wajah (Versi Cloud Compatible)")

st.write("Ambil gambar wajahmu dengan kamera, sistem akan deteksi ekspresimu.")

img_file = st.camera_input("📷 Klik untuk mengambil gambar")

if img_file:
    image = Image.open(img_file)
    st.image(image, caption="📸 Gambar diterima", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        label = class_names[pred_idx]

    st.success(f"🎭 Ekspresi yang terdeteksi: **{label.upper()}**")

    # Otomatis minta ambil ulang
    st.experimental_rerun()
