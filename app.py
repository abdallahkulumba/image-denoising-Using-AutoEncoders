import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# --- Define Autoencoder Model ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- Load Model ---
model_path = "autoencoder_final.pth"
if not os.path.exists(model_path):
    st.error("Model file not found! Please ensure 'autoencoder_final.pth' is in the correct directory.")
    st.stop()

try:
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Load MNIST Dataset and Apply Gaussian Noise ---
transform_clean = transforms.Compose([
    transforms.ToTensor()
])

transform_noisy = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.5 * torch.randn_like(x)),  # Add Gaussian noise
    transforms.Lambda(lambda x: torch.clamp(x, 0., 1.))  # Clamp values between 0 and 1
])

mnist_dataset_clean = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_clean)
mnist_dataset_noisy = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_noisy)

# --- Streamlit UI ---
st.title("Mnist Denoising Autoencoder")
st.write("Select a sample from the MNIST dataset, add Gaussian noise, and denoise using the trained model.")

index = st.slider("Select an MNIST image index:", 0, len(mnist_dataset_clean)-1, 0)

# Get clean and noisy image tensors
image_clean, _ = mnist_dataset_clean[index]
image_noisy, _ = mnist_dataset_noisy[index]

input_tensor = image_noisy.unsqueeze(0)  # Add batch dimension [1, 1, 28, 28]

# --- Denoising Process ---
with torch.no_grad():
    denoised_tensor = model(input_tensor)

# Convert to numpy for display
clean_img = image_clean.squeeze().numpy()
noisy_img = image_noisy.squeeze().numpy()
denoised_img = denoised_tensor.squeeze().numpy()

# --- Display Images ---
st.subheader("🔍 Input vs Noisy vs Denoised Output")
col1, col2, col3 = st.columns(3)

with col1:
    st.image(clean_img, caption="Original MNIST Image", use_column_width=True, clamp=True)

with col2:
    st.image(noisy_img, caption="Noisy Image (Gaussian Noise)", use_column_width=True, clamp=True)

with col3:
    st.image(denoised_img, caption="Denoised Output", use_column_width=True, clamp=True)

st.write("App is Running!")  # Debugging output
# Sidebar: About the Author Section
st.sidebar.write("## About the Author")
st.sidebar.write("""
**Author:** Sserujja Abdallah Kulumba  
**Affiliation:** Islamic University of Technology  
**Email:** abdallahkulumba@iut-dhaka.edu  
**GitHub:** [github.com/Abdallahkulumba](https://github.com/Abdallahkulumba)  
**LinkedIn:** [linkedin.com/in/Abdallahkulumba](https://www.linkedin.com/in/abdallah-kulumba-sserujja/)  
**Facebook:** [facebook.com/Abdallahkulumba](https://www.facebook.com/abdallah.ed.ak)  
""")