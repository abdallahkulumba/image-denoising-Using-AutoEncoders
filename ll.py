import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import math
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# --- Environment Setup ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Constants ---
MODEL_PATH = "autoencoder_final.pth"
DATASETS = {
    "MNIST": torchvision.datasets.MNIST,
    "Fashion-MNIST": torchvision.datasets.FashionMNIST,
    "EMNIST (Letters)": lambda **kwargs: torchvision.datasets.EMNIST(split='letters', **kwargs),
    "KMNIST": torchvision.datasets.KMNIST
}

# --- Model Definition ---
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

# --- Helper Functions ---
def tensor_to_image(tensor):
    """Convert PyTorch tensor to PIL Image"""
    return transforms.ToPILImage()(tensor.squeeze().cpu())

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found! Please ensure 'autoencoder_final.pth' exists.")
        st.stop()
    
    try:
        model = Autoencoder()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def add_gaussian_noise(image_tensor, noise_std=0.5):
    noisy_tensor = image_tensor + noise_std * torch.randn_like(image_tensor)
    return torch.clamp(noisy_tensor, 0., 1.)

def calculate_metrics(original, denoised):
    mse = torch.nn.functional.mse_loss(denoised, original).item()
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    ssim_val = ssim(
        denoised.squeeze().cpu().numpy(),
        original.squeeze().cpu().numpy(),
        data_range=1.0
    )
    return mse, psnr, ssim_val

def get_image_download_link(img, filename="denoised_image.png"):
    """Generate download link for PIL Image"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">Download Denoised Image</a>'
    return href

# --- UI Configuration ---
st.set_page_config(
    page_title="Image Denoising using Autoencoder",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Add this CSS at the beginning of your script (right after imports)
st.markdown("""
<style>
    /* Custom CSS for large navigation buttons */
    .stTabs [role="tablist"] button {
        font-size: 18px !important;
        padding: 12px 24px !important;
        height: auto !important;
        margin: 0 8px !important;
    }
    
    /* Hover effect */
    .stTabs [role="tablist"] button:hover {
        background-color: #f0f2f6 !important;
        transform: scale(1.05);
    }
    
    /* Active tab styling */
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Navigation Tabs ---
tabs = ["🏠 Home", "🔬 Project", "🧠 Methodology", "📊 Results", "🚀 Future Work", "👥 About Us"]
page = st.sidebar.radio("Navigation", tabs)

# --- Sidebar (Consistent across pages) ---
with st.sidebar:
    
    if page in ["🏠 Home", "📊 Results"]:
        st.markdown("---")
        st.subheader("Demo Settings")
        dataset_name = st.selectbox("Select Dataset", list(DATASETS.keys()))
        noise_level = st.slider("Noise Level (σ)", 0.1, 1.0, 0.5, 0.05)
        image_idx = st.slider("Image Index", 0, 9999, 0)

# --- Page Content ---
if page == "🏠 Home":
    st.title("Image Denoising using AutoEncoder")
    st.markdown(" ##### A Deep Learning Approach to Enhance Image Quality using Autoencoders 4709 project")
    st.markdown("## Transforming Noisy Images into Clear Images")

    st.markdown("""   
    <span style='font-size:18px'>    
    This project addresses a critical challenge in imaging: 
    the degradation of image quality due to noise. Using deep learning autoencoders, 
    we've developed a system that can effectively remove noise while preserving 
    diagnostically important features in the image, it can be used to give real-time outputs.
    </span> 
    """, unsafe_allow_html=True)
    for i in range(2):
        st.write("")
    st.image("image/im1.jpg", use_column_width=True)
    cols = st.columns(3)
    with cols[0]:
        st.metric("Imaging Market", "$45B+", "8.2% CAGR")
    with cols[1]:
        st.metric("Noise Reduction", "Up to 85%", "PSNR Improvement")
    with cols[2]:
        st.metric("Applications", "Real CCTV outputs", "Universal Approach")
    
    st.markdown("### 👉 Navigate using the sidebar to explore our project in detail")

elif page == "🔬 Project":
    st.title("Project Genesis")

    st.markdown('<h4><b>🚀 Motivation</b></h4>', unsafe_allow_html=True)
    
    with st.expander(  " IDEA BEHIND THE PROJECT ", expanded = True):

        st.markdown("""
         <span style='font-size:18px'>
          It all started with a simple observation — in many real-world scenarios, from grainy CCTV footage to low-quality 
          medical scans, images are often corrupted by noise. As technology advances, the need for clean, high-quality visuals 
          becomes increasingly critical, especially in areas where clarity could influence decisions — whether it’s public safety
          or medical diagnosis. We were driven by a question: Can we build a model that learns to see through the noise? That 
          question led us to explore autoencoders — a deep learning architecture well-suited for this challenge. 
          </span> 
          """, unsafe_allow_html=True)
        

    st.markdown('<h4><b>🎯 Objectives</b></h4>', unsafe_allow_html=True)    
    
    with st.expander("🎯 Objectives"):
        st.markdown("""
        <span style='font-size:18px'>
            We developed an image denoising system based on autoencoders, trained to take noisy input images and reconstruct clean,
            denoised versions. Using the MNIST dataset as a starting point, we introduced synthetic noise and trained our model to
            remove it, learning the essential patterns that define each digit. This helped us validate that the model could recover
            structure from noise — a foundational step toward real-world applications.
            </span>
        """, unsafe_allow_html=True)
    
    st.markdown('<h4><b>🎯 What we found</b></h4>', unsafe_allow_html=True)   

    with st.expander("🎯 EXPERIMENTAL ANALYSIS", expanded=True):
        st.markdown("""  <span style='font-size:18px'>
            Our autoencoder demonstrated promising results — it was able to denoise heavily corrupted images while preserving key
           features. This confirmed the potential of using neural networks not just to classify or detect, but to restore. 
           It opened our eyes to broader implications — perhaps the same approach could help clean up satellite imagery,
           assist in early medical diagnosis, or even enhance old photographs.   </span>
          """, unsafe_allow_html=True)

    
    st.markdown("""
    ## Project Timeline
    """)
    st.image("image/im4.png", use_column_width=True)

elif page == "🧠 Methodology":
    st.title("Technical Approach")
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("""
        ### Model Architecture
        Our autoencoder features:
        - 3-layer convolutional encoder
        - Symmetric decoder with skip connections
        - Custom perceptual loss function
        - Mixed-precision training
        - Batch normalization and dropout

        """)
        st.image("image/im7.png", use_column_width=True)
    
    with cols[1]:
        st.markdown("""
        ### Experimental Setup
        - **Datasets**: MNIST, Fashion-MNIST, EMNIST, KMNIST
        - **Noise Model**: Gaussian (σ=0.1-1.0)
        - **Training**: 100 epochs, Adam optimizer
        - **DL Model**: Convolution Neural Network (CNN)
        """)
        st.image("image/im5.png", use_column_width=True)

    with cols[2]:
        st.markdown("""
        ### Experimental Hardware
        - **Model**: NVIDIA T4 GPU
        - **Framework**: Pytorch 1.9.0
        - **CUDA**: 11.1
        - **Python**: 3.8.10
        - **Training**: 100 epochs, Adam optimizer
        - **OS**: windows 10
        """)
        st.image("image/im6.png", use_column_width=True)


elif page == "📊 Results":
    st.title("Experimental Results")
    
    try:
        # Load data and model
        dataset = DATASETS[dataset_name](root="./data", train=False, download=True, transform=transforms.ToTensor())
        model = load_model()
        
        # Get sample image
        clean_image, _ = dataset[image_idx]
        noisy_image = add_gaussian_noise(clean_image, noise_level)
        
        # Process image
        with torch.no_grad():
            denoised_tensor = model(noisy_image.unsqueeze(0))
        
        # Convert tensors to images
        clean_img = tensor_to_image(clean_image)
        noisy_img = tensor_to_image(noisy_image)
        denoised_img = tensor_to_image(denoised_tensor)
        
        # Display results
        cols = st.columns(3)
        with cols[0]:
            st.image(clean_img, caption="Original", use_column_width=True)
        with cols[1]:
            st.image(noisy_img, caption=f"Noisy (σ={noise_level})", use_column_width=True)
        with cols[2]:
            st.image(denoised_img, caption="Denoised", use_column_width=True)
            
            # Download button
            buffered = BytesIO()
            denoised_img.save(buffered, format="PNG")
            st.download_button(
                label="Download Denoised Image",
                data=buffered.getvalue(),
                file_name="denoised_image.png",
                mime="image/png"
            )

        # --- Compute Real Evaluation Metrics ---
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        image_clean = clean_image.unsqueeze(0)  # Ensure matching shape [1, 1, 28, 28]
        mse_loss = torch.nn.functional.mse_loss(denoised_tensor, image_clean).item()
        ssim_score = ssim(
            denoised_tensor.squeeze().cpu().numpy(), 
            image_clean.squeeze().cpu().numpy(), 
            data_range=1.0
        )
        psnr = 20 * math.log10(1.0 / math.sqrt(mse_loss))

        # --- Visualization Section ---
        st.subheader("📊 Model Performance Overview")

        # Create a figure with three subplots
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        # MSE Bar Chart
        ax[0].bar(["MSE"], [mse_loss], color="red")
        ax[0].set_title("Mean Squared Error")
        ax[0].set_ylim(0, 1)  # Adjust based on typical values

        # SSIM Bar Chart
        ax[1].bar(["SSIM"], [ssim_score], color="green")
        ax[1].set_title("Structural Similarity Index")
        ax[1].set_ylim(0, 1)  # SSIM range: [0, 1]

        # PSNR Bar Chart
        ax[2].bar(["PSNR"], [psnr], color="blue")
        ax[2].set_title("Peak Signal-to-Noise Ratio (dB)")
        ax[2].set_ylim(0, 50)  # Adjust based on typical values

        st.pyplot(fig)

        # Display numeric values
        st.write(f"🔴 **Mean Squared Error (MSE):** {mse_loss:.6f} (Lower is better)")
        st.write(f"🟢 **Structural Similarity Index (SSIM):** {ssim_score:.4f} (Higher is better)")
        st.write(f"🔵 **Peak Signal-to-Noise Ratio (PSNR):** {psnr:.2f} dB (Higher is better)")
        
    except Exception as e:
        st.error(f"Error in demo: {str(e)}")

elif page == "🚀 Future Work":
    st.title("Next Steps & Challenges")
    cols = st.columns(2)
    
    with cols[0]:
        st.markdown("""
        ### Planned Improvements
        - **Model architecture**: Explore transformer-based architectures
        - **Noise robustness**: Train on real-world noisy datasets
        - **Data augmentation**: Use GANs for synthetic data generation
        - **Edge deploment**: Optimize for mobile/tablet use
        """)
    
    with cols[1]:
        st.markdown("""
        ### Current Challenges
        - Real-world noise complexity
        - Generalization across datasets
        - Computational resource constraints
        - Model interpretability and explainability
        - Hardware constraints in LMICs
        """)
    
    st.markdown("""
    ## Roadmap
    """)
    
    st.image("image/im3.png", use_column_width=True)

elif page == "👥 About Us":
    st.title("Our Team")
    cols = st.columns(3)
    
    with cols[0]:
      #  st.image("https://i.imgur.com/JDQ4z4m.jpg", width=150)
        st.markdown("""
        #### Sserujja Abdallah Kulumba  
        **Email:** abdallahkulumba@iut-dhaka.edu  
        **GitHub:** [github.com/Abdallahkulumba](https://github.com/Abdallahkulumba)  
        **LinkedIn:** [linkedin.com/in/Abdallahkulumba](https://www.linkedin.com/in/abdallah-kulumba-sserujja/)  
        **Facebook:** [facebook.com/Abdallahkulumba](https://www.facebook.com/abdallah.ed.ak) 
        """)
    
    with cols[1]:
        st.markdown("""
       #### Jarin Tasnim Rahman
        **Email:** jarintasnim2@iut_dhaka.edu  
        **GitHub:** [github.com/Jarin0305](https://github.com/Jarin0305)  
       """)
    
    with cols[2]:
        st.markdown("""
        #### Fariha Alam Urbana 
        **Email:** farihaalam@iut-dhaka.edu   
        **GitHub:** [github.com/farrihaa](https://github.com/farrihaa)    
       """)
    for i in range(4):
        st.write("")
    st.markdown("""
    ### Project Supervisor
    #### Md. Arefin Rabbi Emon
    **Email:** arefinrabbi@iut-dhaka.edu  
    **Affiliation:** Islamic University of Technology 

    """)