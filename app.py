import streamlit as st
import torch
from generator import Generator
import matplotlib.pyplot as plt

noise_dim = 100

st.title("GAN Based Image Synthesis System")

@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("model/generator.pth", map_location="cpu"))
    model.eval()
    return model

G = load_model()
num_images = st.slider("Number of Images",1,10,5)

if st.button("Generate Images"):

    cols = st.columns(num_images)

    for i in range(num_images):

        noise = torch.randn(1, noise_dim)

        with torch.no_grad():
            img = G(noise)

        img = img.squeeze().numpy()

        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        cols[i].pyplot(fig)