"""
# My Fashion MNIST Classifier Web Interface

This is a comprehensive web application built with Streamlit to showcase my PyTorch Fashion MNIST classifier.
The app provides an interactive interface to explore the dataset, visualize model predictions, and understand
the deep learning process in an engaging way.

## Features:
- Interactive dataset exploration
- Model architecture visualization
- Real-time predictions on uploaded images
- 3D feature visualization
- Performance metrics dashboard
- Educational content about deep learning

Author: Mohamed El-sadek Mohamed
Framework: Streamlit + PyTorch
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from PIL import Image
import io
import base64

# --- Page Configuration and Advanced Styling ---

st.set_page_config(
    page_title="Fashion AI Explorer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# You can replace "background.png" with your own image file if you have one.
# For now, we will use pure CSS for the animated background.

# --- Animated Background and Glassmorphism CSS ---
st.markdown("""
<style>
/* Animated Gradient Background */
body {
    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    height: 100vh;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Main App container styling */
.stApp {
    background-color: rgba(0,0,0,0); /* Make default background transparent */
}

/* Glassmorphism effect for cards */
.metric-card, .info-box, .st-emotion-cache-1r4qj8v, .st-emotion-cache-4oy321 {
    background: rgba(255, 255, 255, 0.15); /* Semi-transparent white */
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px); /* For Safari */
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    color: white; /* Change text color to white for better contrast */
}

/* Styling for headers and text */
.main-header, .sub-header, h3, p, .st-emotion-cache-10trblm, .st-emotion-cache-16idsys p {
    color: white !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}

.main-header {
    font-size: 3.5rem;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.sub-header {
    font-size: 2rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid rgba(255, 255, 255, 0.3);
    padding-bottom: 10px;
}

/* Sidebar styling */
.st-emotion-cache-16txtl3 {
    background: rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(5px);
    -webkit-backdrop-filter: blur(5px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* Ensure Plotly charts have a transparent background */
.js-plotly-plot .plotly, .js-plotly-plot .plotly-graph-div {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)


# --- Model Definition and Data Loading (No changes here) ---

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.4)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu1(self.conv1(x))))
        x = self.dropout2(self.pool2(self.relu2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout4(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = FashionCNN().to(device)
    try:
        model.load_state_dict(torch.load("fashion_mnist_model.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        st.error("Model file 'fashion_mnist_model.pth' not found. Please train the model first.")
        st.stop()
    return model

@st.cache_data
def load_fashion_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)
    images, labels = next(iter(test_loader))
    return images.numpy(), labels.numpy()

# --- Main App Logic ---

def main():
    st.markdown('<h1 class="main-header">✨ Fashion AI Explorer ✨</h1>', unsafe_allow_html=True)
    
    model = load_model()
    
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["🏠 Home", "📊 Dataset Explorer", "🧠 Model Architecture", "🔮 Predictions", "📈 3D Visualizations"]
    )
    
    images, labels = load_fashion_mnist_data()
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Dataset Explorer":
        show_dataset_explorer(images, labels)
    elif page == "🧠 Model Architecture":
        show_model_architecture()
    elif page == "🔮 Predictions":
        show_predictions_page(model, images, labels)
    elif page == "📈 3D Visualizations":
        show_3d_visualizations_page(model, images, labels)

def show_home_page():
    st.markdown('<h2 class="sub-header">🚀 Welcome to the Future of Fashion Tech</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <p>This interactive application demonstrates a powerful PyTorch-based Convolutional Neural Network (CNN) trained on the Fashion MNIST dataset. Explore the dataset, understand the model's architecture, and even test it with your own images. This project showcases a complete deep learning workflow, from data to deployment.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🌟 Key Features
        - **Native PyTorch Build**: Crafted with modern PyTorch best practices.
        - **Interactive 3D Visualizations**: Dive deep into data clusters with PCA.
        - **Real-time Predictions**: Test the model's knowledge instantly.
        - **Sleek Web Interface**: A dynamic and engaging user experience.
        """)
    with col2:
        st.markdown("""
        ### 🛠️ Technology Stack
        - **PyTorch**: For building and training the neural network.
        - **Streamlit**: For creating this interactive web application.
        - **Plotly**: For advanced and interactive data visualizations.
        - **Scikit-learn**: For dimensionality reduction (PCA).
        """)

def show_dataset_explorer(images, labels):
    st.markdown('<h2 class="sub-header">📊 Dataset Explorer</h2>', unsafe_allow_html=True)
    
    # Class distribution
    st.markdown("### Class Distribution")
    class_df = pd.DataFrame({'Class': [CLASS_NAMES[i] for i in labels]}).value_counts().reset_index(name='Count')
    fig_bar = px.bar(class_df, x='Class', y='Count', title="Distribution of Items in Test Set", color='Class')
    fig_bar.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_title="",
        yaxis_title="Number of Images"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Sample images
    st.markdown("### Sample Images from Each Class")
    cols = st.columns(5)
    for i in range(10):
        class_indices = np.where(labels == i)[0]
        if len(class_indices) > 0:
            sample_idx = class_indices[0]
            img = images[sample_idx].reshape(28, 28) * 0.5 + 0.5
            with cols[i % 5]:
                st.image(img, caption=CLASS_NAMES[i], use_column_width=True)

def show_model_architecture():
    st.markdown('<h2 class="sub-header">🧠 Model Architecture</h2>', unsafe_allow_html=True)
    model = FashionCNN()
    st.markdown("### Detailed Model Summary")
    st.code(str(model), language='python')

def show_predictions_page(model, images, labels):
    st.markdown('<h2 class="sub-header">🔮 Live Predictions</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Upload Your Image")
        uploaded_file = st.file_uploader("Choose a 28x28 grayscale image...", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('L').resize((28, 28))
            st.image(image, caption="Uploaded Image", width=150)
            
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            image_tensor = transform(image).unsqueeze(0)
            
            pred, conf = predict_image(model, image_tensor)
            st.success(f"**Prediction:** {CLASS_NAMES[pred]} (Confidence: {conf:.2f})")

    with col2:
        st.markdown("### Sample Predictions from Test Set")
        if st.button("Show Random Samples"):
            sample_cols = st.columns(3)
            for i in range(3):
                idx = np.random.randint(0, len(images))
                img_display = images[idx].reshape(28, 28) * 0.5 + 0.5
                true_label = labels[idx]
                
                img_tensor = torch.tensor(images[idx]).unsqueeze(0).to(device)
                pred_label, conf = predict_image(model, img_tensor)
                
                with sample_cols[i]:
                    st.image(img_display, use_column_width=True)
                    st.write(f"**True:** {CLASS_NAMES[true_label]}")
                    st.write(f"**Pred:** {CLASS_NAMES[pred_label]} ({conf:.2f})")

def show_3d_visualizations_page(model, images, labels):
    st.markdown('<h2 class="sub-header">📈 3D Feature Visualizations</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <p>This visualization uses Principal Component Analysis (PCA) to compress the model's high-dimensional understanding of images into just three dimensions. This allows us to 'see' how the model groups different types of clothing together.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Crunching the numbers for 3D visualization..."):
        df_3d, _ = create_3d_feature_visualization(model, images, labels)
    
    fig_3d = px.scatter_3d(
        df_3d, x="PC1", y="PC2", z="PC3", color="label_name",
        title="3D PCA of Fashion MNIST Features (from CNN)",
        labels={"PC1": "Component 1", "PC2": "Component 2", "PC3": "Component 3"},
        height=700
    )
    fig_3d.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend_title_text=''
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

# --- Helper functions (cached for performance) ---

@st.cache_data
def create_3d_feature_visualization(_model, images, labels):
    subset_size = min(2000, len(images))
    indices = np.random.choice(len(images), subset_size, replace=False)
    
    feature_representations = []
    labels_for_3d = []
    
    _model.eval()
    with torch.no_grad():
        for i in indices:
            img_tensor = torch.tensor(images[i]).unsqueeze(0).to(device)
            x = _model.dropout1(_model.pool1(_model.relu1(_model.conv1(img_tensor))))
            x = _model.dropout2(_model.pool2(_model.relu2(_model.conv2(x))))
            x = _model.dropout3(_model.relu3(_model.conv3(x)))
            x = _model.flatten(x)
            features = _model.relu4(_model.fc1(x))
            feature_representations.append(features.cpu().numpy().flatten())
            labels_for_3d.append(labels[i])

    feature_representations = np.array(feature_representations)
    labels_for_3d = np.array(labels_for_3d)
    
    pca = PCA(n_components=3)
    components = pca.fit_transform(feature_representations)
    
    df_3d = pd.DataFrame(components, columns=["PC1", "PC2", "PC3"])
    df_3d["label"] = labels_for_3d
    df_3d["label_name"] = df_3d["label"].apply(lambda x: CLASS_NAMES[x])
    
    return df_3d, pca

def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), confidence.item()

if __name__ == "__main__":
    main()
