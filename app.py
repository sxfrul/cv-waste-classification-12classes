import streamlit as st
import numpy as np
from PIL import Image
import time
import os

# --- 1. Library Imports (Cached for speed) ---
@st.cache_resource
def load_libraries():
    # We import these inside a function so the app loads the UI fast
    # before loading the heavy ML libraries
    import tensorflow as tf
    import torch
    from torchvision import models, transforms
    import torch.nn as nn
    return tf, torch, models, transforms, nn

tf, torch, models, transforms, nn = load_libraries()

# --- 2. Configuration & Class Names ---
# These must match the ALPHABETICAL order of folders in your dataset
CLASS_NAMES = [
    'batteries', 'biological', 'brown-glass', 'cardboard', 'clothes', 
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

# Set page config
st.set_page_config(page_title="Waste Classification Demo", page_icon="♻️")

st.title("♻️ Waste Classification AI Comparison")
st.write("Upload an image to see how a **Simple CNN (TensorFlow)** compares against **MobileNetV3 (PyTorch)**.")

# --- 3. Load Models ---
@st.cache_resource
def load_tf_model():
    """Loads the Keras Simple CNN model (.keras format)"""
    try:
        # UPDATED: Now loading the .keras format
        model = tf.keras.models.load_model('simple_cnn_waste.keras')
        return model
    except Exception as e:
        st.error(f"Error loading TensorFlow model: {e}")
        return None

@st.cache_resource
def load_pt_model():
    """Loads the PyTorch MobileNetV3 model"""
    try:
        # Recreate architecture
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, len(CLASS_NAMES))
        
        # Load weights
        # map_location=torch.device('cpu') ensures it runs on servers without GPUs
        model.load_state_dict(torch.load('best_waste_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

with st.spinner("Loading AI Models... (this may take a minute)"):
    tf_model = load_tf_model()
    pt_model = load_pt_model()

# --- 4. Prediction Logic ---

def predict_tf(image, model):
    """
    Preprocess and predict using TensorFlow Simple CNN
    Must match training: 64x64, 1./255 rescale
    """
    start_time = time.time()
    
    # Resize to 64x64
    img = image.resize((64, 64))
    img_array = np.array(img)
    
    # Normalize (1./255)
    img_array = img_array / 255.0
    
    # Expand dims to create batch (1, 64, 64, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array, verbose=0)
    end_time = time.time()
    
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    
    return CLASS_NAMES[class_idx], confidence, end_time - start_time

def predict_pt(image, model):
    """
    Preprocess and predict using PyTorch MobileNetV3
    Must match training: 224x224, ImageNet Normalization
    """
    start_time = time.time()
    
    # Define Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Transform and add batch dimension
    img_t = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, classes = torch.max(probs, 1)
        
    end_time = time.time()
    
    return CLASS_NAMES[classes.item()], conf.item(), end_time - start_time

# --- 5. UI Layout ---

uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify Waste'):
        if tf_model and pt_model:
            col1, col2 = st.columns(2)
            
            # --- TensorFlow Prediction ---
            with col1:
                st.subheader("Simple CNN (TF)")
                try:
                    label, conf, t = predict_tf(image, tf_model)
                    st.success(f"**{label.upper()}**")
                    st.write(f"Confidence: {conf:.2%}")
                    st.write(f"Time: {t:.4f}s")
                except Exception as e:
                    st.error(f"Error: {e}")

            # --- PyTorch Prediction ---
            with col2:
                st.subheader("MobileNetV3 (PyTorch)")
                try:
                    label, conf, t = predict_pt(image, pt_model)
                    st.success(f"**{label.upper()}**")
                    st.write(f"Confidence: {conf:.2%}")
                    st.write(f"Time: {t:.4f}s")
                except Exception as e:
                    st.error(f"Error: {e}")
            
            # Comparison Logic
            st.divider()
            st.info("💡 **Observation:** The MobileNetV3 model usually takes slightly longer to run because it is much deeper (more layers), but it is often more robust to complex backgrounds than the Simple CNN.")