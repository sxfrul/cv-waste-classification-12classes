import streamlit as st
import numpy as np
from PIL import Image
import time
import os

# --- 1. Library Imports (Cached) ---
@st.cache_resource
def load_libraries():
    import tensorflow as tf
    import torch
    from torchvision import models, transforms
    import torch.nn as nn
    return tf, torch, models, transforms, nn

tf, torch, models, transforms, nn = load_libraries()

# --- 2. Configuration ---
CLASS_NAMES = [
    'batteries', 'biological', 'brown-glass', 'cardboard', 'clothes', 
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

st.set_page_config(page_title="Waste Classification Demo", page_icon="♻️", layout="wide")

st.title("♻️ Waste Classification AI Comparison")
st.markdown("Upload an image to compare **Simple CNN (TensorFlow)** vs **MobileNetV3 (PyTorch)**.")

# --- 3. Load Models ---
@st.cache_resource
def load_tf_model():
    try:
        model = tf.keras.models.load_model('simple_cnn_waste.keras')
        return model
    except Exception as e:
        st.error(f"Error loading TensorFlow model: {e}")
        return None

@st.cache_resource
def load_pt_model():
    try:
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, len(CLASS_NAMES))
        model.load_state_dict(torch.load('best_waste_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

with st.spinner("Loading AI Models..."):
    tf_model = load_tf_model()
    pt_model = load_pt_model()

# --- 4. Prediction Logic ---
def predict_tf(image, model):
    start_time = time.time()
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array, verbose=0)
    end_time = time.time()
    
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    return CLASS_NAMES[class_idx], confidence, end_time - start_time

def predict_pt(image, model):
    start_time = time.time()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, classes = torch.max(probs, 1)
        
    end_time = time.time()
    return CLASS_NAMES[classes.item()], conf.item(), end_time - start_time

# --- 5. UI Layout (Centered Image & Matched Height) ---

uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Split into two main columns
    col1, col2 = st.columns([1, 1])
    
    # --- Left Column: Image (Centered) ---
    with col1:
        st.info("🖼️ **Input Image**")
        
        # UI TRICK: Create 3 sub-columns [spacer, image, spacer] to center content
        # The middle column is 6x wider than the side spacers
        c1, c2, c3 = st.columns([1, 6, 1])
        
        with c2:
            # Resize image to approx height of the two result boxes combined (approx 450px)
            fixed_height = 450
            aspect_ratio = image.width / image.height
            new_width = int(fixed_height * aspect_ratio)
            display_image = image.resize((new_width, fixed_height))
            
            # Display image (centered by the columns above)
            st.image(display_image)

    # --- Right Column: Results ---
    with col2:
        st.info("📊 **Predictions**")
        
        if st.button('Run Analysis', type="primary", use_container_width=True):
            if tf_model and pt_model:
                
                # Container 1: TF
                with st.container(border=True):
                    st.markdown("### 🧠 Simple CNN (TF)")
                    try:
                        label, conf, t = predict_tf(image, tf_model)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Prediction", label.upper())
                        m2.metric("Confidence", f"{conf:.1%}")
                        m3.metric("Time", f"{t:.4f}s")
                    except Exception as e:
                        st.error(f"Error: {e}")

                # Container 2: PyTorch
                with st.container(border=True):
                    st.markdown("### 🚀 MobileNetV3 (PyTorch)")
                    try:
                        label, conf, t = predict_pt(image, pt_model)
                        p1, p2, p3 = st.columns(3)
                        p1.metric("Prediction", label.upper())
                        p2.metric("Confidence", f"{conf:.1%}")
                        p3.metric("Time", f"{t:.4f}s")
                    except Exception as e:
                        st.error(f"Error: {e}")