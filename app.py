# app.py - Complete Streamlit App
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="♻️ Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 20px;
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">♻️ AI Waste Classifier</h1>', unsafe_allow_html=True)
st.markdown("### Upload an image to classify recyclable waste")

# Class information
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
RECYCLING_TIPS = {
    'cardboard': '📦 Flatten boxes and remove tape',
    'glass': '🥛 Rinse and remove caps/lids',
    'metal': '🥫 Rinse cans and remove labels',
    'paper': '📄 Keep dry and remove staples',
    'plastic': '🧴 Check recycling number (1-7)',
    'trash': '🚮 May not be recyclable - check local guidelines'
}

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('waste_classifier_6class.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload an image of waste material"
)

if uploaded_file is not None:
    # Display image
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        if model:
            with st.spinner("🔍 Analyzing..."):
                # Preprocess image
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                
                # Handle grayscale images
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                predictions = model.predict(img_array, verbose=0)[0]
                predicted_idx = np.argmax(predictions)
                predicted_class = CLASSES[predicted_idx]
                confidence = predictions[predicted_idx] * 100
            
            # Display results
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.success(f"### 🏷️ {predicted_class.upper()}")
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Recycling tip
            st.info(f"**♻️ Tip:** {RECYCLING_TIPS[predicted_class]}")
            
            # Check if recyclable
            if predicted_class != 'trash':
                st.success("✅ This item appears to be recyclable!")
            else:
                st.warning("⚠️ This may not be recyclable")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show all probabilities
            with st.expander("📊 View detailed probabilities"):
                for i, (cls, prob) in enumerate(zip(CLASSES, predictions)):
                    st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")
        else:
            st.error("Model not loaded. Please check if 'waste_classifier_6class.h5' is in the same folder.")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **AI Waste Classifier**
    
    Classifies waste into 6 categories:
    - ♻️ **Recyclable:**
      - Cardboard
      - Glass  
      - Metal
      - Paper
      - Plastic
    - 🗑️ **Non-recyclable:**
      - Trash
    """)
    
    st.header("📸 Tips")
    st.write("""
    For best results:
    1. Clear, well-lit images
    2. Single item per image
    3. Plain background
    4. Avoid blurry photos
    """)

# Footer
st.markdown("---")
st.caption("Built with TensorFlow & Streamlit | ♻️ Waste Classification Model")
