# app.py - COMPLETE WORKING CODE
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="♻️ AI Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    .prediction-box {
        background: #f8fff8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">♻️ AI Waste Classifier</h1>', unsafe_allow_html=True)
st.markdown("### Upload an image to classify recyclable waste")

# Class information
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

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
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

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
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.success(f"### 🏷️ {predicted_class.upper()}")
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Recycling info
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

# Footer
st.markdown("---")
st.caption("Built with TensorFlow & Streamlit | Waste Classification Model") Streamlit | ♻️ Waste Classification Model")
