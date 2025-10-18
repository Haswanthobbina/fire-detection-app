import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
# FIX: Import 'img_to_array' from the correct module
from tensorflow.keras.utils import img_to_array

# --- 1. Load Your Trained Model ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('FFD.keras')
    return model

model = load_my_model()

# --- 2. Define Class Names ---
# This is from your notebook's Cell 8 (class_mapping)
# class 0 is 'fire', class 1 is 'nofire'
class_names = ['fire', 'nofire']

# --- 3. Create the Prediction Function ---
# This function preprocesses the image and returns the *probability of fire*
def predict_fire(img):
    # Keras model expects 150x150
    size = (150, 150)
    
    # Resize and pad the image to be 150x150
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # Convert the PIL image to a NumPy array
    # FIX: Use the directly imported 'img_to_array' function
    img_array = img_to_array(img)
    
    # Add an extra dimension for the batch (model expects (1, 150, 150, 3))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Rescale the image just like you did in training (Cell 7)
    img_array /= 255.0

    # Make the prediction
    # The model outputs the probability of "nofire" (class 1)
    prediction = model.predict(img_array)
    prob_nofire = prediction[0][0]
    
    # We want the probability of "fire" (class 0)
    prob_fire = 1 - prob_nofire
        
    return float(prob_fire)

# --- 4. Build the Sidebar ---
st.sidebar.title("About This Project")
st.sidebar.write("""
This app uses a Convolutional Neural Network (CNN), built with Keras/TensorFlow, 
to detect forest fires in images.

It was trained on 'the-wildfire-dataset' from Kaggle.
""")

st.sidebar.subheader("Model Performance Metrics")
st.sidebar.write("**Final Test Accuracy: 80.47%**") # From your Cell 16

# This data is estimated from your notebook's plots (Cell 12 & 13)
epochs = list(range(1, 13))
train_acc = [0.62, 0.74, 0.76, 0.75, 0.77, 0.78, 0.79, 0.80, 0.81, 0.80, 0.82, 0.81]
val_acc = [0.73, 0.75, 0.71, 0.67, 0.77, 0.76, 0.77, 0.79, 0.72, 0.75, 0.82, 0.80]
train_loss = [0.86, 0.50, 0.48, 0.49, 0.47, 0.46, 0.45, 0.43, 0.42, 0.44, 0.40, 0.42]
val_loss = [0.50, 0.50, 0.51, 0.59, 0.47, 0.45, 0.44, 0.43, 0.49, 0.49, 0.43, 0.44]

# Create DataFrames for plotting
acc_df = pd.DataFrame({
    'Epoch': epochs,
    'Train Accuracy': train_acc,
    'Validation Accuracy': val_acc
}).set_index('Epoch')

loss_df = pd.DataFrame({
    'Epoch': epochs,
    'Train Loss': train_loss,
    'Validation Loss': val_loss
}).set_index('Epoch')

st.sidebar.subheader("Model Accuracy (Training vs. Validation)")
st.sidebar.line_chart(acc_df)

st.sidebar.subheader("Model Loss (Training vs. Validation)")
st.sidebar.line_chart(loss_df)


# --- 5. Build the Main Page Interface ---

st.title("Forest Fire Detection 🔥")
st.write("Upload an image and the model will predict if it contains fire.")

# Create file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded file as a PIL Image
    img = Image.open(uploaded_file)
    
    # Create two columns
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Display the uploaded image
        # FIX: Use 'use_container_width' instead of the deprecated 'use_column_width'
        st.image(img, caption='Uploaded Image.', use_container_width=True)
    
    with col2:
        # Add a spinner while the model is classifying
        with st.spinner('Classifying...'):
            prob_fire = predict_fire(img)
        
        st.subheader("Prediction Results")
        
        # --- Advanced Feature: Confidence Slider ---
        threshold = st.slider("Confidence Threshold for 'FIRE' (%)", 0.0, 100.0, 50.0, 1.0)
        
        confidence_fire_pct = prob_fire * 100
        
        # Display the result based on the slider
        if confidence_fire_pct > threshold:
            st.error(f"🔥 Prediction: **FIRE**")
            st.write(f"Confidence: **{confidence_fire_pct:.2f}%**")
        else:
            st.success(f"✅ Prediction: **NO FIRE**")
            st.write(f"Confidence (No Fire): **{(1-prob_fire)*100:.2f}%**")
        
        st.write("---")
        
        # Show the raw probability
        st.write(f"Raw Probability of Fire: {prob_fire:.4f}")
        # A progress bar is a nice visual for the probability
        st.progress(float(prob_fire))
        st.write("*(Adjust the confidence threshold using the slider above to see how it affects the prediction.)*")    
# --- End of app.py --- in last write made by haswanth 
