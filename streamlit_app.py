import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('fashion_CNN_new.h5')

# Define class labels for Fashion MNIST dataset
class_labels = [
    'T-shirt/top', 'Trouser', 'Bag', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Pullover', 'Ankle boot'
]

# Function to preprocess uploaded image
def preprocess_image(image):
    image = image.resize((28, 28))
    image = image.convert('L')
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 28, 28, 1)

# Main function for Streamlit app
def main():
    st.title("Image Classification Using CNN on Fashion Products")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image with reduced size
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=False, width=200)
        
        # Preprocess and predict
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image)
        
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction[0])
        
        st.write(f"Predicted Class: {class_labels[predicted_class_index]}")
        
if __name__ == "__main__":
    main()