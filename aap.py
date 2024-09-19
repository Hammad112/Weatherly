import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import base64
from io import BytesIO

# Load the trained model
model = load_model('weather_classification_model.h5')

# Define the class labels (adjust according to your classes)
class_labels = ['Cloudy', 'Rainy', 'Shine', 'Sunrise', 'Foggy']

# Function to preprocess the image for model prediction
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the input size of VGG16/VGG19
    image = np.array(image)
    if image.shape[-1] == 4:  # If PNG with transparency, convert to RGB
        image = image[:, :, :3]
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to display image in center using base64 encoding
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_data = buffered.getvalue()
    encoded = base64.b64encode(img_data).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

# Streamlit app
st.title("Weather Classification")

# Upload image
uploaded_file = st.file_uploader("Upload an image of the weather", type=["jpg", "jpeg", "png", "bmp", "tiff", "jfif"])

if uploaded_file is not None:
    try:
        # Open the image
        image = Image.open(uploaded_file)

        # Resize the image to 214x150
        resized_image = image.resize((214, 150))

        # Get base64-encoded string of the resized image
        image_base64 = image_to_base64(resized_image)

        # Display the resized image, center it, and add a thin black border
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="{image_base64}" alt="Resized Image" 
                style="border: 1px solid black; border-radius: 5px;" width="300" height="200" />
            </div>
            """, unsafe_allow_html=True
        )

        # Preprocess the image for model prediction
        processed_image = preprocess_image(image)

        # Make predictions
        predictions = model.predict(processed_image)

        # Get the class label
        predicted_class = class_labels[np.argmax(predictions)]

        # Display the prediction with transparent background and bold text
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px; margin-bottom:20px;">
                <div style="
                    background-color: rgba(0, 0, 0, 0.2);  /* Transparent background */
                    color: white;
                    font-weight: bold;
                    padding: 15px;
                    display: inline-block;
                    font-size: 20px;
                    border-radius: 10px;
                ">
                    Prediction: {predicted_class}
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a proper image file.")
else:
    st.write("Please upload an image to classify.")
