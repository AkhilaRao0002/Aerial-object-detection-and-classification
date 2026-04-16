import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("bird_drone_resnet50.h5")
    return model

model = load_model()

# Class names (IMPORTANT: match your training order)
class_names = ['bird', 'drone']

# Title
st.title("🛩️ Aerial object Classifier")
st.write("Upload an image to classify whether it is a Bird 🦅 or a 🚁Drone")

st.sidebar.title("Dashboard")
st.sidebar.markdown("### About")
st.sidebar.info("""This application aims to classify aerial images into two categories — Bird or Drone — to locate and label these objects in real-world scenes.
""")

st.sidebar.markdown("### 🌐 Domain")
st.sidebar.write("Artificial Intelligence & Computer Vision")

st.sidebar.markdown("### 💼 Real time Business Use Cases")
st.sidebar.write("""
- 🛡️ Surveillance & Security : 
	Identify drones in restricted airspace for timely alerts. \n
- ✈️ Airspace Monitoring : 
	 Monitor runway zones for bird activity.\n
- 🐦 Wildlife Protection : 
	Detect birds near wind farms or airports to prevent accidents.\n   
- 🌿 Environmental Research :  
	Track bird populations using aerial footage without misclassification. 
""")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # same as training
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    # Convert to class + confidence
    if prediction > 0.5:
        label = class_names[1]  # drone
        confidence = prediction
    else:
        label = class_names[0]  # bird
        confidence = 1 - prediction

    # Displaying the result
    st.subheader("Prediction:")
    st.success(f"{label.upper()}")

    st.subheader("Confidence Score:")
    st.info(f"{confidence*100:.2f}%")