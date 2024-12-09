

import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# Load the trained EfficientNet model
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)  # Binary classification
    model.load_state_dict(torch.load("efficientnet_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit UI
st.title("Skin Cancer Detection with EfficientNet")
st.write("Upload an image or take a picture using your camera to classify it as benign or malignant.")

# Option to upload a file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Option to take a picture using the camera
camera_image = st.camera_input("Take a picture")

# Check if an image is uploaded or taken from the camera
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)
    st.write("Classifying...")

if uploaded_file is not None or camera_image is not None:
    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()
    
    # Display result
    if prediction > 0.5:
        st.write("**Prediction:** Malignant (Positive)")
    else:
        st.write("**Prediction:** Benign (Negative)")

