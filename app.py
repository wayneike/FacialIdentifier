
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Load the model
class ImprovedCNNModel(torch.nn.Module): # Replace with your actual model architecture
    def __init__(self, num_classes=40):
        super(ImprovedCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25) # Added dropout for regularization
        self.fc1 = nn.Linear(128 * 16 * 16, 512) # Adjusted input features
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x) # Apply dropout
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Apply dropout
        x = self.fc2(x)
        return x

model = ImprovedCNNModel()
model.load_state_dict(torch.load('CV_Proj2_weights.pth'))
model.eval()

# Define image transformations (Must be the same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.title("Facial Identifier")
st.header("Scenario: Facility Access Control by Facial Recognition.", divider=True)
st.subheader("Dataset: ttps://www.kaggle.com/datasets/kasikrit/att-database-of-faces/", divider=True)
st.subheader("Upload an image and get the prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["pgm"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted[0]].item()

    st.write(f"Prediction: {predicted.item() + 1}") # Labels are 1-indexed
    st.write(f"Confidence: {confidence:.2f}")
