import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

classes = ["Betel","Guava","Neem"]

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("plant_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

st.title("Medicinal Plant Detector")

file = st.file_uploader("Upload a leaf image")

if file:
    img = Image.open(file)
    st.image(img)

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(output,1).item()

    st.success(f"Predicted Plant: {classes[prediction]}")