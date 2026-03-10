import streamlit as st
import torch
from torchvision import models, transforms, datasets
from PIL import Image

st.title("🌿 Medicinal Plant Detection AI")

st.write("""
This AI model can identify **3 medicinal plants only**:

• Neem  
• Betel  
• Guava  

Important Note: Upload a clear image of one of these leaves.  
Blurry images may lead to wrong prediction.
""")

# Load class names from dataset (same order used during training)
dataset = datasets.ImageFolder("dataset")
classes = dataset.classes

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("plant_model.pth", map_location="cpu"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Predicted Plant: **{classes[predicted.item()]}**")
