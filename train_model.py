import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Device selection (GPU for Mac M2)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Dataset location
data_dir = "dataset"

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)
print("Classes:", dataset.classes)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load pretrained ResNet18
model = models.resnet18(weights="DEFAULT")

# Freeze earlier layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for 3 plants
model.fc = nn.Linear(model.fc.in_features, 3)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

epochs = 30

for epoch in range(epochs):

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "plant_model.pth")

print("Model training finished and saved.")
