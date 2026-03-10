# Medicinal Plant Detection AI 🌿

This project is a **deep learning web application** that detects medicinal plants from leaf images.

The model is trained using **PyTorch ResNet18** and deployed using **Streamlit**.

## Supported Plants

The model currently detects three medicinal plants:

- Neem
- Betel
- Guava

## Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- PIL (Python Imaging Library)

## Project Structure

Medicinal-Plant-Detection-AI
│
├── app.py              # Streamlit web application
├── train_model.py      # Model training script
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── dataset/            # Training dataset (not included in repo)

## How to Run the Project

### 1 Install dependencies

pip install -r requirements.txt

### 2 Train the model

python train_model.py

### 3 Run the web app

streamlit run app.py

Then open the browser at:

http://localhost:8501

## Model

The model uses **ResNet18** architecture for image classification.

Input image size: 224 × 224

## Future Improvements

- Add more medicinal plants
- Improve dataset size
- Deploy the application online

## Author

Muhammad Taif  
BCA Student (AI, Cloud Computing, DevOps with IBM)
