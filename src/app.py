import os
from models.image_predictor import train_model
from helper.analysis import analyze_and_output_json
from preprocessing.data_loader import train_loader, test_loader
from models.model_initialization import model, criterion, optimizer
import torch

model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")

print("Loading model...")

def load_model(model_path):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Check if the model file exists
if os.path.exists(model_path):
    print("Model file found. Loading model...")
    model = load_model(model_path)
else:
    print("Model file not found. Training model...")
    # Train the model
    model = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20)    

# Example usage
img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                        "brain_tumor_dataset", 
                        "Testing", 
                        "glioma_tumor", 
                        "image(31).jpg")
output = analyze_and_output_json(model, img_path)
print(output)