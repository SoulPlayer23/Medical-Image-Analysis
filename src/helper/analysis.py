from PIL import Image
import torch
from models.response_generator import generate_explanation
from preprocessing.data_preprocess import transform
from preprocessing.dataset_class import categories
from models.image_predictor import device
import json

def analyze_and_output_json(model, img_path):
    print("Analyzing image...")
    # Preprocess the image
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    # Predict image class
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][class_idx].item()
        class_label = categories[class_idx]

    # Generate explanation
    explanation = generate_explanation(class_label, confidence)

    # Create JSON output
    result = {
        "classification": class_label,
        "confidence": float(confidence),
        "explanation": explanation
    }
    return json.dumps(result, indent=4)