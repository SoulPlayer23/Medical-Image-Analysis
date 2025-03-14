import torch
import requests
from transformers import AutoTokenizer, BioGptForCausalLM
from models.model_initialization import device

tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model_llm = BioGptForCausalLM.from_pretrained("microsoft/biogpt").to(device)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_llm.to(device)


def generate_response(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False  # Set to True if you want streaming output
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        raise Exception(f"Error generating response: {response.text}")
    
def generate_explanation(class_label, confidence):
    label = class_label.replace("_", " ")
    if label == 'no tumor':
        return "No tumor detected."
    else:
        prompt = (
            f"Based on the MRI scan analysis, the classification result is '{label}' "
            f"with a confidence score of {confidence:.2f}. Provide a single line explanation of what "
            f"{label} condition typically indicates. Start with the text: The scan indicates that the patient has "
            f"a {label} tumor, which is a type of brain tumor that..."
        )
        return generate_response(prompt)