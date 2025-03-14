import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from preprocessing.data_loader import categories

# Load pre-trained ResNet50 model
model = models.resnet50(weights='IMAGENET1K_V1')

# Replace the final fully connected layer
num_classes = len(categories)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)