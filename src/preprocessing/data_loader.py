import os
from torch.utils.data import DataLoader
from preprocessing.dataset_class import data_dir, categories, BrainTumorDataset
from preprocessing.data_preprocess import transform

# Paths to training and testing folders
train_dir = os.path.join(data_dir, 'Training')
test_dir = os.path.join(data_dir, 'Testing')

# Create datasets
train_dataset = BrainTumorDataset(train_dir, categories, transform=transform)
test_dataset = BrainTumorDataset(test_dir, categories, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)