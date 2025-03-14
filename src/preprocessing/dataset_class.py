import os
from PIL import Image
from torch.utils.data import Dataset

# Define absolute path to dataset - adjust this to match your actual dataset location
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(project_root, "brain_tumor_dataset")

# Verify dataset directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory not found at: {data_dir}")
    
categories = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Custom Dataset Class
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, categories, transform=None):
        self.root_dir = root_dir
        self.categories = categories
        self.transform = transform
        self.images = []
        self.labels = []

        # Populate image paths and labels
        for category in categories:
            path = os.path.join(root_dir, category)
            class_num = categories.index(category)
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                self.images.append(img_path)
                self.labels.append(class_num)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')  # Ensure images are in RGB format

        if self.transform:
            img = self.transform(img)

        return img, label