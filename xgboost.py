import os
import cv2
import numpy as np
import torch
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset

import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Data Preparation

data_dir = r'C:\Users\switc\Desktop\SEM 7 PROJECT\Crop Disease Detection\Crop Disease Detection\Crop Leaves Disease Detection Training\data\dataset'

# Define image preprocessing and data augmentation transforms for PyTorch
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load data using PyTorch's DataLoader
class PlantDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.image_paths = []
        self.labels = []

        for i, label in enumerate(self.classes):
            label_dir = os.path.join(data_dir, label)
            image_files = os.listdir(label_dir)
            self.image_paths.extend([os.path.join(label_dir, file) for file in image_files])
            self.labels.extend([i] * len(image_files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Create data loaders
image_datasets = {x: PlantDataset(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Step 2: Feature Extraction using a Pretrained CNN

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x

# Initialize the feature extractor
feature_extractor = FeatureExtractor()

# Extract features from the dataset
def extract_features(data_loader, model):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, labels_batch in data_loader:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(labels_batch.cpu().numpy())
    return np.vstack(features), np.hstack(labels)

train_features, train_labels = extract_features(dataloaders['train'], feature_extractor)
test_features, test_labels = extract_features(dataloaders['val'], feature_extractor)

# Step 3: Train XGBoost Model

# Train an XGBoost model using the extracted features
xgb_model = xgb.XGBClassifier()
xgb_model.fit(train_features, train_labels)

# Step 4: Model Evaluation

y_pred = xgb_model.predict(test_features)
accuracy = accuracy_score(test_labels, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 5: Save the XGBoost Model
joblib.dump(xgb_model, 'best_model.h5')
