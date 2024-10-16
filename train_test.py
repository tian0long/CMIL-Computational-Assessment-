# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:22:45 2024

@author: lt766
"""



# Glomeruli Image Classification with Pre-trained ResNet18

# Import necessary libraries for file operations, data manipulation, image processing, visualization, 
# machine learning utilities, deep learning model operations, and to handle warnings.

import cv2
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torchvision import models

# Suppress future warnings from libraries to ensure clean output.
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define data paths and experiment parameters
base_dir = r'C:\Users\lt766\Downloads\public'
results_dir = 'results'  # Directory for output visualizations.
csv_file = os.path.join(base_dir, 'public.csv')  # CSV file with image annotations
root_dir = base_dir  # Root directory for image data
learning_rate = 0.001  # Optimizer's learning rate
batch_size = 32  # Batch size for model training
num_epochs = 10 # Number of training epochs

# Create directory for results if it does not exist
os.makedirs(results_dir, exist_ok=True)

# Try to load the CSV file with annotations and exit if the file cannot be read
try:
    annotations = pd.read_csv(csv_file)
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit()

# Read and describe the annotations to get insights into the dataset
annotations = pd.read_csv(csv_file)
print(annotations.describe())

# Visualize the distribution of image categories to identify potential class imbalances
print("\nCategory distribution:")
print(annotations['ground truth'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='ground truth', data=annotations)
plt.title('Category distribution')
plt.xlabel('Category')
plt.ylabel('Count')
category_distribution_fig = os.path.join(results_dir, 'category_distribution.png')
plt.savefig(category_distribution_fig)
plt.close()

# Collect and visualize image dimensions for preprocessing consistency
img_dims = {'height': [], 'width': []}
for idx in range(len(annotations)):
    img_label = annotations.iloc[idx, 1]
    subdir = "non_globally_sclerotic_glomeruli" if img_label == 0 else "globally_sclerotic_glomeruli"
    img_path = os.path.join(root_dir, subdir, annotations.iloc[idx, 0])
    with Image.open(img_path) as img:
        img_dims['height'].append(img.height)
        img_dims['width'].append(img.width)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(img_dims['height'], bins=20, kde=True)
plt.title('Image height distribution')
plt.xlabel('Height')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.histplot(img_dims['width'], bins=20, kde=True)
plt.title('Image width distribution')
plt.xlabel('Width')
plt.ylabel('Count')
plt.tight_layout()
size_distribution_fig = os.path.join(results_dir, 'size_distribution.png')
plt.savefig(size_distribution_fig)
plt.close()

# Dataset Preparation and Preprocessing
# Function to apply CLAHE to enhance image contrast for better model performance
def apply_clahe(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl1 = clahe.apply(gray)
    img_clahe_rgb = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_clahe_rgb)
    return img_pil

# Custom dataset class for handling glomeruli images with optional transformations and filtering by class
class GlomeruliDataset(Dataset):
    def __init__(self, annotations_df, root_dir, transform=None, class_filter=None):
        self.annotations = annotations_df
        self.root_dir = root_dir
        self.transform = transform
        self.class_filter = class_filter
        if self.class_filter is not None:
            self.annotations = self.annotations[self.annotations['ground truth'] == self.class_filter]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_label = self.annotations.iloc[idx, 1]
        subdir = "non_globally_sclerotic_glomeruli" if img_label == 0 else "globally_sclerotic_glomeruli"
        img_path = os.path.join(self.root_dir, subdir, self.annotations.iloc[idx, 0])
        image = apply_clahe(img_path)
        if self.transform:
            image = self.transform(image)
        return image, int(img_label)

# Define image transformations for data augmentation and normalization
majority_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

minority_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset_majority = GlomeruliDataset(annotations_df=annotations[annotations['ground truth']==0], root_dir=root_dir, transform=majority_transforms, class_filter=0)
dataset_minority = [GlomeruliDataset(annotations_df=annotations[annotations['ground truth']==1], root_dir=root_dir, transform=minority_transforms, class_filter=1) for _ in range(4)]
dataset = ConcatDataset([dataset_majority] + dataset_minority)

train_val_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
train_set, val_set = train_test_split(train_val_set, test_size=0.25, random_state=42)  
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print("Dataset preparation complete.")

# Model Preparation
# Configure pre-trained ResNet18 for binary classification with selective layer freezing
model = models.resnet18(pretrained=True)
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Training and Validation
# Implementation of the training and validation loop with loss and accuracy tracking
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        train_loss += loss.item() * images.size(0)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            val_loss += loss.item() * images.size(0)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
        scheduler.step()
        train_loss_avg = train_loss / train_total
        val_loss_avg = val_loss / val_total
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')


    # Plot training and validation accuracies after all epochs
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    accuracy_plot_path = os.path.join(results_dir, 'training_validation_accuracy.png')
    plt.savefig(accuracy_plot_path)
    plt.close()

    print(f"Plots saved to {results_dir}")

# Save the trained model later use and further evaluation.
model_save_path = os.path.join('model', 'model.pth')
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate model performance on test data to assess generalization.
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Calculate perforamnce metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

# Print calculated metrics to understand model performance
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')


