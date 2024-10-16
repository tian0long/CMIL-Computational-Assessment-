# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:12:43 2024

@author: lt766
"""
import cv2
import os
import sys
import torch
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset


# Custom Dataset class that extends PyTorch's Dataset class to handle the loading of images from a directory.
# It includes functionality to preprocess and transform the images as needed for input into a neural network.

# CLAHE (Contrast Limited Adaptive Histogram Equalization) function to enhance image contrast, which can improve
# image quality for classification tasks, particularly in medical imaging.
def apply_clahe(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    img = cv2.imread(image_path)  # Read the image from the specified path.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)  # Apply CLAHE to enhance contrast.
    cl1 = clahe.apply(gray)  # Apply the CLAHE algorithm to the grayscale image.
    img_clahe_rgb = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)  # Convert the image back to RGB format.
    img_pil = Image.fromarray(img_clahe_rgb)  # Convert the image to a PIL image for further processing.
    return img_pil

class ImageDataset(Dataset):
    # Initializes the dataset, setting the root directory for images and applying any specified transformations.
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Directory containing the images.
        self.transform = transform  # Transformation pipeline to be applied to each image.
        self.images = os.listdir(root_dir)  # List all image files in the directory.

    # Returns the number of images in the dataset.
    def __len__(self):
        return len(self.images)

    # Retrieves an image and its associated transformations by index, then returns the transformed image and its filename.
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])  # Construct the full path to the image file.
        image = apply_clahe(img_name)  # Apply CLAHE preprocessing to the image.
        if self.transform:
            image = self.transform(image)  # Apply any additional transformations (e.g., resizing, normalization).
        return image, self.images[idx]  # Return the preprocessed image and its filename.


# Function to load a pre-trained ResNet18 model from a saved model file and adjust it for binary classification.
# The model is modified to output two classes, suitable for tasks like binary medical image classification.
def load_model(model_path):
    model = models.resnet18()  # Load the pre-trained ResNet18 model.
    num_ftrs = model.fc.in_features  # Get the number of input features for the final layer.
    model.fc = torch.nn.Linear(num_ftrs, 2)  # Modify the final fully connected layer for binary classification.
    model.load_state_dict(torch.load(model_path))  # Load the model weights from the specified file.
    model.eval()  # Set the model to evaluation mode (no gradients needed during inference).
    return model


# Function to make predictions using the loaded model and a DataLoader.
# This function iterates over batches of images, runs them through the model, and collects predictions.
def predict(model, dataloader, device):
    predictions = []  # List to store predictions.
    files = []  # List to store filenames of the processed images.
    with torch.no_grad():  # Disable gradient computation to save memory and speed up inference.
        for images, image_files in dataloader:  # Iterate over the DataLoader's batches of images.
            images = images.to(device)  # Move images to the specified device (CPU/GPU).
            outputs = model(images)  # Forward pass through the model to obtain predictions.
            _, predicted = torch.max(outputs.data, 1)  # Get the index of the maximum score (predicted class).
            predictions.extend(predicted.cpu().numpy())  # Store predictions and move them to CPU as numpy arrays.
            files.extend(image_files)  # Store the filenames of the processed images.
    return files, predictions  # Return filenames and corresponding predictions.


# Main function that handles the end-to-end process: loading the model, preparing the dataset, predicting labels,
# and saving the results to a CSV file.
def main(folder_path, model_path, output_csv):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU.

    # Define a transformation pipeline that resizes images and normalizes them to match the model's expected input format.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 pixels.
        transforms.ToTensor(),  # Convert the images to PyTorch tensors.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images based on pre-trained model's expectations.
    ])

    # Create an ImageDataset using the specified folder and transformation pipeline.
    dataset = ImageDataset(root_dir=folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # Create DataLoader for batch processing, no shuffling for inference.

    # Load the pre-trained model and move it to the selected device.
    model = load_model(model_path).to(device)

    # Make predictions on the dataset using the model and DataLoader.
    files, predictions = predict(model, dataloader, device)

    # Save the predictions along with the corresponding image filenames to a CSV file.
    df = pd.DataFrame({
        'Image': files,  # Column for image filenames.
        'Label': predictions  # Column for predicted labels.
    })
    df.to_csv(output_csv, index=False)  # Save the DataFrame as a CSV file.
    print(f"Prediction results saved to {output_csv}")  # Notify the user that the results have been saved.


# This section ensures the script can be executed from the command line.
if __name__ == "__main__":

    # Command line arguments for specifying the folder containing images, the trained model file, and the output CSV file.
    folder_path = r''  # Path to the folder containing images for prediction.
    model_path = r''  # Path to the pre-trained model file.
    output_csv = r''  # Path to save the prediction results as a CSV file.
    
    # Call the main function, initiating the prediction process.
    main(folder_path, model_path, output_csv)
