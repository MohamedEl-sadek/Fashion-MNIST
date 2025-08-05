"""
# My PyTorch Fashion MNIST Classifier: A Deep Dive into Image Classification

Welcome to my personal journey into building a robust image classification model using PyTorch! This project is a comprehensive exploration of deep learning, from data preparation and model architecture design to training, evaluation, and insightful visualization. My goal here is not just to classify images, but to truly understand the underlying mechanisms and present them in a clear, well-documented manner.

## Project Overview

This project focuses on the Fashion MNIST dataset, a challenging benchmark for image classification. We'll be building a Convolutional Neural Network (CNN) from scratch using PyTorch, a powerful and flexible deep learning framework. The entire process is meticulously documented, reflecting my personal approach to solving real-world machine learning problems.

## Key Features

- **Native PyTorch Implementation**: A complete, modern PyTorch workflow for building and training deep learning models.
- **Comprehensive Data Handling**: Efficient loading, preprocessing, and augmentation of the Fashion MNIST dataset.
- **Custom CNN Architecture**: A thoughtfully designed CNN model tailored for image classification tasks.
- **Detailed Training Loop**: A clear and concise training and validation process, demonstrating best practices.
- **Performance Evaluation**: Thorough evaluation metrics to assess model performance.
- **Extensive Documentation**: Markdown explanations and inline comments to guide you through every line of code.
- **Interactive 3D Visualizations**: Beyond standard 2D plots, we'll explore innovative 3D visualizations to gain deeper insights into the model's behavior and data.
- **Model Saving/Loading**: Ability to save the trained model and load it for future use or inference.

## Dataset: Fashion MNIST

The Fashion MNIST dataset consists of 70,000 grayscale images of fashion articles, divided into 60,000 training examples and 10,000 testing examples. Each image is 28x28 pixels, representing one of 10 fashion categories. This dataset serves as an excellent stepping stone for understanding image classification with deep learning.

Let's dive into the code!
"""

# Importing the essential libraries for our deep learning adventure
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA

# Setting a random seed for reproducibility, because consistency is key in experiments!
torch.manual_seed(42)
np.random.seed(42)

# --- Data Loading and Preprocessing: The Foundation of Any Good Model ---

# Defining our image dimensions, as every pixel counts!
IMG_ROWS, IMG_COLS = 28, 28
NUM_CLASSES = 10

print("\n--- Preparing Our Fashion MNIST Data ---")
print("Loading dataset... This might take a moment, but good things come to those who wait!")

# Using torchvision.datasets.FashionMNIST for seamless data loading and preprocessing.
# This handles downloading, extracting, and loading the dataset directly into PyTorch tensors.
# I'm applying a transformation to prepare images for the model.
transform = transforms.Compose([
    transforms.ToTensor(), # Converts a PIL Image or numpy.ndarray to a PyTorch Tensor.
    transforms.Normalize((0.5,), (0.5,)) # Normalizes a tensor image with mean and standard deviation.
])

# Loading the training and test datasets.
# download=True ensures the dataset is downloaded if not already present.
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Splitting the training dataset into training and validation sets.
# I'm using an 80/20 split for training and validation.
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Creating DataLoader objects for efficient batch processing during training.
# This allows us to feed data to our model in small, manageable chunks.
batch_size = 256 # My chosen batch size, a good balance between speed and stability.

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training data samples: {len(train_dataset)}")
print(f"Validation data samples: {len(val_dataset)}")
print(f"Test data samples: {len(test_dataset)}")
print("Data preparation complete! We're ready to build our model.")

# --- Model Definition: Crafting Our Convolutional Neural Network ---

print("\n--- Designing Our PyTorch CNN Model ---")

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # My first convolutional layer: 32 filters, 3x3 kernel.
        # It learns to detect basic features like edges and corners.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU() # My chosen activation function for non-linearity.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Downsampling to reduce spatial dimensions.
        self.dropout1 = nn.Dropout(0.25) # My first dropout layer to prevent overfitting.

        # My second convolutional layer: 64 filters, 3x3 kernel.
        # It learns more complex patterns from the features detected by conv1.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        # My third convolutional layer: 128 filters, 3x3 kernel.
        # Even more complex feature extraction!
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.4) # A bit more dropout here, as this layer is deeper.

        # Flattening the output from convolutional layers to feed into dense layers.
        # This transforms our 2D feature maps into a 1D vector.
        self.flatten = nn.Flatten()

        # My first fully connected (dense) layer.
        # This layer takes the flattened features and learns high-level representations.
        self.fc1 = nn.Linear(128 * 7 * 7, 128) # 128 * 7 * 7 comes from the output size of conv3 after pooling
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.3) # Another dropout layer for regularization.

        # My final fully connected layer: the output layer.
        # It produces scores for each of our 10 fashion categories.
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        # The forward pass defines how data flows through our network.
        x = self.dropout1(self.pool1(self.relu1(self.conv1(x))))
        x = self.dropout2(self.pool2(self.relu2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.conv3(x))) # No pooling after third conv layer
        x = self.flatten(x)
        x = self.dropout4(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x

# Instantiating our model and moving it to the GPU if available.
# Because deep learning loves GPUs!
model = FashionCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Defining our loss function and optimizer.
# CrossEntropyLoss is perfect for multi-class classification.
# Adam optimizer is my go-to for its efficiency and effectiveness.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # My chosen learning rate.

print("Model architecture defined. Here's a summary of my masterpiece:")
print(model)

# --- Training the Model: The Heart of Deep Learning ---

print("\n--- Training Our Fashion MNIST Classifier ---")

EPOCHS = 10 # My chosen number of training epochs. You can adjust this!

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(1, EPOCHS + 1):
    # Training phase
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # Clear gradients from previous step
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward() # Backpropagation: calculate gradients
        optimizer.step() # Update model parameters

        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_accuracy = 100 * correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    # Validation phase
    model.eval() # Set model to evaluation mode
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad(): # No need to calculate gradients during validation
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            running_val_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += target.size(0)
            correct_val += (predicted == target).sum().item()

    epoch_val_loss = running_val_loss / len(val_dataset)
    epoch_val_accuracy = 100 * correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)

    print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%')

print("Training complete! Our model has learned a lot.")

# Save the trained model for later use
torch.save(model.state_dict(), 'fashion_mnist_model.pth')
print("Model saved as 'fashion_mnist_model.pth'.")

# --- Model Evaluation: How Did We Do? ---

print("\n--- Evaluating Our Model's Performance ---")

model.eval() # Set model to evaluation mode
correct_test = 0
total_test = 0
all_predictions = []
all_true_labels = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total_test += target.size(0)
        correct_test += (predicted == target).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(target.cpu().numpy())

test_accuracy = 100 * correct_test / total_test
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Generating a classification report for a detailed breakdown of performance per class.
# This helps us identify strengths and weaknesses of our model.
target_names = [f"Class {i}" for i in range(NUM_CLASSES)]
print("\nClassification Report:")
print(classification_report(all_true_labels, all_predictions, target_names=target_names))

# --- Visualization: Bringing Our Results to Life ---

print("\n--- Visualizing Training Progress and Predictions ---")

# Plotting training and validation accuracy over epochs.
# This helps us see if our model is learning effectively and if overfitting is occurring.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_accuracies, 'o-', label='Training Accuracy')
plt.plot(range(1, EPOCHS + 1), val_accuracies, 'o-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Plotting training and validation loss over epochs.
# This shows how well our model is minimizing errors during training.
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), train_losses, 'o-', label='Training Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, 'o-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_progress.png') # Saving the plot for later review.
print("Training progress plot saved as 'training_progress.png'.")

# Visualizing a subset of correctly predicted images.
# It's always satisfying to see our model get things right!
plt.figure(figsize=(10, 10))
plt.suptitle('Correctly Predicted Fashion Items', fontsize=16)
correct_indices = np.where(np.array(all_predictions) == np.array(all_true_labels))[0]

# Get actual images from the test_loader for visualization
all_test_images = []
for data, _ in test_loader:
    all_test_images.append(data.cpu().numpy())
all_test_images = np.vstack(all_test_images)

for i, correct in enumerate(correct_indices[:9]): # Displaying first 9 correct predictions
    plt.subplot(3, 3, i + 1)
    # Denormalize the image for proper display
    img = all_test_images[correct].reshape(IMG_ROWS, IMG_COLS) * 0.5 + 0.5
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.title(f"Pred: {target_names[all_predictions[correct]]}\nTrue: {target_names[all_true_labels[correct]]}", fontsize=8)
    plt.axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig('correct_predictions.png')
print("Correct predictions plot saved as 'correct_predictions.png'.")

# Visualizing a subset of incorrectly predicted images.
# This is crucial for understanding where our model struggles and for future improvements.
plt.figure(figsize=(10, 10))
plt.suptitle('Incorrectly Predicted Fashion Items', fontsize=16)
incorrect_indices = np.where(np.array(all_predictions) != np.array(all_true_labels))[0]
for i, incorrect in enumerate(incorrect_indices[:9]): # Displaying first 9 incorrect predictions
    plt.subplot(3, 3, i + 1)
    # Denormalize the image for proper display
    img = all_test_images[incorrect].reshape(IMG_ROWS, IMG_COLS) * 0.5 + 0.5
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.title(f"Pred: {target_names[all_predictions[incorrect]]}\nTrue: {target_names[all_true_labels[incorrect]]}", fontsize=8)
    plt.axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig('incorrect_predictions.png')
print("Incorrect predictions plot saved as 'incorrect_predictions.png'.")

# --- Advanced Visualization: Exploring Our Data in 3D! ---

print("\n--- Preparing for 3D Visualization ---")

# Let's get some feature representations from our trained model.
# We'll use the output of the first fully connected layer before the final classification.

model.eval() # Set model to evaluation mode
feature_representations = []
labels_for_3d = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # Forward pass up to the fc1 layer
        x = model.dropout1(model.pool1(model.relu1(model.conv1(data))))
        x = model.dropout2(model.pool2(model.relu2(model.conv2(x))))
        x = model.dropout3(model.relu3(model.conv3(x)))
        x = model.flatten(x)
        features = model.relu4(model.fc1(x)) # Get features from the first dense layer
        feature_representations.append(features.cpu().numpy())
        labels_for_3d.extend(target.cpu().numpy())

feature_representations = np.vstack(feature_representations)
labels_for_3d = np.array(labels_for_3d)

print(f"Extracted {feature_representations.shape[0]} feature vectors of size {feature_representations.shape[1]}.")

# Reducing dimensionality for 3D visualization using PCA.
# PCA helps us project high-dimensional data into a lower-dimensional space
# while retaining as much variance as possible. This makes it plottable in 3D.
print("Applying PCA to reduce dimensionality for 3D visualization...")
pca = PCA(n_components=3) # I want to visualize in 3 dimensions.
components = pca.fit_transform(feature_representations)

# Creating a DataFrame for Plotly.
# This makes it easy to plot with labels and colors.
print("Creating 3D scatter plot...")
df_3d = pd.DataFrame(components, columns=["PC1", "PC2", "PC3"])
df_3d["label"] = labels_for_3d
df_3d["label_name"] = df_3d["label"].apply(lambda x: target_names[x])

fig = px.scatter_3d(df_3d, x="PC1", y="PC2", z="PC3", color="label_name",
                    title="3D PCA of Fashion MNIST Features",
                    labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2", "PC3": "Principal Component 3"},
                    height=700)
fig.update_traces(marker=dict(size=3))
fig.write_html("fashion_mnist_3d_pca.html")
print("3D PCA visualization saved as 'fashion_mnist_3d_pca.html'.")

print("\n--- My PyTorch Fashion MNIST Classifier: Mission Accomplished! ---")
print("We've successfully built, trained, and evaluated a CNN model using PyTorch.")
print("The visualizations provide valuable insights into its performance.")
print("You can now use the 'fashion_mnist_model.pth' file with the Streamlit app or for further inference.")
