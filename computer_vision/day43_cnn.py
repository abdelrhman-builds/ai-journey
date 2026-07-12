# =============================================
# Day 43 - CNNs (Convolutional Neural Networks)
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Builds and trains a real CNN on the MNIST handwritten
# digit dataset, demonstrating convolution and pooling
# operations in a genuine, working image classifier.
# =============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# =============================================
# SECTION 1: Device Setup
# =============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================
# SECTION 2: Load MNIST Dataset
# =============================================
# MNIST: 70,000 images of handwritten digits (0-9),
# each 28x28 pixels, grayscale. The "hello world" of
# computer vision — small enough to train quickly,
# real enough to demonstrate genuine CNN capability.

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts image to a tensor,
                                 # scales pixel values to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))
        # normalizes using MNIST's known mean/std —
        # helps the network train faster and more stably
    ])

    train_dataset = datasets.MNIST(root='./data', train=True,
                                     download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"Training images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    return train_loader, test_loader, train_dataset


# =============================================
# SECTION 3: Visualize Sample Images
# =============================================

def visualize_samples(dataset):
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    for i in range(6):
        image, label = dataset[i]
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=100)
    print("Sample images saved as mnist_samples.png")
    plt.close()


# =============================================
# SECTION 4: Define the CNN Architecture
# =============================================

class SimpleCNN(nn.Module):
    """
    A CNN with 2 convolutional layers + pooling, followed
    by 2 fully-connected layers for final classification.

    Architecture flow:
    Input (1x28x28) -> Conv1 -> ReLU -> Pool -> Conv2 -> ReLU -> Pool
                     -> Flatten -> FC1 -> ReLU -> FC2 -> Output (10 classes)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolutional layer:
        # 1 input channel (grayscale), 16 output channels (16 filters),
        # 3x3 kernel size — each filter learns to detect a different
        # simple pattern (edges, curves, etc.)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        # Second convolutional layer:
        # 16 input channels (from conv1's output), 32 output channels —
        # combines simple patterns into more complex ones
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Max pooling: reduces each 2x2 region to its single max value,
        # halving the width and height each time it's applied
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 2 pooling operations: 28x28 -> 14x14 -> 7x7
        # 32 channels x 7 x 7 = 1568 values going into the fully-connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)

    def forward(self, x):
        # First conv block: convolution -> activation -> pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the 3D feature maps into a 1D vector for the FC layers
        x = x.view(x.size(0), -1)
        # Fully connected layers for final classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================================
# SECTION 5: Training Loop
# =============================================

def train_model(model, train_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Train Accuracy: {accuracy:.2f}%")


# =============================================
# SECTION 6: Evaluation
# =============================================

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    return accuracy


# =============================================
# SECTION 7: Run Everything
# =============================================

if __name__ == "__main__":
    train_loader, test_loader, train_dataset = load_data()
    visualize_samples(train_dataset)

    model = SimpleCNN().to(device)
    print(f"\nModel architecture:\n{model}")

    print("\nTraining...")
    train_model(model, train_loader, epochs=3)

    evaluate_model(model, test_loader)