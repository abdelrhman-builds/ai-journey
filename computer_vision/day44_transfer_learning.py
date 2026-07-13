# =============================================
# Day 44 - CNN Architectures + Transfer Learning
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Uses a pretrained ResNet-18 (trained on ImageNet) and
# fine-tunes it for a new task, demonstrating transfer
# learning — reusing general visual knowledge instead of
# training a CNN from scratch (Day 43's approach).
# =============================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_data():
    """
    CIFAR-10: 60,000 32x32 color images across 10 classes
    (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).
    Genuinely harder than MNIST — real photos, not clean digits.
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)

    train_subset = torch.utils.data.Subset(train_dataset, range(2000))
    test_subset = torch.utils.data.Subset(test_dataset, range(500))

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    print(f"Training subset: {len(train_subset)} images")
    print(f"Test subset: {len(test_subset)} images")
    print(f"Classes: {train_dataset.classes}")

    return train_loader, test_loader, train_dataset.classes


def build_transfer_model(num_classes=10):
    """
    Loads ResNet-18 with weights ALREADY TRAINED on ImageNet
    (1.2 million images, 1000 classes). Replaces only the
    final classification layer to output OUR 10 classes.
    """
    model = models.resnet18(weights='IMAGENET1K_V1')

    # FREEZE all existing layers
    for param in model.parameters():
        param.requires_grad = False

    # REPLACE the final layer — new layer is trainable
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model.to(device)


def train_transfer_model(model, train_loader, epochs=3):
    # Only optimize the NEW final layer's parameters
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    start_time = time.time()

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

    elapsed = time.time() - start_time
    print(f"\nTraining time: {round(elapsed, 1)} seconds")
    return elapsed


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
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")