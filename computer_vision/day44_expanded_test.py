# =============================================
# Day 44 - Expanded Transfer Learning Test
# Author: Abdelrhman
# Date: July 2026
# =============================================
# Follow-up to day44_transfer_learning.py's initial small-scale
# demo (2000 images, 3 epochs, 69% accuracy). This expanded
# version uses 5x more data and more epochs to confirm that
# the initial result was correctly limited by its deliberately
# small scale, not a limitation of transfer learning itself.
# =============================================

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from day44_transfer_learning import (
    build_transfer_model, train_transfer_model, evaluate_model, device
)

# Redefine transform here since it was local to load_data()
# in the original file and not accessible from this script
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

train_dataset_full = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
test_dataset_full = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# 10,000 training images (5x more than the original 2,000)
train_subset_large = torch.utils.data.Subset(train_dataset_full, range(10000))
# Full 10,000-image test set (20x more than the original 500)
test_loader_large = DataLoader(test_dataset_full, batch_size=64, shuffle=False)
train_loader_large = DataLoader(train_subset_large, batch_size=64, shuffle=True)

print(f"Expanded training set: {len(train_subset_large)} images")
print(f"Full test set: {len(test_dataset_full)} images")

# Fresh model — not reusing the one trained on the small subset
model_expanded = build_transfer_model(num_classes=10)

# 10 epochs instead of 3
train_transfer_model(model_expanded, train_loader_large, epochs=10)

evaluate_model(model_expanded, test_loader_large)