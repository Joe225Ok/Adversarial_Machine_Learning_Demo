import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

def get_cifar10_loaders(data_path, batch_size=64, val_split=0.1, num_workers=0):
    """
    Returns train, val, test DataLoaders for CIFAR-10.
    Automatically downloads dataset if not present.
    """
    os.makedirs(data_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    # Split train into train + val
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.dataset.classes

    return train_loader, val_loader, test_loader, class_names
