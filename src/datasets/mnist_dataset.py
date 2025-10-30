import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_mnist_loaders(data_path, batch_size=64, val_split=0.1, num_workers=0):
    """
    Returns train, val, test DataLoaders for MNIST.
    Automatically downloads dataset if not present in data_path.
    """
    os.makedirs(data_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = [str(i) for i in range(10)]
    return train_loader, val_loader, test_loader, class_names
