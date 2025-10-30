# src/datasets/dataset_factory.py
import os
from .mnist_dataset import get_mnist_loaders
from .cifar10_dataset import get_cifar10_loaders
from .caltech101_dataset import get_caltech101_loaders

# Root data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, "../../data")
DATA_DIR = os.path.abspath(DATA_DIR)
print(f"[dataset_factory] DATA_DIR = {DATA_DIR}")

def get_dataset_loaders(dataset_name, batch_size=64, val_split=0.1, num_workers=0):
    dataset_name = dataset_name.upper()
    if dataset_name == "MNIST":
        data_path = os.path.join(DATA_DIR, "MNIST")
        return get_mnist_loaders(data_path=data_path, batch_size=batch_size, val_split=val_split, num_workers=num_workers)
    elif dataset_name == "CIFAR10":
        data_path = os.path.join(DATA_DIR, "CIFAR10")
        return get_cifar10_loaders(data_path=data_path, batch_size=batch_size, val_split=val_split, num_workers=num_workers)
    elif dataset_name == "CALTECH101":
        data_path = os.path.join(DATA_DIR, "CALTECH101")
        return get_caltech101_loaders(data_path=data_path, batch_size=batch_size, val_split=val_split, num_workers=num_workers)
    else:
        raise ValueError(f"[dataset_factory] Unknown dataset: {dataset_name}")
