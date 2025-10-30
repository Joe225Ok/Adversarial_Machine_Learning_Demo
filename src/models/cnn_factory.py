# src/models/cnn_factory.py
import os
from datasets.caltech101_dataset import get_caltech101_loaders, download_and_extract_caltech101
from models.caltech101_cnn import Caltech101CNN
from models.mnist_cnn import MNISTCNN
from models.cifar10_cnn import CIFAR10CNN

def build_model(dataset_name):
    """
    Build a CNN model for the specified dataset.
    Supports: MNIST, CIFAR10, CALTECH101
    """
    dataset_name = dataset_name.upper()

    if dataset_name == "CALTECH101":
        # ---------------- Data path ----------------
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "../../data/CALTECH101")
        data_path = os.path.abspath(data_path)

        # Ensure dataset is downloaded and extracted
        extracted_path = download_and_extract_caltech101(data_path)
        print(f"[cnn_factory] Caltech101 dataset is ready at: {extracted_path}")

        # ---------------- Get number of classes dynamically ----------------
        _, _, _, class_names = get_caltech101_loaders(
            data_path=data_path,
            batch_size=1,
            val_split=0.1,
            num_workers=0
        )
        num_classes = len(class_names)

        # ---------------- Return CNN model ----------------
        return Caltech101CNN(num_classes=num_classes)

    elif dataset_name == "MNIST":
        return MNISTCNN(num_classes=10)

    elif dataset_name == "CIFAR10":
        return CIFAR10CNN(num_classes=10)

    else:
        raise ValueError(f"[cnn_factory] Unknown dataset: {dataset_name}")
