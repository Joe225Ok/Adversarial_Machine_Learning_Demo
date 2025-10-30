# src/datasets/caltech101_dataset.py
import os
import tarfile
import zipfile
import urllib.request
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

CALTECH101_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"

def download_and_extract_caltech101(data_path):
    base_dir = os.path.join(data_path, "Caltech101")
    os.makedirs(base_dir, exist_ok=True)
    zip_path = os.path.join(base_dir, "caltech-101.zip")

    # Download if not exists
    if not os.path.exists(zip_path):
        print("[Caltech101] Downloading dataset ...")
        urllib.request.urlretrieve(CALTECH101_URL, zip_path)
        print("[Caltech101] Download complete.")

    # Extract main ZIP
    extracted_dir = os.path.join(base_dir, "caltech-101")
    if not os.path.exists(extracted_dir):
        print("[Caltech101] Extracting main ZIP ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        print("[Caltech101] Main ZIP extraction complete.")

    # Check for object categories
    obj_cat_dir = os.path.join(extracted_dir, "101_ObjectCategories")
    if not os.path.exists(obj_cat_dir):
        raise RuntimeError(f"[Caltech101] Expected folder not found after extraction: {obj_cat_dir}")

    return obj_cat_dir

def get_caltech101_loaders(data_path, batch_size=64, val_split=0.1, num_workers=0):
    # Resize images to 224x224 for CNN
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset_dir = download_and_extract_caltech101(data_path)
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

    # Split into train/val/test
    total_len = len(full_dataset)
    val_len = int(val_split * total_len)
    test_len = int(val_split * total_len)
    train_len = total_len - val_len - test_len
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_len, val_len, test_len]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = full_dataset.classes
    return train_loader, val_loader, test_loader, class_names
