# src/main.py
import os
import sys
import torch
from datasets.dataset_factory import get_dataset_loaders
from models.cnn_factory import build_model
from training.train import train_model
from visualization.visualize import visualize_adversarial_examples

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <epsilon> <dataset>")
        sys.exit(1)
    eps = float(sys.argv[1])
    dataset_name = sys.argv[2].upper()

    epochs = 6
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dataset={dataset_name} | eps={eps} | epochs={epochs} | batch_size={batch_size} | device={device}")

    # Load dataset
    train_loader, val_loader, test_loader, class_names = get_dataset_loaders(dataset_name, batch_size=batch_size)

    # Build model

    model = build_model(dataset_name).to(device)
 

    ckpt_path = os.path.join("src", "models", f"{dataset_name.lower()}_cnn.pth")

    # Load checkpoint if exists
    load_model = False
    if os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"[INFO] Loaded checkpoint: {ckpt_path}")
            load_model = True
        except RuntimeError as e:
            print(f"[WARNING] Could not load checkpoint due to mismatch: {e}")
            print("[INFO] Re-training model from scratch...")
            load_model = False

    if not load_model:
        print("[train] Starting training...")
        train_model(model, train_loader, val_loader, epochs=epochs, device=device)
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Model checkpoint saved at {ckpt_path}")

    visualize_adversarial_examples(model, test_loader, eps, class_names, device=device, dataset_name=dataset_name)

if __name__ == "__main__":
    main()
