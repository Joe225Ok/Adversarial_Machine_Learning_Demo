# Adversarial Attacks on CNN Image Classifiers

This repository provides a complete implementation for generating, evaluating, and visualizing adversarial attacks on three popular image datasets: **CALTECH101**, **CIFAR10**, and **MNIST**. It includes training code, FGSM attack generation, logging, and visualization utilities with automatic image saving under `src/plots/`.

---

## 🚀 Quickstart

### 1. Dataset folders

The dataset folders are already present. GitHub enforces a **100 MB limit**, so the actual datasets are **not included**, but the folder structure is ready. Datasets will be automatically downloaded via PyTorch utilities when you run `main.py`.

The folder structure:

data/
 ├── CALTECH101/
 ├── CIFAR10/
 └── MNIST/

Trained models are saved under src/models/

### 2. Trained models are not included

The following trained models exceed GitHub's file-size limit and are **not uploaded**: `src/models/caltech101_cnn.pth`, `src/models/cifar10_cnn.pth`, `src/models/mnist_cnn.pth`. This does **not** prevent the code from running. If models are not found, they will be trained automatically and saved to `src/models/`.

---

## ▶️ Run an Attack

Use:

python src/main.py <epsilon> <dataset>

Examples:

python src/main.py 0.03 MNIST  
python src/main.py 0.02 CIFAR10  
python src/main.py 0.01 CALTECH101

- `<epsilon>` controls attack strength.  
- `<dataset>` must be one of: `MNIST`, `CIFAR10`, `CALTECH101`.

All results (success logs, plots, images) are automatically saved under `src/plots/`. Filenames follow the pattern `successful_attacks_FGSM_<dataset>_eps<epsilon>.png`. Files are **overwritten** if they already exist.

---

## 📊 Visualizations

The visualization module:

- Filters out any sample where **GT**, **Orig pred**, or **Adv pred** is `BACKGROUND_Google`, `Faces`, or `Faces_easy`.  
- Selects up to the top 5 strongest successful attacks per dataset.  
- Displays original vs. adversarial predictions with confidence percentages.  
- Produces high-quality plots.  
- Saves images with descriptive filenames in `src/plots/`.

Example output: `successful_attacks_FGSM_MNIST_eps0.03.png`, `successful_attacks_FGSM_CIFAR10_eps0.02.png`

---

### Pipeline Overview

           ┌───────────────────────┐
           │        Dataset        │
           └────────────┬──────────┘
                        │
           ┌────────────▼──────────┐
           │     Build Model       │
           │     (CNN Factory)     │
           └────────────┬──────────┘
                        │
           ┌────────────▼──────────┐
           │  Original Predictions │
           └────────────┬──────────┘
                        │
           ┌────────────▼──────────┐
           │  Adversarial FGSM     │
           └────────────┬──────────┘
                        │
           ┌─────────────────────────────┐
           │  Adversarial Predictions    │
           └────────────┬────────────────┘
                        │
           ┌────────────▼──────────┐
           │    Visualization      │
           └───────────────────────┘


---

## ✅ Notes

- The code is fully portable: plots are saved under `src/plots/` relative to the project folder, regardless of where `main.py` is executed.  
- FGSM attack generation is integrated in `src/visualization/visualize.py` via `fgsm()` from `src/training/attack.py`.  
- All plots are overwritten automatically if a file with the same name exists.  
- Models are loaded from checkpoint if available; otherwise, training occurs automatically.

