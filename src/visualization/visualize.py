# src/visualization/visualize.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from training.attack import fgsm


# ----------------------------------------------------
#  Denormalization
# ----------------------------------------------------
def denormalize(image, dataset_name="CIFAR10"):
    """
    Converts normalized tensors back into displayable images.
    """
    image = image.detach()

    if dataset_name == "MNIST":
        mean = 0.1307
        std = 0.3081
        img = image * std + mean
        img = img.squeeze().cpu().numpy()
        return np.clip(img, 0, 1)

    # CIFAR10 or Caltech101
    if dataset_name == "CIFAR10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    else:
        # Caltech101
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    img = image.clone().cpu()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)

    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


# ----------------------------------------------------
#  Adversarial Visualization
# ----------------------------------------------------
def visualize_adversarial_examples(
        model, dataloader, epsilon, class_names,
        device, dataset_name="CIFAR10", max_examples=5):
    """
    Visualize strongest FGSM adversarial attacks with all features:
    denormalization, dataset auto-detect, filtering of BACKGROUND_Google, Faces, Faces_easy,
    strongest-attack ranking, readable class names, and confidence percentages.
    """
    model.eval()

    # ---------------------------------------------
    # Load one batch
    # ---------------------------------------------
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    # ---------------------------------------------
    # Generate FGSM adversarial examples
    # ---------------------------------------------
    adv_images = fgsm(model, images, labels, epsilon, device=device)

    with torch.no_grad():
        orig_logits = model(images)
        adv_logits = model(adv_images)

        orig_probs = torch.softmax(orig_logits, dim=1)
        adv_probs = torch.softmax(adv_logits, dim=1)

        orig_preds = torch.argmax(orig_probs, dim=1)
        adv_preds = torch.argmax(adv_probs, dim=1)

    # ---------------------------------------------
    # Filter unwanted classes: BACKGROUND_Google, Faces, Faces_easy
    # ---------------------------------------------
    excluded_classes = []
    for c in ["BACKGROUND_Google", "Faces", "Faces_easy"]:
        if c in class_names:
            excluded_classes.append(class_names.index(c))

    success_mask = (orig_preds == labels) & (adv_preds != labels)
    success_indices = torch.where(success_mask)[0].tolist()

    filtered_indices = []
    for idx in success_indices:
        gt = labels[idx].item()
        op = orig_preds[idx].item()
        ap = adv_preds[idx].item()
        if any(x in excluded_classes for x in [gt, op, ap]):
            continue
        filtered_indices.append(idx)

    if len(filtered_indices) == 0:
        print("No successful attacks to show.")
        return

    # ---------------------------------------------
    # Rank attacks by strength (orig_conf * adv_conf)
    # ---------------------------------------------
    ranked = [(idx, orig_probs[idx, orig_preds[idx]].item() * adv_probs[idx, adv_preds[idx]].item())
              for idx in filtered_indices]
    ranked.sort(key=lambda x: x[1], reverse=True)
    top = ranked[:max_examples]
    selected = [i for (i, _) in top]

    print(f"\nShowing {len(selected)} strongest adversarial attacks:")

    # ---------------------------------------------
    # Slice tensors
    # ---------------------------------------------
    images_s = images[selected]
    adv_s = adv_images[selected]
    labels_s = labels[selected]
    orig_s = orig_preds[selected]
    adv_s_pred = adv_preds[selected]

    # ---------------------------------------------
    # Print summary with confidence
    # ---------------------------------------------
    print("\nPer-example summary:")
    for r, (idx, strength) in enumerate(top):
        gt = class_names[labels[idx]]
        op = class_names[orig_preds[idx]]
        ap = class_names[adv_preds[idx]]
        orig_conf = orig_probs[idx, orig_preds[idx]].item() * 100
        adv_conf = adv_probs[idx, adv_preds[idx]].item() * 100
        print(f"#{r}: idx={idx} | strength={strength:.4f} | "
              f"GT={gt} | Orig={op} ({orig_conf:.1f}%) -> Adv={ap} ({adv_conf:.1f}%)")

    # ---------------------------------------------
    # Plot images with confidence
    # ---------------------------------------------
    N = len(selected)
    fig, axes = plt.subplots(2, N, figsize=(3 * N, 6), dpi=150)
    if N == 1:
        axes = np.expand_dims(axes, axis=1)

    def imshow(ax, img_tensor, title):
        img = denormalize(img_tensor, dataset_name)
        ax.imshow(img, cmap="gray" if dataset_name == "MNIST" else None, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for i in range(N):
        orig_conf_pct = orig_probs[selected[i], orig_s[i]].item() * 100
        adv_conf_pct = adv_probs[selected[i], adv_s_pred[i]].item() * 100
        imshow(
            axes[0, i],
            images_s[i],
            title=f"GT:{class_names[labels_s[i]]}\nOrig:{class_names[orig_s[i]]} ({orig_conf_pct:.1f}%)"
        )
        imshow(
            axes[1, i],
            adv_s[i],
            title=f"Adv:{class_names[adv_s_pred[i]]} ({adv_conf_pct:.1f}%)"
        )

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "plots",
                             f"successful_attacks_FGSM_{dataset_name}_eps{epsilon}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved figure to {save_path}")
    plt.show()
