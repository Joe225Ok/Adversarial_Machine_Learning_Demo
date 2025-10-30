# src/training/attack.py

import torch
import torch.nn as nn

def fgsm(model, images, labels, eps, device="cpu"):
    """
    Single-step FGSM on a batch (untargeted).
    images: tensor (N,C,H,W) in [0,1] (or normalized if model expects that)
    returns adv_images (clamped to valid range)
    """
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad = images.grad.data
    adv = images + eps * grad.sign()
    adv = torch.clamp(adv, 0.0, 1.0)
    return adv.detach()

# def pgd(model, images, labels, eps=0.03, alpha=0.007, iters=10, device="cpu"):
#     """
#     Basic PGD (projected gradient descent) implementation.
#     """
#     images = images.clone().detach().to(device)
#     labels = labels.to(device)
#     ori_images = images.clone().detach()

#     adv_images = images + torch.zeros_like(images).uniform_(-eps, eps)
#     adv_images = torch.clamp(adv_images, 0.0, 1.0)
#     for i in range(iters):
#         adv_images.requires_grad = True
#         outputs = model(adv_images)
#         loss = nn.CrossEntropyLoss()(outputs, labels)
#         model.zero_grad()
#         loss.backward()
#         grad = adv_images.grad.data
#         adv_images = adv_images + alpha * grad.sign()
#         eta = torch.clamp(adv_images - ori_images, -eps, eps)
#         adv_images = torch.clamp(ori_images + eta, 0.0, 1.0).detach()
#     return adv_images

def evaluate_fgsm(model, test_loader, eps, device="cpu", max_batches=None):
    """
    Evaluate clean accuracy and FGSM-robust accuracy on the provided test_loader.
    Returns (clean_acc, adv_acc, total_samples)
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    correct_clean = 0
    correct_adv = 0
    total = 0

    with torch.no_grad():
        for b_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # clean
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct_clean += (preds == labels).sum().item()

            # adversarial (need gradients -> no_grad not used)
            adv_images = fgsm(model, images, labels, eps, device=device)
            outputs_adv = model(adv_images)
            preds_adv = outputs_adv.argmax(dim=1)
            correct_adv += (preds_adv == labels).sum().item()

            total += labels.size(0)

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    clean_acc = correct_clean / total
    adv_acc = correct_adv / total
    return clean_acc, adv_acc, total
