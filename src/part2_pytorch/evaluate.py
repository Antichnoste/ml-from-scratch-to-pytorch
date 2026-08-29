"""
Оценка обученной FashionCNN на test set.

Выводит:
- итоговую accuracy на test set (ключевая цифра для README/резюме)
- confusion matrix по 10 классам
- classification report (precision/recall/f1 по каждому классу)
- сохраняет несколько примеров, где модель ошиблась (для notebooks/04)
"""

import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import FashionCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_test_loader(batch_size: int = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    test_set = datasets.FashionMNIST(
        root="../../data", train=False, download=True, transform=transform
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2), test_set


@torch.no_grad()
def evaluate(model, loader):
    """Прогоняет модель по всему test set, собирает предсказания и вероятности."""
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []

    for images, labels in loader:
        images_device = images.to(DEVICE)
        outputs = model(images_device)
        preds = outputs.argmax(dim=1).cpu()

        all_preds.append(preds)
        all_labels.append(labels)
        all_images.append(images)

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_images).numpy(),
    )


def find_misclassified(preds, labels, images, n_examples: int = 8):
    """Возвращает индексы первых n_examples ошибок модели — для визуализации в notebook."""
    wrong_idx = np.where(preds != labels)[0]
    return wrong_idx[:n_examples]


def main():
    test_loader, test_set = get_test_loader()

    model = FashionCNN().to(DEVICE)
    model.load_state_dict(torch.load("fashion_cnn_best.pth", map_location=DEVICE))

    preds, labels, images = evaluate(model, test_loader)

    accuracy = (preds == labels).mean()
    print("=" * 60)
    print(f"Test accuracy: {accuracy:.4f}")
    print("=" * 60)

    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=3))

    cm = confusion_matrix(labels, preds)
    print("Confusion matrix (строки=истина, столбцы=предсказание):")
    print(cm)

    wrong_idx = find_misclassified(preds, labels, images, n_examples=8)
    print(f"\nНайдено {(preds != labels).sum()} ошибок из {len(labels)} примеров.")
    print(f"Индексы первых {len(wrong_idx)} ошибок сохранены в misclassified_indices.json")

    results = {
        "test_accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "misclassified_indices": wrong_idx.tolist(),
        "misclassified_true_labels": labels[wrong_idx].tolist(),
        "misclassified_pred_labels": preds[wrong_idx].tolist(),
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()