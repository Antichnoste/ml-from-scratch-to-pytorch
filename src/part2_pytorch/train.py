"""
Обучение FashionCNN на Fashion-MNIST.

По умолчанию:
- скачивает Fashion-MNIST в ./data (один раз, дальше берётся из кэша)
- делит train на train/val (90%/10%) — test set не трогаем до самого конца
- обучает n_epochs эпох, логирует loss/accuracy по эпохам
- сохраняет веса лучшей (по val accuracy) модели в fashion_cnn_best.pth
- сохраняет историю обучения в history.json (для графиков в notebooks/04)
"""

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from cnn_model import FashionCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_dataloaders(batch_size: int = 64, val_fraction: float = 0.1, seed: int = 42):
    """Скачивает Fashion-MNIST, делит train на train/val, возвращает DataLoader'ы."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_full = datasets.FashionMNIST(
        root="../../data", train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root="../../data", train=False, download=True, transform=transform
    )

    n_val = int(len(train_full) * val_fraction)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer=None):
    """
    Один проход по данным. Если optimizer передан — режим обучения (с backward),
    иначе — режим оценки (torch.no_grad, без обновления весов).
    Возвращает (средний loss, accuracy) за эпоху.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train(n_epochs: int = 15, lr: float = 1e-3, batch_size: int = 64):
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    model = FashionCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"Обучение на устройстве: {DEVICE}")
    print("=" * 60)

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Эпоха {epoch:2d}/{n_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "fashion_cnn_best.pth")
            print(f"  -> новый лучший результат, модель сохранена (val_acc={val_acc:.4f})")

    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 60)
    print(f"Лучшая val accuracy: {best_val_acc:.4f}")
    print("Веса сохранены в fashion_cnn_best.pth, история — в history.json")

    return model, history


if __name__ == "__main__":
    train(n_epochs=15, lr=1e-3, batch_size=64)