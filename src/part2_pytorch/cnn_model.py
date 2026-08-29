"""
Простая CNN для классификации Fashion-MNIST (10 классов одежды/обуви, 28x28 grayscale).

Архитектура:
    Input (1, 28, 28)
    -> Conv2d(1->32, 3x3) -> ReLU -> MaxPool2d(2x2)   =>  (32, 14, 14)
    -> Conv2d(32->64, 3x3) -> ReLU -> MaxPool2d(2x2)  =>  (64, 7, 7)
    -> Flatten                                          => (64*7*7,)
    -> Linear(64*7*7 -> 128) -> ReLU -> Dropout
    -> Linear(128 -> 10)                                => логиты по 10 классам

Логика: два свёрточных блока постепенно увеличивают число фильтров (32 -> 64),
одновременно уменьшая пространственное разрешение (28 -> 14 -> 7) через pooling.
Первый блок ловит простые паттерны (края, текстуры), второй — более сложные
комбинации этих паттернов. В конце — обычный полносвязный классификатор поверх
извлечённых признаков.
"""

import torch
import torch.nn as nn


class FashionCNN(nn.Module):
    def __init__(self, n_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        # padding=1 при kernel_size=3 сохраняет пространственный размер после свёртки
        # (28x28 -> 28x28), уменьшение размера происходит только на MaxPool
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 28x28 -> 14x14
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: тензор формы (batch_size, 1, 28, 28)
        возвращает: логиты формы (batch_size, n_classes) — без softmax,
        т.к. nn.CrossEntropyLoss в train.py применяет softmax внутри себя
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = FashionCNN()
    dummy_input = torch.randn(8, 1, 28, 28)
    output = model(dummy_input)
    print(f"Вход:  {dummy_input.shape}")
    print(f"Выход: {output.shape}")
    assert output.shape == (8, 10), "Неверная форма выхода!"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Обучаемых параметров: {n_params:,}")