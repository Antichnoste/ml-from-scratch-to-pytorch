"""
Логистическая регрессия "с нуля": sigmoid + binary cross-entropy,
обучение через gradient_descent.py. Сравнение со sklearn.LogisticRegression
на Breast Cancer Wisconsin — маленький, чистый датасет для бинарной классификации.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gradient_descent import gradient_descent


def sigmoid(z: np.ndarray) -> np.ndarray:
    # clip для численной стабильности (избегаем overflow в exp)
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def bce_loss(X_aug: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Binary cross-entropy: -(1/n) * sum(y*log(p) + (1-y)*log(1-p))"""
    z = X_aug @ weights
    p = sigmoid(z)
    eps = 1e-15  # чтобы не брать log(0)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bce_grad(X_batch: np.ndarray, y_batch: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Градиент BCE по весам (форма идентична градиенту MSE в линейной регрессии,
    потому что sigmoid + BCE математически "сокращаются" красиво):
    dL/dw = (1/n) * X^T @ (sigmoid(X @ w) - y)
    """
    n = X_batch.shape[0]
    p = sigmoid(X_batch @ weights)
    error = p - y_batch
    return (1.0 / n) * (X_batch.T @ error)


class LogisticRegressionScratch:
    def __init__(self, lr: float = 0.1, n_epochs: int = 300, batch_size: int | None = None):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weights_ = None
        self.history_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        self.weights_, self.history_ = gradient_descent(
            grad_fn=bce_grad,
            X=X,
            y=y,
            n_features=n_features,
            lr=self.lr,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            loss_fn=bce_loss,
            verbose=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        return sigmoid(X_aug @ self.weights_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def run_comparison():
    """Обучает обе версии на Breast Cancer Wisconsin и печатает сравнение метрик."""
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- наша реализация ---
    model_scratch = LogisticRegressionScratch(lr=0.1, n_epochs=300)
    model_scratch.fit(X_train_scaled, y_train)
    pred_scratch = model_scratch.predict(X_test_scaled)
    proba_scratch = model_scratch.predict_proba(X_test_scaled)

    # --- sklearn ---
    model_sklearn = LogisticRegression(max_iter=1000)
    model_sklearn.fit(X_train_scaled, y_train)
    pred_sklearn = model_sklearn.predict(X_test_scaled)
    proba_sklearn = model_sklearn.predict_proba(X_test_scaled)[:, 1]

    def print_metrics(name, y_true, y_pred, y_proba):
        print(f"--- {name} ---")
        print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
        print(f"F1:        {f1_score(y_true, y_pred):.4f}")
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")
        print(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")
        print()

    print("=" * 50)
    print("Логистическая регрессия: GD с нуля vs sklearn")
    print("=" * 50)
    print_metrics("С нуля (GD)", y_test, pred_scratch, proba_scratch)
    print_metrics("sklearn", y_test, pred_sklearn, proba_sklearn)

    return {
        "scratch": {"y_pred": pred_scratch, "y_proba": proba_scratch},
        "sklearn": {"y_pred": pred_sklearn, "y_proba": proba_sklearn},
        "y_test": y_test,
        "history": model_scratch.history_,
    }


if __name__ == "__main__":
    run_comparison()