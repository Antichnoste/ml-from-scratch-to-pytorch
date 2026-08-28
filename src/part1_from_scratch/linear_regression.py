"""
Линейная регрессия "с нуля": MSE как функция потерь, аналитический градиент,
обучение через gradient_descent.py. В конце — сравнение со sklearn.LinearRegression
на датасете Diabetes (встроен в sklearn, не требует скачивания).
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gradient_descent import gradient_descent


def mse_loss(X_aug: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """MSE = (1/n) * sum((y_pred - y)^2)"""
    y_pred = X_aug @ weights
    return float(np.mean((y_pred - y) ** 2))


def mse_grad(X_batch: np.ndarray, y_batch: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Градиент MSE по весам: dL/dw = (2/n) * X^T @ (X @ w - y)
    X_batch уже содержит столбец единиц (bias).
    """
    n = X_batch.shape[0]
    y_pred = X_batch @ weights
    error = y_pred - y_batch
    return (2.0 / n) * (X_batch.T @ error)


class LinearRegressionScratch:
    """Обёртка в духе sklearn API: fit / predict."""

    def __init__(self, lr: float = 0.1, n_epochs: int = 300, batch_size: int | None = None):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weights_ = None
        self.history_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        self.weights_, self.history_ = gradient_descent(
            grad_fn=mse_grad,
            X=X,
            y=y,
            n_features=n_features,
            lr=self.lr,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            loss_fn=mse_loss,
            verbose=False,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_aug @ self.weights_


def run_comparison():
    """Обучает обе версии на датасете Diabetes (встроен в sklearn) и печатает сравнение."""
    data = load_diabetes()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # масштабирование признаков — важно для сходимости GD
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- наша реализация ---
    model_scratch = LinearRegressionScratch(lr=0.1, n_epochs=300)
    model_scratch.fit(X_train_scaled, y_train)
    pred_scratch = model_scratch.predict(X_test_scaled)
    mse_scratch = mean_squared_error(y_test, pred_scratch)

    # --- sklearn ---
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_train_scaled, y_train)
    pred_sklearn = model_sklearn.predict(X_test_scaled)
    mse_sklearn = mean_squared_error(y_test, pred_sklearn)

    print("=" * 50)
    print("Линейная регрессия: GD с нуля vs sklearn")
    print("=" * 50)
    print(f"MSE (с нуля, GD):        {mse_scratch:.4f}")
    print(f"MSE (sklearn, OLS):      {mse_sklearn:.4f}")
    print(f"Разница в MSE:           {abs(mse_scratch - mse_sklearn):.4f}")
    print()
    print("Веса (первые 5, с нуля): ", np.round(model_scratch.weights_[1:6], 4))
    print("Веса (первые 5, sklearn):", np.round(model_sklearn.coef_[:5], 4))

    return {
        "mse_scratch": mse_scratch,
        "mse_sklearn": mse_sklearn,
        "history": model_scratch.history_,
    }


if __name__ == "__main__":
    run_comparison()