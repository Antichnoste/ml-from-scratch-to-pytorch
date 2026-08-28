"""
Реализация градиентного спуска с нуля.

Три варианта:
- batch GD: градиент считается по всей выборке на каждом шаге
- mini-batch GD: градиент считается по случайному подмножеству (batch_size)
- SGD: частный случай mini-batch при batch_size=1

Используется как общий "движок" для linear_regression.py и logistic_regression.py —
им нужно только передать свою функцию grad_fn(X, y, weights) -> gradient.
"""

import numpy as np


def gradient_descent(
    grad_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
    lr: float = 0.01,
    n_epochs: int = 200,
    batch_size: int | None = None,
    seed: int = 42,
    loss_fn=None,
    verbose: bool = False,
):
    """
    Универсальный градиентный спуск.

    Параметры
    ---------
    grad_fn : callable(X_batch, y_batch, weights) -> np.ndarray
        Возвращает градиент функции потерь по весам (включая bias как weights[0],
        если X_batch уже содержит столбец единиц).
    X : np.ndarray, shape (n_samples, n_features)
        Матрица признаков БЕЗ столбца единиц — он добавляется внутри.
    y : np.ndarray, shape (n_samples,)
        Целевая переменная.
    n_features : int
        Число признаков (без учёта bias).
    lr : float
        Learning rate.
    n_epochs : int
        Число проходов по данным.
    batch_size : int | None
        Если None — batch GD (вся выборка сразу).
        Если 1 — SGD.
        Иначе — mini-batch GD.
    loss_fn : callable(X, y, weights) -> float, optional
        Если передан, используется для логирования истории loss по эпохам.
    verbose : bool
        Печатать loss каждые 20 эпох.

    Возвращает
    ----------
    weights : np.ndarray, shape (n_features + 1,)
        weights[0] — bias (intercept), weights[1:] — коэффициенты признаков.
    history : list[float]
        Значения loss по эпохам (пусто, если loss_fn не передан).
    """
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]

    # добавляем столбец единиц для bias-члена
    X_aug = np.hstack([np.ones((n_samples, 1)), X])
    weights = np.zeros(n_features + 1)

    history = []
    bs = batch_size if batch_size is not None else n_samples

    for epoch in range(n_epochs):
        # перемешиваем данные каждую эпоху
        perm = rng.permutation(n_samples)
        X_shuffled, y_shuffled = X_aug[perm], y[perm]

        for start in range(0, n_samples, bs):
            end = start + bs
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            grad = grad_fn(X_batch, y_batch, weights)
            weights -= lr * grad

        if loss_fn is not None:
            current_loss = loss_fn(X_aug, y, weights)
            history.append(current_loss)
            if verbose and epoch % 20 == 0:
                print(f"epoch {epoch:4d} | loss = {current_loss:.6f}")

    return weights, history


def predict_trajectory_2d(grad_fn, loss_fn, start_point, lr=0.1, n_steps=50):
    """
    Вспомогательная функция для демонстрации GD на функции двух переменных
    (например, f(x, y) = x^2 + y^2) — удобно для визуализации в notebooks/01.

    grad_fn : callable(point: np.ndarray) -> np.ndarray, градиент в точке
    loss_fn : callable(point: np.ndarray) -> float, значение функции в точке
    start_point : np.ndarray, shape (2,)

    Возвращает список точек траектории (для contour plot).
    """
    point = np.array(start_point, dtype=float)
    trajectory = [point.copy()]

    for _ in range(n_steps):
        grad = grad_fn(point)
        point = point - lr * grad
        trajectory.append(point.copy())

    return np.array(trajectory)


if __name__ == "__main__":
    # Быстрая самопроверка: находим минимум f(x, y) = x^2 + y^2 (минимум в (0, 0))
    def f(p):
        return p[0] ** 2 + p[1] ** 2

    def grad_f(p):
        return np.array([2 * p[0], 2 * p[1]])

    traj = predict_trajectory_2d(grad_f, f, start_point=[4.0, 3.0], lr=0.1, n_steps=30)
    print("Старт:", traj[0])
    print("Финиш:", traj[-1])
    print(f"Значение функции в финише: {f(traj[-1]):.6f}")