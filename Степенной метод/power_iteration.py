import numpy as np


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    n = data.shape[0]

    x = np.random.rand(n)  # Случайный ненулевой вектор
    x = x / np.linalg.norm(x)  # Нормировка вектора

    for _ in range(num_steps):
        x_new = np.dot(data, x)  # Умножение матрицы на вектор
        c = np.linalg.norm(x_new)  # Коэффициент нормировки

        u = x_new / c  # Нормированный вектор

        x = u

    lambda_ = np.dot(np.dot(data, u), u)  # Приближенное значение для наибольшего собственного числа

    return float(lambda_), u