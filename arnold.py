from __future__ import annotations

import numpy as np


def _arnold_indices(n: int):
    """Предвычисляет индексные массивы для прямого Arnold (A = [[1,1],[1,2]])."""
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    xp = (x + y) % n
    yp = (x + 2 * y) % n
    return xp, yp


def _inv_arnold_indices(n: int):
    """Предвычисляет индексные массивы для обратного Arnold (A^{-1} = [[2,-1],[-1,1]])."""
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    xp = (2 * x - y) % n
    yp = (-x + y) % n
    return xp, yp


def arnold_transform(img: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Прямое преобразование Арнольда (векторизованное) для массива NxN.
    A = [[1,1],[1,2]] mod N: out[x,y] → tmp[(x+y)%n, (x+2y)%n].
    """
    a = np.asarray(img)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Arnold transform requires a square 2D array.")
    n = a.shape[0]
    xp, yp = _arnold_indices(n)
    out = a.copy()
    for _ in range(int(iterations)):
        tmp = np.empty_like(out)
        tmp[xp, yp] = out          # векторизованная запись
        out = tmp
    return out


def inverse_arnold_transform(img: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Обратное преобразование Арнольда (векторизованное) для массива NxN.
    A^{-1} = [[2,-1],[-1,1]] mod N: out[x,y] → tmp[(2x-y)%n, (-x+y)%n].
    """
    a = np.asarray(img)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Arnold transform requires a square 2D array.")
    n = a.shape[0]
    xp, yp = _inv_arnold_indices(n)
    out = a.copy()
    for _ in range(int(iterations)):
        tmp = np.empty_like(out)
        tmp[xp, yp] = out          # векторизованная запись с обратной матрицей
        out = tmp
    return out



