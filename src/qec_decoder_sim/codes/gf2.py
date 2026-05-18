from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def row_echelon(mat: np.ndarray, *, reduced: bool = False) -> tuple[np.ndarray, int, np.ndarray, list[int]]:
    m, n = mat.shape
    matrix = np.array(mat, copy=True, dtype=bool)
    transform = np.identity(m, dtype=bool)
    pivot_row = 0
    pivot_cols: list[int] = []

    for col in range(n):
        if pivot_row >= m:
            break

        if not matrix[pivot_row, col]:
            swap_row_index = pivot_row + np.argmax(matrix[pivot_row:m, col])
            if matrix[swap_row_index, col]:
                matrix[[swap_row_index, pivot_row]] = matrix[[pivot_row, swap_row_index]]
                transform[[swap_row_index, pivot_row]] = transform[[pivot_row, swap_row_index]]

        if matrix[pivot_row, col]:
            if reduced:
                elimination_rows = [row for row in range(m) if row != pivot_row]
            else:
                elimination_rows = list(range(pivot_row + 1, m))
            for row in elimination_rows:
                if matrix[row, col]:
                    matrix[row] ^= matrix[pivot_row]
                    transform[row] ^= transform[pivot_row]
            pivot_cols.append(col)
            pivot_row += 1

    return matrix.astype(np.uint8), pivot_row, transform.astype(np.uint8), pivot_cols


def rank(mat: np.ndarray) -> int:
    return row_echelon(mat)[1]


def kernel(mat: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    transpose = mat.T
    m, _ = transpose.shape
    _, matrix_rank, transform, pivot_cols = row_echelon(transpose)
    return transform[matrix_rank:m], matrix_rank, pivot_cols


def row_basis(mat: np.ndarray) -> np.ndarray:
    _, _, _, pivot_cols = row_echelon(mat.T)
    if not pivot_cols:
        return np.zeros((0, mat.shape[1]), dtype=np.uint8)
    return np.array(mat[pivot_cols], dtype=np.uint8)


@dataclass(frozen=True)
class CSSLogicals:
    lx: np.ndarray
    lz: np.ndarray


def _compute_nontrivial_logicals(ker_checks: np.ndarray, image_other_checks: np.ndarray) -> np.ndarray:
    if ker_checks.size == 0:
        return np.zeros((0, image_other_checks.shape[1] if image_other_checks.size else 0), dtype=np.uint8)

    stack = np.vstack([image_other_checks, ker_checks]) if image_other_checks.size else np.array(ker_checks, copy=True)
    _, _, _, pivots = row_echelon(stack.T)
    image_rows = image_other_checks.shape[0]
    logical_row_indices = [index for index in range(image_rows, stack.shape[0]) if index in pivots]
    if not logical_row_indices:
        return np.zeros((0, stack.shape[1]), dtype=np.uint8)
    return np.array(stack[logical_row_indices], dtype=np.uint8)


def compute_css_logicals(hx: np.ndarray, hz: np.ndarray) -> CSSLogicals:
    hz_perp, _, _ = kernel(hz)
    hx_perp, _, _ = kernel(hx)
    hx_basis = row_basis(hx)
    hz_basis = row_basis(hz)

    lx = _compute_nontrivial_logicals(hz_perp, hx_basis)
    lz = _compute_nontrivial_logicals(hx_perp, hz_basis)
    return CSSLogicals(lx=lx, lz=lz)