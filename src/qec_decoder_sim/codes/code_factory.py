from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import numpy as np
from scipy.sparse import csc_matrix, identity, kron

from .gf2 import CSSLogicals, compute_css_logicals


@dataclass(frozen=True)
class BBCodeSpec:
    l: int
    m: int
    a_x_pows: tuple[int, ...]
    a_y_pows: tuple[int, ...]
    b_x_pows: tuple[int, ...]
    b_y_pows: tuple[int, ...]


@dataclass(frozen=True)
class BBCSSCode:
    hx: np.ndarray
    hz: np.ndarray
    logicals: CSSLogicals


@dataclass(frozen=True)
class BBBlockCode:
    code: BBCSSCode
    a_blocks: tuple[csc_matrix, ...]
    b_blocks: tuple[csc_matrix, ...]


@dataclass(frozen=True)
class DecodingView:
    check_matrix: np.ndarray
    observables: np.ndarray


def _create_circulant_matrix(size: int, powers: tuple[int, ...]) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.uint8)
    for col in range(size):
        for power in powers:
            matrix[(col + power) % size, col] = 1
    return matrix


def _build_bb_sparse_blocks(spec: BBCodeSpec) -> tuple[tuple[csc_matrix, ...], tuple[csc_matrix, ...]]:
    shift_l = _create_circulant_matrix(spec.l, (-1,))
    shift_m = _create_circulant_matrix(spec.m, (-1,))
    x = kron(shift_l, identity(spec.m, dtype=np.uint8), format="csc")
    y = kron(identity(spec.l, dtype=np.uint8), shift_m, format="csc")

    a_blocks = tuple((x**power).astype(np.uint8) for power in spec.a_x_pows) + tuple(
        (y**power).astype(np.uint8) for power in spec.a_y_pows
    )
    b_blocks = tuple((y**power).astype(np.uint8) for power in spec.b_y_pows) + tuple(
        (x**power).astype(np.uint8) for power in spec.b_x_pows
    )
    return a_blocks, b_blocks


def build_bb_css_code(spec: BBCodeSpec) -> BBCSSCode:
    a_blocks, b_blocks = _build_bb_sparse_blocks(spec)
    a_matrix = reduce(lambda lhs, rhs: lhs + rhs, a_blocks).toarray().astype(np.uint8) % 2
    b_matrix = reduce(lambda lhs, rhs: lhs + rhs, b_blocks).toarray().astype(np.uint8) % 2

    hx = np.hstack((a_matrix, b_matrix)).astype(np.uint8)
    hz = np.hstack((b_matrix.T, a_matrix.T)).astype(np.uint8)
    logicals = compute_css_logicals(hx, hz)
    return BBCSSCode(hx=hx, hz=hz, logicals=logicals)


def build_bb_block_code(spec: BBCodeSpec) -> BBBlockCode:
    a_blocks, b_blocks = _build_bb_sparse_blocks(spec)
    return BBBlockCode(code=build_bb_css_code(spec), a_blocks=a_blocks, b_blocks=b_blocks)


def build_decoding_view(code: BBCSSCode, basis: str) -> DecodingView:
    basis_lower = basis.lower()
    if basis_lower == "x":
        return DecodingView(check_matrix=code.hz, observables=code.logicals.lz)
    if basis_lower == "z":
        return DecodingView(check_matrix=code.hx, observables=code.logicals.lx)
    raise ValueError("basis must be either 'x' or 'z'.")