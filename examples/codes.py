"""Constructions used by the examples and tests (plain numpy)."""
import numpy as np


def bivariate_bicycle(l, m, a_terms, b_terms):
    """Bivariate bicycle code of Bravyi et al. (2024): Hx = [A | B], Hz = [B^T | A^T]
    with A, B sums of monomials x^i y^j on Z_l x Z_m, given as lists of (i, j)."""
    x = np.kron(np.roll(np.eye(l, dtype=np.uint8), 1, axis=1), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), np.roll(np.eye(m, dtype=np.uint8), 1, axis=1))

    def poly(terms):
        P = np.zeros((l * m, l * m), dtype=np.uint8)
        for i, j in terms:
            P ^= (np.linalg.matrix_power(x, i) @ np.linalg.matrix_power(y, j) % 2).astype(np.uint8)
        return P

    A, B = poly(a_terms), poly(b_terms)
    return np.hstack([A, B]), np.hstack([B.T, A.T])


def gross_code():
    """[[144,12,12]] gross code: A = x^3 + y + y^2, B = y^3 + x + x^2 on Z_12 x Z_6."""
    return bivariate_bicycle(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])


def bb72_code():
    """[[72,12,6]] bivariate bicycle code (same polynomials on Z_6 x Z_6)."""
    return bivariate_bicycle(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
