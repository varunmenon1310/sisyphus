"""Bit-packed linear algebra over GF(2).

Matrices are stored row-wise as uint64 words (bit j of word w holds column
64*w + j). Row reduction uses the Method of Four Russians (M4RI): k pivots
are eliminated per pass with a table of all 2^k combinations of their rows.
Also: kernels, random combinations of rows, and the per-column parity words
used to verify that a candidate is nontrivial.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .params import BITORDER


def pack_gf2_matrix_u64(mat):
    m, n = mat.shape
    n_words = (n + 63) // 64
    padded = np.zeros((m, n_words * 64), dtype=np.uint8)
    padded[:, :n] = mat
    packed = np.packbits(padded, axis=1, bitorder=BITORDER)
    return np.ascontiguousarray(packed).view(np.uint64)

def unpack_gf2_matrix_u64(packed_u64, n_cols):
    packed_u8 = np.ascontiguousarray(packed_u64).view(np.uint8)
    return np.unpackbits(packed_u8, axis=1, count=n_cols, bitorder=BITORDER)

@njit(inline='always', cache=True)
def _popcount_u64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

@njit(inline='always', cache=True)
def _bit(A, i, col):
    """Entry (i, col) of a packed matrix."""
    return int((A[i, col >> 6] >> np.uint64(col & 63)) & np.uint64(1))

@njit(inline='always', cache=True)
def _xor_weight_row(x, A, i, nw):
    """Hamming weight of x XOR A[i] over the first nw words. A straight loop:
    it vectorizes, and beats every early-abort variant for rows up to a few
    hundred words."""
    wt = 0
    for w in range(nw):
        wt += int(_popcount_u64(x[w] ^ A[i, w]))
    return wt

@njit(cache=True)
def _rref_m4ri_order(A, perm, k, tab, pivot_col):
    """Reduced row echelon form of A, choosing pivot columns in the order given
    by `perm`, eliminating k pivots per pass ("method of four Russians"): the
    k pivot rows are reduced against each other, a 2^k table of their sums is
    built, and every other row is fixed with a single table lookup. Records
    pivot_col[row] for each pivot row (-1 otherwise). Returns the rank."""
    m, nwt = A.shape
    n_proc = perm.shape[0]
    for i in range(m):
        pivot_col[i] = -1
    cols = np.empty(k, dtype=np.int64)
    row = 0
    ci = 0
    while ci < n_proc and row < m:
        # -- collect up to k pivots among rows[row:] for the next columns --
        q = 0
        while q < k and ci < n_proc and row + q < m:
            c = perm[ci]
            ci += 1
            cw = c >> 6
            cb = np.uint64(c & 63)
            found = -1
            for i in range(row + q, m):
                # bit c of row i after reduction by the q pivots found so far
                b = int((A[i, cw] >> cb) & np.uint64(1))
                for t in range(q):
                    if _bit(A, i, cols[t]):
                        b ^= int((A[row + t, cw] >> cb) & np.uint64(1))
                if b:
                    found = i
                    break
            if found < 0:
                continue                      # column c has no pivot
            if found != row + q:
                for w in range(nwt):
                    tmp = A[found, w]
                    A[found, w] = A[row + q, w]
                    A[row + q, w] = tmp
            for t in range(q):                # reduce the new pivot row
                if _bit(A, row + q, cols[t]):
                    for w in range(nwt):
                        A[row + q, w] ^= A[row + t, w]
            for t in range(q):                # and the earlier ones by it
                if (A[row + t, cw] >> cb) & np.uint64(1):
                    for w in range(nwt):
                        A[row + t, w] ^= A[row + q, w]
            cols[q] = c
            pivot_col[row + q] = c
            q += 1
        if q == 0:
            break
        # -- table of all sums of the q pivot rows, indexed by bit pattern --
        for w in range(nwt):
            tab[0, w] = 0
        for t in range(q):
            idx = 1 << t
            for w in range(nwt):
                tab[idx, w] = A[row + t, w]
            for p in range(idx + 1, 2 * idx):
                for w in range(nwt):
                    tab[p, w] = tab[p - idx, w] ^ tab[idx, w]
        # -- one pass eliminates the q columns from every other row --
        for i in range(m):
            if i >= row and i < row + q:
                continue
            p = 0
            for t in range(q):
                p |= _bit(A, i, cols[t]) << t
            if p != 0:
                for w in range(nwt):
                    A[i, w] ^= tab[p, w]
        row += q
    return row

@njit(cache=True)
def _nullspace_packed_u64(H_packed, n_cols, k, tab):
    """Basis of ker(H) over GF(2), packed. Reduce H, then one basis vector per
    free column: a 1 there plus the pivot columns of the rows that contain it."""
    m, nw = H_packed.shape
    A = H_packed.copy()
    perm = np.arange(n_cols)
    pivot_col = np.empty(m, dtype=np.int64)
    rank = _rref_m4ri_order(A, perm, k, tab, pivot_col)
    is_piv = np.zeros(n_cols, dtype=np.bool_)
    for r in range(rank):
        is_piv[pivot_col[r]] = True
    N = np.zeros((n_cols - rank, nw), dtype=np.uint64)
    j = 0
    for f in range(n_cols):
        if is_piv[f]:
            continue
        N[j, f >> 6] |= np.uint64(1) << np.uint64(f & 63)
        for r in range(rank):
            if _bit(A, r, f):
                pc = pivot_col[r]
                N[j, pc >> 6] |= np.uint64(1) << np.uint64(pc & 63)
        j += 1
    return N

def _kernel_packed(H, n_cols, k):
    """Packed basis of ker(H) for a uint8 matrix H."""
    tab = np.zeros((1 << k, (n_cols + 63) // 64), dtype=np.uint64)
    return _nullspace_packed_u64(pack_gf2_matrix_u64(H), n_cols, k, tab)

@njit(cache=True)
def _random_combinations(P_packed, coeff):
    """coeff (q, r) in {0,1} -> the q sums of rows of P_packed it selects."""
    q, r = coeff.shape
    nw = P_packed.shape[1]
    out = np.zeros((q, nw), dtype=np.uint64)
    for a in range(q):
        for b in range(r):
            if coeff[a, b]:
                for w in range(nw):
                    out[a, w] ^= P_packed[b, w]
    return out

@njit(cache=True)
def _pairing_words(G_packed, R_packed, ns):
    """S[i] packs the parities <G_i, R_j> for j < 64 ns."""
    m, nw = G_packed.shape
    nproj = R_packed.shape[0]
    S = np.zeros((m, ns), dtype=np.uint64)
    for i in range(m):
        for j in range(nproj):
            acc = np.uint64(0)
            for w in range(nw):
                acc ^= G_packed[i, w] & R_packed[j, w]
            if _popcount_u64(acc) & np.uint64(1):
                S[i, j >> 6] |= np.uint64(1) << np.uint64(j & 63)
    return S

@njit(cache=True)
def _sig_transpose(R_packed, n_cols, ns):
    """colsig[c] packs the parities <e_c, R_j> = R_j[c] for j < 64 ns: the
    certification words of the unit vector e_c. The certification words of any
    codeword are the XOR of colsig over its support."""
    nproj = R_packed.shape[0]
    out = np.zeros((n_cols, ns), dtype=np.uint64)
    for j in range(nproj):
        jw = j >> 6
        jb = np.uint64(1) << np.uint64(j & 63)
        for c in range(n_cols):
            if (R_packed[j, c >> 6] >> np.uint64(c & 63)) & np.uint64(1):
                out[c, jw] |= jb
    return out
