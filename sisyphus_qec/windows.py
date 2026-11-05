"""Detector time windows for circuit-level detector error models.

A window is a range of consecutive detectors; its sub-problem contains the
faults whose detectors all lie inside it. Solutions of a window, padded with
zeros, are solutions of the full problem.
"""
from __future__ import annotations

import numpy as np

from .gf2 import _rref_m4ri_order, pack_gf2_matrix_u64


def _detector_windows(check_H, colsig, levels, min_size, m4ri_k):
    """Dyadic detector windows of the check rows (see WINDOWS): for j = 1..levels,
    windows [a, a + s) with s = ceil(m / 2^j) and a = 0, s/2, s, ... Each window's
    sub-instance is the set of columns whose every check lies in it. Windows with no
    possible logical (all signature words zero), full rank, or the same column set
    as a smaller window are dropped. Returns a list of
    (cols, Hp_sub, colsig_sub, n_sub, nw_sub, rank_sub, level, a, s)."""
    H = np.ascontiguousarray(check_H, dtype=np.uint8)
    m, n = H.shape
    nz = H.any(axis=0)
    lo = np.where(nz, H.argmax(axis=0), m)                   # first check of each column
    hi = np.where(nz, m - 1 - H[::-1].argmax(axis=0), -1)    # last check
    out, seen = [], set()
    for j in range(levels, 0, -1):                           # small windows first
        sz = -(-m // (1 << j))
        if sz < min_size:
            continue
        step = max(1, sz // 2)
        for a in range(0, max(1, m - step), step):
            b = min(m, a + sz)
            cols = np.nonzero((lo >= a) & (hi < b))[0]
            if cols.size < 2:
                continue
            key = (int(cols[0]), int(cols[-1]), int(cols.size))
            if key in seen:
                continue
            cs = np.ascontiguousarray(colsig[cols])
            if not cs.any():
                continue
            sub = H[a:b][:, cols]
            sub = sub[sub.any(axis=1)]
            Hp_s = pack_gf2_matrix_u64(np.ascontiguousarray(sub))
            nw_s = (cols.size + 63) // 64
            tab = np.zeros((1 << m4ri_k, nw_s), dtype=np.uint64)
            Hs = Hp_s.copy()
            rk = int(_rref_m4ri_order(Hs, np.arange(cols.size), m4ri_k, tab,
                                      np.empty(Hp_s.shape[0], dtype=np.int64)))
            if rk >= cols.size or rk == 0:
                continue
            seen.add(key)
            out.append((cols.astype(np.int64), Hp_s, cs, int(cols.size), nw_s, rk, j, a, b - a))
    return out

def _expand_window_buf(buf, cols, nw):
    """Rows of packed window codewords -> packed full-width rows (vectorized)."""
    cnt = buf.shape[0]
    if cnt == 0:
        return np.zeros((0, nw), dtype=np.uint64)
    bits = np.unpackbits(np.ascontiguousarray(buf).view(np.uint8), axis=1,
                         bitorder="little")[:, :cols.size]
    full = np.zeros((cnt, nw * 64), dtype=np.uint8)
    full[:, cols] = bits
    return np.packbits(full, axis=1, bitorder="little").view(np.uint64).reshape(cnt, nw)

def _expand_window_cw(cw_sub, cols, nw):
    """Packed codeword of a window sub-instance -> packed full-width codeword."""
    out = np.zeros(nw, dtype=np.uint64)
    bits = np.unpackbits(cw_sub.view(np.uint8), bitorder="little")[:cols.size]
    full = cols[np.nonzero(bits)[0]]
    np.bitwise_or.at(out, full >> 6, np.left_shift(np.uint64(1), (full & 63).astype(np.uint64)))
    return out
