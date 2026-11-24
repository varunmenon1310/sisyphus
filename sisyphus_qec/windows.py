"""Detector time windows for circuit-level detector error models.

A window is a range of consecutive detectors; its sub-problem contains the
faults whose detectors all lie inside it. Solutions of a window, padded with
zeros, are solutions of the full problem.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .gf2 import _rref_m4ri_order, pack_gf2_matrix_u64


def _detector_windows(check_H, colsig, levels, min_size, m4ri_k):
    """Dyadic detector windows of the check rows (see WINDOWS): for j = 1..levels,
    windows [a, a + s) with s = ceil(m / 2^j) and a = 0, s/2, s, ... Each window's
    sub-instance is the set of columns whose every check lies in it. Windows with no
    possible logical (all signature words zero), full rank, or the same column set
    as a smaller window are dropped. Returns a list of
    (cols, Hp_sub, colsig_sub, n_sub, nw_sub, rank_sub, level, a, s, key), where key
    identifies the sub-instance up to a shift of the detector and column indices: two
    windows with the same key have identical check matrices and signature words in their
    local coordinates (time-translates of each other in a periodic memory circuit), so
    they are the same search problem."""
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
            h = hashlib.md5()
            h.update(np.array([b - a, cols.size], dtype=np.int64).tobytes())
            h.update(np.packbits(sub, axis=1).tobytes())
            h.update(cs.tobytes())
            wkey = h.hexdigest()
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
            out.append((cols.astype(np.int64), Hp_s, cs, int(cols.size), nw_s, rk, j, a, b - a,
                        wkey))
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


def _detector_period(check_H, colsig):
    """Smallest shift T of the detector index under which the middle half of the problem
    is invariant (the detectors per round of a memory circuit), or 0. Detector d gets a
    type: a hash of the shapes (detector offsets from d, signature words) of the columns
    whose first detector is d; T is the smallest shift that maps the type sequence of the
    middle half onto itself."""
    import scipy.sparse as sp
    H = np.ascontiguousarray(check_H, dtype=np.uint8)
    m, n = H.shape
    if m < 12:
        return 0
    C = sp.csc_matrix(H)
    indptr, idx = C.indptr, C.indices
    typ = [hashlib.md5() for _ in range(m)]
    for c in range(n):
        s, e = indptr[c], indptr[c + 1]
        if s == e:
            continue
        r = np.sort(idx[s:e])
        typ[r[0]].update(hashlib.md5((r - r[0]).tobytes() + colsig[c].tobytes()).digest())
    t = np.array([int.from_bytes(h.digest()[:8], "little") for h in typ], dtype=np.uint64)
    a, b = m // 4, (3 * m) // 4
    for T in range(1, m // 3):
        if b - T <= a:
            break
        if np.array_equal(t[a:b - T], t[a + T:b]):
            return T
    return 0


def _window_instance(H, colsig, lo, hi, a, b, level, m4ri_k, known=None):
    """Sub-instance of the detector window [a, b) (the columns whose checks all lie in
    it), as an entry of the _detector_windows list, or None if it has no possible
    logical or full rank. If its key is in `known`, returns the key only (no reduction)."""
    cols = np.nonzero((lo >= a) & (hi < b))[0]
    if cols.size < 2:
        return None
    cs = np.ascontiguousarray(colsig[cols])
    if not cs.any():
        return None
    sub = H[a:b][:, cols]
    h = hashlib.md5()
    h.update(np.array([b - a, cols.size], dtype=np.int64).tobytes())
    h.update(np.packbits(sub, axis=1).tobytes())
    h.update(cs.tobytes())
    key = h.hexdigest()
    if known is not None and key in known:
        return key
    sub = sub[sub.any(axis=1)]
    Hp_s = pack_gf2_matrix_u64(np.ascontiguousarray(sub))
    nw_s = (cols.size + 63) // 64
    tab = np.zeros((1 << m4ri_k, nw_s), dtype=np.uint64)
    rk = int(_rref_m4ri_order(Hp_s.copy(), np.arange(cols.size), m4ri_k, tab,
                              np.empty(Hp_s.shape[0], dtype=np.int64)))
    if rk >= cols.size or rk == 0:
        return None
    return (cols.astype(np.int64), Hp_s, cs, int(cols.size), nw_s, rk, level, a, b - a, key)


PERIOD_WINDOW_HALVES = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)


def _period_windows(check_H, colsig, T, m4ri_k, max_frac=0.5):
    """Detector windows aligned to the period T, in half periods: sizes k T / 2 (k in
    PERIOD_WINDOW_HALVES, size <= max_frac m), one window starting at detector 0, one
    ending at the last detector, and the windows in between starting every T / 2 from
    the phase p (the cut in [m/4, m/4 + T) crossed by the fewest columns, i.e. a round
    boundary). Light solutions of a memory circuit sit in the bulk (where all translates
    are equivalent) or at one of the two time boundaries, and the search inside a window
    gets much harder as the window grows (more equally light solutions to get stuck on),
    so each light solution should have a window only slightly longer than its span.
    Equivalent windows (same key) are kept once. Returns entries like _detector_windows
    with level = k (half periods), and the number of windows each entry stands for."""
    H = np.ascontiguousarray(check_H, dtype=np.uint8)
    m, n = H.shape
    nz = H.any(axis=0)
    lo = np.where(nz, H.argmax(axis=0), m)
    hi = np.where(nz, m - 1 - H[::-1].argmax(axis=0), -1)
    cross = np.zeros(m + 2, dtype=np.int64)          # columns with lo < d <= hi
    np.add.at(cross, lo[nz] + 1, 1)
    np.add.at(cross, hi[nz] + 1, -1)
    cross = np.cumsum(cross)[:m + 1]
    q0 = m // 4
    p = (q0 + int(np.argmin(cross[q0:q0 + T]))) % T
    half = max(1, T // 2)
    out, count = [], {}
    for k in PERIOD_WINDOW_HALVES:
        s = (k * T) // 2
        if s > max_frac * m:
            break
        starts = [0, m - s] + [a for a in range(p % half, m - s + 1, half) if 0 < a < m - s]
        for a in starts:
            e = _window_instance(H, colsig, lo, hi, a, a + s, k, m4ri_k, known=count)
            if e is None:
                continue
            if isinstance(e, str):
                count[e] += 1
                continue
            count[e[9]] = 1
            out.append(e)
    return out, [count[e[9]] for e in out]


class _SparseChecks:
    """Column-compressed copy of a check matrix (indptr, indices as int32), the first and
    last check of every column and the per-column signature words: what a worker needs
    to cut out a detector window on its own (lazy windows of the adaptive scheduler)."""

    def __init__(self, check_H, colsig):
        import scipy.sparse as sp
        C = sp.csc_matrix(np.ascontiguousarray(check_H, dtype=np.uint8))
        C.sort_indices()
        self.m, self.n = C.shape
        self.indptr = C.indptr.astype(np.int64)
        self.indices = C.indices.astype(np.int32)
        cnt = np.diff(self.indptr)
        nz = cnt > 0
        self.lo = np.full(self.n, self.m, dtype=np.int64)
        self.hi = np.full(self.n, -1, dtype=np.int64)
        self.lo[nz] = self.indices[self.indptr[:-1][nz]]
        self.hi[nz] = self.indices[self.indptr[1:][nz] - 1]
        self.colsig = np.ascontiguousarray(colsig)

    def window_cols(self, a, b):
        return np.nonzero((self.lo >= a) & (self.hi < b))[0].astype(np.int64)

    def window_key(self, a, b, cols):
        """Same key for windows whose sub-instances coincide in local coordinates."""
        cnt = (self.indptr[cols + 1] - self.indptr[cols]).astype(np.int64)
        idx = np.concatenate([self.indices[self.indptr[c]:self.indptr[c + 1]] for c in cols]) \
            if cols.size else np.zeros(0, np.int32)
        h = hashlib.md5()
        h.update(np.array([b - a, cols.size], dtype=np.int64).tobytes())
        h.update(cnt.tobytes())
        h.update((idx.astype(np.int64) - a).tobytes())
        h.update(np.ascontiguousarray(self.colsig[cols]).tobytes())
        return h.hexdigest()

    def build(self, a, b, cols, level, m4ri_k):
        """Window entry (as in _detector_windows) or None (no possible logical, full rank)."""
        if cols.size < 2:
            return None
        cs = np.ascontiguousarray(self.colsig[cols])
        if not cs.any():
            return None
        nw_s = (cols.size + 63) // 64
        rows = np.zeros(b - a, dtype=np.int64)
        for j, c in enumerate(cols):
            rows[self.indices[self.indptr[c]:self.indptr[c + 1]] - a] = 1
        rmap = np.cumsum(rows) - 1
        Hp_s = np.zeros((int(rows.sum()), nw_s), dtype=np.uint64)
        for j, c in enumerate(cols):
            r = rmap[self.indices[self.indptr[c]:self.indptr[c + 1]] - a]
            Hp_s[r, j >> 6] |= np.uint64(1) << np.uint64(j & 63)
        tab = np.zeros((1 << m4ri_k, nw_s), dtype=np.uint64)
        rk = int(_rref_m4ri_order(Hp_s.copy(), np.arange(cols.size), m4ri_k, tab,
                                  np.empty(Hp_s.shape[0], dtype=np.int64)))
        if rk >= cols.size or rk == 0:
            return None
        return (cols, Hp_s, cs, int(cols.size), nw_s, rk, level, a, b - a, "")


PERIOD_WINDOW_QUARTERS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128)


def _period_window_specs(sc, T, max_frac=0.5):
    """Lazy period windows on a _SparseChecks: sizes k T / 4 (k in PERIOD_WINDOW_QUARTERS,
    from half a period up to max_frac m), each anchored at detector 0, anchored at the last
    detector, and in between at steps of T / 4 (sizes up to 2 T) or T / 2 (larger) from the
    phase p (the cut in [m/4, m/4 + T) crossed by the fewest columns). Light solutions can
    span less than a round and straddle a round boundary at any phase (a 305-detector window
    holds the lightest fault set of a 390-detector-per-round Mitten circuit), and small
    windows are cheap. Returns the distinct windows as dicts (a, b, level = k, cols, key,
    count) without building their sub-instances."""
    m = sc.m
    nz = sc.hi >= 0
    cross = np.zeros(m + 2, dtype=np.int64)
    np.add.at(cross, sc.lo[nz] + 1, 1)
    np.add.at(cross, sc.hi[nz] + 1, -1)
    cross = np.cumsum(cross)[:m + 1]
    q0 = m // 4
    p = (q0 + int(np.argmin(cross[q0:q0 + T]))) % T
    out, where = [], {}
    for k in PERIOD_WINDOW_QUARTERS:
        s = (k * T) // 4
        if s < 2 or s > max_frac * m:
            continue
        # phases at fixed offsets inside each period (so that every phase is exactly
        # periodic and its bulk windows merge into one class)
        offs = sorted(set((q * T) // 4 for q in range(4))) if k <= 8 else [0, T // 2]
        starts = [0, m - s] + sorted(a for j in range(-1, m // T + 1) for o in offs
                                     for a in [p + j * T + o] if 0 < a < m - s)
        for a in starts:
            cols = sc.window_cols(a, a + s)
            if cols.size < 2 or not sc.colsig[cols].any():
                continue
            key = sc.window_key(a, a + s, cols)
            if key in where:
                out[where[key]]["count"] += 1
                continue
            where[key] = len(out)
            out.append(dict(a=a, b=a + s, level=k, cols=cols, key=key, count=1))
    return out
