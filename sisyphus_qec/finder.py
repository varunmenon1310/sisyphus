"""Sisyphus: minimum distance of CSS codes and circuit-level detector error models.

Core search (kernel + public API). Given a binary check matrix H and a
description of the "trivial" vectors (stabilizers), it searches for the
lightest vector c with H c = 0 that is not trivial, and returns the lightest
ones found. Every returned vector is verified, so the reported weight is a
valid upper bound on the true distance.

Method (details in README.md): a randomized information-set decoding (ISD)
search. Each search chain keeps a generator matrix in systematic form and moves
between neighbouring information sets by single-column pivot swaps
(Canteaut--Chabaud); at every step it extracts candidates as single rows
(Prange), sums of two or three rows (Lee--Brickell) found by a collision search
on random column subsets (Stern), and reduces them greedily. Column weights
learned from the low-weight codewords seen so far bias the swaps and the
restarts toward promising columns (a stochastic hill climb with restarts and
partial forgetting, as in iterated local search). Independent groups of chains
run in parallel processes (island model). For circuit-level detector error
models, part of the search runs on time windows of the detectors.

Public API
----------
    css_distance(Hx, Hz, ...)                  # both sides of a CSS code
    side_distance(check_H, dual_H, ...)        # one side
    side_distance(H_dem, None, pair_rows=L)    # circuit distance from a DEM
    metacheck_distance(M, H, ...)              # syndrome-space distance
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit

# ════════════════════════════════════════════════════════════════════════════
# 1. Search parameters
# ════════════════════════════════════════════════════════════════════════════
# Defaults were fixed by an equal-wall-clock comparison on the hard P=384
# instances. Every one of them can be overridden
# per call through **params.

SWAPS_PER_SHOT = 8        # pivot swaps between two read-offs (applied as one batch)
RESTART_INTERVAL = 150    # base restart cadence in shots; scaled with the basis size
LB_M = 256                # light list of the read-off. 48 until 2026-11-03 (exhaustive
                          # pairs/triples over it); 256 with the hashed read-off below.
                          # With lb_hash_l = 0 (exhaustive read-off) pass lb_M = 48.
LB_P = 2                  # 0: rows only, 1: + pairs, 2: + triples
LB_HASH_L = 12            # Stern-style hashed read-off (default since 2026-11-03 15:05;
                          # 0 = the exhaustive Lee-Brickell read-off). L > 0: per window,
                          # L random free columns are drawn and every row is keyed by
                          # its bits there; pairs are read from shared buckets over
                          # ALL rows (their sum vanishes on the window), triples from
                          # an anchor among the LB_R lightest rows, a second row in the
                          # light list of lb_M rows and a third row from the bucket
                          # that cancels the window (any row). Exhaustive pairs among
                          # the LB_R lightest rows are kept. A light sum is missed by a
                          # window with probability about (1 - w_free/n_free)^L.
                          # Measured 2026-11-03 (L=12, M=256): per shot 1.9-2.2x the cost,
                          # P=192 reaches d in 96/96 islands in 350 rounds instead of 2300
                          # (2.4x more hits per core-hour); P=384 Z weight-20 hits 13/400
                          # islands vs 11/1600 (2.4x per core-hour).
LB_HASH_WINDOWS = 1       # windows per shot in hashed mode (each costs one pass)
LB_R = 48                 # triple anchors (hashed mode)
LB_HASH_P4 = 0            # hashed mode, Stern meet-in-the-middle for 4-row sums:
                          # pairs of light-list rows are keyed by their XOR on a
                          # wider window of LB_HASH_L + 4 columns; an anchor pair
                          # (anchor among the LB_R lightest, second row in the light
                          # list) is matched with the pairs whose key cancels its own.
                          # 0 = off, 1 = on (2026-11-03).
GREEDY_PASSES = 4         # coset-descent sweeps per read-off word (0 = off)
GREEDY_KICKS = 0          # iterated-local-search kicks after descent (12 in the
                          # evolved finder; they triple per-shot cost and lose at
                          # equal wall-clock, so off by default)
GREEDY_KICK_SIZE = 2      # rows added per kick (cycled 2/3/4 on code-sized bases)
M4RI_K = 8                # pivots eliminated per pass in the restart RREF
PROJ_WORDS = 1            # 64-bit words of certification parities per row
MIGRATE_INTERVAL = 100    # shots between island migrations
MIGRATE_BOOST = 0.0       # theta increment on the migrated support. 20.0 until
                          # 2026-11-03: migration made the learners of an island
                          # converge on one word (see KICK_ROUNDS); 0 keeps them
                          # independent, which tripled the rate of reaching the
                          # common light words on P=384 and did not slow P=192.
MIX_RATE = 0.0            # experimental: expected random row additions per shot
                          # (row i ^= row j). The basis stays a basis of the kernel
                          # but leaves reduced form, as the swap bug fixed on
                          # 2026-11-02 did; off by default.
STALL_ROUNDS = 0          # experimental: an island whose best weight has not
                          # improved for this many rounds forgets its learned
                          # priorities and restarts all walkers (a fresh island
                          # inside the same process; found codewords are kept).
                          # 0 = off.
KICK_ROUNDS = 1000        # per-walker iterated-local-search kick: a learner whose own
                          # best weight has not improved for this many rounds forgets
                          # a fraction KICK_FRAC of the priorities on the support of
                          # its best word, forgets its best weight and restarts. The
                          # learned priorities otherwise collapse onto the support of
                          # the current best word (participation ratio == best weight,
                          # measured 2026-11-03), which turns the walker into one slow
                          # local descent stuck at the nearest local minimum. 0 = off.
                          # Default since 2026-11-03 (was 0).
KICK_FRAC = 0.3           # fraction of the best support forgotten per kick (1 = all;
                          # 0.3 keeps the walker near its region, best on P=192/P=384)
KICK_PATIENCE = 2.0       # adaptive kick trigger: also require the wait to exceed this
                          # multiple of the walker's longest gap between improvements
                          # so far (scale-free; 0 = fixed period KICK_ROUNDS only)
MAX_CODEWORDS = 2000      # hits of the current best weight stored per island
                          # (the returned set is the distinct words among them)
ISLAND_SHOTS = 250_000    # default per-island shot budget. An island that spends it
                          # without reaching the cutoff is replaced by a freshly
                          # seeded island (fresh theta and random streams). 250k
                          # is about where islands reach d=20 on the P=384 code.
BIG_BASE = 400_000        # rows*words above which the "huge base" rules apply
                          # (circuit detector error models): descent is capped
                          # at 2 sweeps and feeds theta learning directly
RESTART_FROM_H = 1        # 1: a restart rebuilds the walker's basis from the check
                          # matrix H instead of re-reducing the stored generator basis
                          # (dist_m4ri's random-window step): H (r x n) is eliminated
                          # in the column order, then every free column gives one
                          # systematic generator row (a 1 there plus the pivot columns
                          # of the H rows that contain it); its certification words
                          # are the XOR of per-column signature words. The information
                          # set has the same distribution (the learners' order is
                          # reversed, so learned supports still leave it). On a
                          # rate-1/2 code the elimination is r/m times cheaper than
                          # the generator restart (P=384: 11 ms vs 56 ms; circuit DEMs,
                          # whose kernel basis has 10-40x more rows than H: 16-35x,
                          # e.g. bb360 1.6 s vs 55 s per walker). Default 1 since
                          # 2026-11-04 (identical basis; the circuit-level benchmark
                          # against dist_m4ri is 2-10x faster with it, P=384 unchanged).
                          # 0 = re-reduce the stored generator basis (the old path).
COMPACT_VERSION = "2026-11-04c"   # compact module revision (c: entering tie-break = lowest column)
COMPACT = 1               # 1: walkers store each basis row on the r redundant columns
                          # only (distance_finder_compact.py, 2026-11-04): same search,
                          # n/r times less memory and per-shot work (4x on rate-1/2
                          # codes, 10x on circuit DEMs), no kernel basis computed.
                          # 0: the full-width walker of this file.
WINDOWS = "auto"          # default since 2026-11-05 (A/B on 19 circuit DEMs, 3 seeds:
                          # never slower, 5-30x faster on double gross / bb360; codes are
                          # untouched since 'auto' needs n >= 6 m). Before: 0.
                          # detector-window sub-instances (2026-11-04 night; 0 = off):
                          # levels of dyadic windows of the check rows (detectors, in
                          # their given order, = time order in a Stim DEM). Window
                          # [a, a+s), s = ceil(m / 2^j), offsets multiples of s/2,
                          # j = 1..WINDOWS: its sub-instance is the columns whose every
                          # check lies in the window (their codewords, zero-extended,
                          # are codewords of the full instance, certified by the same
                          # signature words). A fraction WINDOW_FRAC of the islands
                          # runs on a random window for WINDOW_ROUNDS rounds. Why:
                          # dist_m4ri's hits on circuit DEMs all come from its index-
                          # window steps, the light words of multi-round DEMs span
                          # 0.5-3% of the columns, and the one-round APM DEM contains
                          # the code as the columns of one detector layer (code alone:
                          # 22 in 53 s; inside the DEM: 140-183 after 300 s).
WINDOW_FRAC = 0.5         # fraction of the islands that run on a window (WINDOWS > 0)
WINDOW_ROUNDS = 5000      # rounds of a window island (windows rotate)
WINDOW_MIN = 16           # smallest window, in checks
WINDOW_AUTO_RATIO = 6     # windows='auto': on when n_cols >= 6 * rows (circuit DEMs have
                          # n/m = 8-13, codes 2-4: their windows are empty or tiny)
WINDOW_AUTO_LEVELS = 5    # levels used by windows='auto'
EXPLORER_RESTART = 0      # restart cadence (rounds) of the uniform explorer walker;
                          # 0 = the island's restart interval. With RESTART_FROM_H = 1
                          # and 1-4 the explorer draws a fresh uniform information set
                          # every few shots, like dist_m4ri's RW steps, read off with
                          # the full pairs + triples + descent machinery.

# Walker presets (eta, decay, margin, tournament size, exploration eps). One
# island runs all four in lockstep: three learners of different aggressiveness
# plus one uniform-random explorer. The explorer never gets below weight 262
# on P=384 (the learners reach 106), but it is what gives enumeration breadth
# on small codes (gross, cutoff=0: 504 distinct weight-12 words in 20k shots
# with it, 191 without), and with migration off and the kick on, the three
# learners alone match four learners on P=192 and P=384 (2026-11-03).
_PORTFOLIO = [
    (1.0, 0.010, 3, 8, 0.10),
    (1.5, 0.006, 2, 8, 0.08),
    (1.0, 0.010, 3, 8, 0.25),
    (1.0, 0.000, 0, 1, 1.00),
]
ISLAND_W = len(_PORTFOLIO)
BITORDER = "little"


def _params(**overrides):
    p = dict(swaps_per_shot=SWAPS_PER_SHOT, restart_interval=None, lb_M=LB_M,
             lb_p=LB_P, lb_hash_l=LB_HASH_L, lb_hash_windows=LB_HASH_WINDOWS, lb_R=LB_R,
             lb_hash_p4=LB_HASH_P4,
             greedy_passes=GREEDY_PASSES, greedy_kicks=GREEDY_KICKS,
             kick_size=GREEDY_KICK_SIZE, m4ri_k=M4RI_K, proj_words=PROJ_WORDS,
             migrate_interval=MIGRATE_INTERVAL, migrate_boost=MIGRATE_BOOST,
             max_codewords=MAX_CODEWORDS, mix_rate=MIX_RATE, stall_rounds=STALL_ROUNDS,
             kick_rounds=KICK_ROUNDS, kick_frac=KICK_FRAC, kick_patience=KICK_PATIENCE,
             restart_from_h=RESTART_FROM_H, explorer_restart=EXPLORER_RESTART,
             windows=WINDOWS, window_frac=WINDOW_FRAC, window_rounds=WINDOW_ROUNDS,
             window_min=WINDOW_MIN)
    unknown = set(overrides) - set(p)
    if unknown:
        raise TypeError(f"unknown finder parameters: {sorted(unknown)}")
    p.update(overrides)
    return p


# ════════════════════════════════════════════════════════════════════════════
# 2. Result types
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SideResult:
    """Outcome of one search (one side of a CSS code, or a metacheck/circuit
    instance). `distance` is the lightest certified logical found (an upper
    bound on the true distance), or -1 if the side was skipped. `codewords`
    holds the distinct logicals of that weight found during the run, one per
    row, and `counts[i]` is how often row i was hit (each island stores at
    most `max_codewords` hits, so the set is complete only up to that cap)."""
    distance: int
    codewords: np.ndarray
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    shots: int = 0
    wall_time: float = 0.0
    islands: int = 1          # islands that ran (each with fresh seeds and learning state)
    t_best: float = -1.0      # estimated wall time at which `distance` was first reached
                              # (compact walker; -1 = unknown)

    @property
    def seeds(self) -> int:
        """Older name of `islands`."""
        return self.islands

    @property
    def num_codewords(self) -> int:
        return int(self.codewords.shape[0])

    @property
    def witness(self):
        """One certified codeword of weight `distance`, or None."""
        return self.codewords[0].astype(np.uint8) if self.num_codewords else None


@dataclass
class CssDistanceResult:
    """`x`: X-type logicals (ker Hz minus rowspan Hx); `z`: Z-type logicals."""
    x: SideResult
    z: SideResult

    @property
    def distance(self) -> int:
        ds = [s.distance for s in (self.x, self.z) if s.distance >= 0]
        return min(ds) if ds else -1


# Names used by older call sites.
DistanceEstimate = SideResult
CssDistanceEstimate = CssDistanceResult


# ════════════════════════════════════════════════════════════════════════════
# 3. Bit packing
# ════════════════════════════════════════════════════════════════════════════
# Binary matrices are stored as rows of 64-bit words, little-endian within a
# word, so one XOR processes 64 matrix entries.

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


# ════════════════════════════════════════════════════════════════════════════
# 4. Linear algebra over GF(2)
# ════════════════════════════════════════════════════════════════════════════
# Row layout used throughout: each row of a walker's basis has `nw` data
# words (the codeword bits) followed by `ns` certification words (its
# parities against the random pairing vectors). Row operations act on the
# whole row, so the certification words are maintained for free.

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
def _row_stats(A, nw, row_wt, is_log):
    """row_wt[i] = weight of the data part of row i; is_log[i] = certification
    words non-zero (row i is a logical)."""
    m, nwt = A.shape
    for i in range(m):
        wt = 0
        for w in range(nw):
            wt += int(_popcount_u64(A[i, w]))
        row_wt[i] = wt
        nz = False
        for w in range(nw, nwt):
            if A[i, w] != 0:
                nz = True
                break
        is_log[i] = nz


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


@njit(cache=True)
def _restart_from_h(Hp, Hs, perm, k, tab, pivH, colsig, A, pivot_col, idx, nw):
    """Fresh reduced basis of ker(H) with pivot columns chosen by `perm`, built
    from the check matrix (see RESTART_FROM_H): H is copied to the scratch Hs
    and reduced in the column order perm (its pivots are the redundant set);
    row j of A is the systematic codeword of the j-th free column f_j: a 1 at
    f_j plus, for every H row that has a 1 at f_j, that row's pivot column.
    Certification words: colsig[f_j] XOR colsig over those pivot columns.
    Fills A (m rows, nw data words + ns certification words) and pivot_col;
    returns m, or -1 if n - rank(H) != m (then the caller falls back)."""
    r = Hp.shape[0]
    m, nwt = A.shape
    ns = nwt - nw
    n_cols = idx.shape[0]
    for i in range(r):
        for w in range(nw):
            Hs[i, w] = Hp[i, w]
    rank = _rref_m4ri_order(Hs, perm, k, tab, pivH)
    if n_cols - rank != m:
        return -1
    for c in range(n_cols):
        idx[c] = -1
    for rho in range(rank):
        idx[pivH[rho]] = -2
    j = 0
    for c in range(n_cols):
        if idx[c] == -2:
            continue
        idx[c] = j
        pivot_col[j] = c
        for w in range(nwt):
            A[j, w] = np.uint64(0)
        A[j, c >> 6] = np.uint64(1) << np.uint64(c & 63)
        for s in range(ns):
            A[j, nw + s] = colsig[c, s]
        j += 1
    for rho in range(rank):
        q = pivH[rho]
        qw = q >> 6
        qb = np.uint64(1) << np.uint64(q & 63)
        for w in range(nw):
            word = Hs[rho, w]
            base = w << 6
            while word != 0:
                lowbit = word & (np.uint64(0) - word)
                f = base + int(_popcount_u64(lowbit - np.uint64(1)))
                word ^= lowbit
                if f == q:
                    continue
                jj = idx[f]
                A[jj, qw] |= qb
                for s in range(ns):
                    A[jj, nw + s] ^= colsig[q, s]
    return m


# ════════════════════════════════════════════════════════════════════════════
# 5. Walker kernels
# ════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _swap_batch(A, pivot_col, row_wt, is_log, theta, nw, s, tourn, eps,
                mu_list, u_rows, ws_rows, in_ws, tab, uc):
    """Move the information set by s single-column pivot swaps, applied as one
    pass over the basis.

    Swap selection (per swap): the pivot row to evict is the winner of a
    `tourn`-way tournament on theta of its pivot column (highest theta loses
    its pivot role); the entering column is the set bit of that row with the
    lowest theta. With probability eps both choices are uniform instead.

    Batching: the sequence of s swaps is a linear map whose effect on a row
    depends only on the row's bits at the s entering columns, so a 2^s table
    of corrections is built from the evicted rows and every other row gets one
    table XOR. Rows in the working set (the evicted ones) are updated eagerly.
    Row weights, logical flags and pivot columns are maintained.

    The table is indexed by the DISTINCT entering columns (listed in `uc`). A
    column can enter, leave and enter again within one batch; giving it two
    index bits would make its two corrections cancel and leave rows with a 1
    at another row's pivot (fixed 2026-11-02)."""
    m, nwt = A.shape
    n_ops = 0
    n_ws = 0
    for _j in range(s):
        r = np.random.randint(0, m)
        if np.random.random() >= eps:
            best_th = theta[pivot_col[r]]
            for _t in range(tourn - 1):
                rc = np.random.randint(0, m)
                th = theta[pivot_col[rc]]
                if th > best_th:
                    best_th = th
                    r = rc
        if in_ws[r] == 0:                     # bring row r up to date
            for q in range(n_ops):
                if _bit(A, r, mu_list[q]):
                    for w in range(nwt):
                        A[r, w] ^= u_rows[q, w]
            in_ws[r] = 1
            ws_rows[n_ws] = r
            n_ws += 1
        lam = pivot_col[r]
        cnt = 0
        for w in range(nw):
            cnt += int(_popcount_u64(A[r, w]))
        if cnt <= 1:
            continue                          # nothing to swap in
        mu = -1
        if np.random.random() < eps:          # uniform entering column
            pick = np.random.randint(0, cnt - 1)
            seen = 0
            for w in range(nw):
                word = A[r, w]
                base = w << 6
                while word != 0:
                    lowbit = word & (np.uint64(0) - word)
                    col = base + int(_popcount_u64(lowbit - np.uint64(1)))
                    word ^= lowbit
                    if col == lam:
                        continue
                    if seen == pick:
                        mu = col
                        break
                    seen += 1
                if mu >= 0:
                    break
        else:                                 # lowest-theta entering column
            best_mu_th = 1e18
            for w in range(nw):
                word = A[r, w]
                base = w << 6
                while word != 0:
                    lowbit = word & (np.uint64(0) - word)
                    col = base + int(_popcount_u64(lowbit - np.uint64(1)))
                    word ^= lowbit
                    if col == lam:
                        continue
                    if theta[col] < best_mu_th:
                        best_mu_th = theta[col]
                        mu = col
        if mu < 0:
            continue
        mu_list[n_ops] = mu
        for w in range(nwt):
            u_rows[n_ops, w] = A[r, w]
        wi = mu >> 6
        bm = np.uint64(1) << np.uint64(mu & 63)
        for t in range(n_ws):                 # eager update of the working set
            i = ws_rows[t]
            if i != r and (A[i, wi] & bm):
                for w in range(nwt):
                    A[i, w] ^= u_rows[n_ops, w]
        pivot_col[r] = mu
        n_ops += 1
    if n_ops > 0:
        # distinct entering columns, one table bit each
        n_u = 0
        for j in range(n_ops):
            c = mu_list[j]
            dup = False
            for k in range(n_u):
                if uc[k] == c:
                    dup = True
                    break
            if not dup:
                uc[n_u] = c
                n_u += 1
        # correction for a unit vector at each distinct entering column, then all sums
        for w in range(nwt):
            tab[0, w] = 0
        for j in range(n_u):
            idx = 1 << j
            for w in range(nwt):
                tab[idx, w] = 0
            mu = uc[j]
            ebit = np.uint64(1) << np.uint64(mu & 63)
            tab[idx, mu >> 6] |= ebit
            for q in range(n_ops):
                muq = mu_list[q]
                if (tab[idx, muq >> 6] >> np.uint64(muq & 63)) & np.uint64(1):
                    for w in range(nwt):
                        tab[idx, w] ^= u_rows[q, w]
            tab[idx, mu >> 6] ^= ebit
            for p in range(idx + 1, 2 * idx):
                for w in range(nwt):
                    tab[p, w] = tab[p - idx, w] ^ tab[idx, w]
        for i in range(m):                    # the single pass
            if in_ws[i]:
                continue
            p = 0
            for j in range(n_u):
                p |= _bit(A, i, uc[j]) << j
            if p != 0:
                wt = 0
                for w in range(nw):
                    a = A[i, w] ^ tab[p, w]
                    A[i, w] = a
                    wt += int(_popcount_u64(a))
                row_wt[i] = wt
                nz = False
                for w in range(nw, nwt):
                    a = A[i, w] ^ tab[p, w]
                    A[i, w] = a
                    if a != 0:
                        nz = True
                is_log[i] = nz
    for t in range(n_ws):                     # refresh the working set
        i = ws_rows[t]
        wt = 0
        for w in range(nw):
            wt += int(_popcount_u64(A[i, w]))
        row_wt[i] = wt
        nz = False
        for w in range(nw, nwt):
            if A[i, w] != 0:
                nz = True
                break
        is_log[i] = nz
        in_ws[i] = 0


@njit(cache=True)
def _readoff(A, row_wt, is_log, nw, M, lb_p, light_idx, light_wt, shot_cw,
             pair_tmp):
    """Read candidate codewords off the current basis.

    Level 0 (Prange): every logical row is a codeword; the lightest wins. The
    M lightest logical rows are kept in `light_idx` (sorted by weight).
    Level 1 (Lee--Brickell pairs) and level 2 (triples) add sums of two or
    three rows of that list. A sum is a logical iff the XOR of the rows'
    certification words is non-zero. Triples are pruned by the triangle
    inequality wt(a+b+c) >= wt(a+b) - wt(c).

    Writes the lightest word (data and certification words) to `shot_cw`;
    returns (its weight, size of the light list)."""
    m, nwt = A.shape
    n_light = 0
    min_w = 1 << 40
    min_i = -1
    for i in range(m):
        if not is_log[i]:
            continue
        wt = row_wt[i]
        if wt < min_w:
            min_w = wt
            min_i = i
        if n_light < M:                       # insertion into the sorted list
            pos = n_light
            while pos > 0 and light_wt[pos - 1] > wt:
                light_idx[pos] = light_idx[pos - 1]
                light_wt[pos] = light_wt[pos - 1]
                pos -= 1
            light_idx[pos] = i
            light_wt[pos] = wt
            n_light += 1
        elif wt < light_wt[M - 1]:
            pos = M - 1
            while pos > 0 and light_wt[pos - 1] > wt:
                light_idx[pos] = light_idx[pos - 1]
                light_wt[pos] = light_wt[pos - 1]
                pos -= 1
            light_idx[pos] = i
            light_wt[pos] = wt
    if min_i < 0:
        return min_w, 0
    for w in range(nwt):
        shot_cw[w] = A[min_i, w]
    if lb_p == 0 or n_light < 2:
        return min_w, n_light
    for a in range(n_light):
        ia = light_idx[a]
        for b in range(a + 1, n_light):
            ib = light_idx[b]
            nz = False
            for w in range(nw, nwt):
                if (A[ia, w] ^ A[ib, w]) != 0:
                    nz = True
                    break
            if lb_p >= 2:
                wt = 0
                for w in range(nw):
                    x = A[ia, w] ^ A[ib, w]
                    pair_tmp[w] = x
                    wt += int(_popcount_u64(x))
                if nz and wt < min_w:
                    min_w = wt
                    for w in range(nw):
                        shot_cw[w] = pair_tmp[w]
                    for w in range(nw, nwt):
                        shot_cw[w] = A[ia, w] ^ A[ib, w]
                for c in range(b + 1, n_light):
                    ic = light_idx[c]
                    if wt - light_wt[c] >= min_w:
                        continue
                    nz3 = False
                    for w in range(nw, nwt):
                        if (A[ia, w] ^ A[ib, w] ^ A[ic, w]) != 0:
                            nz3 = True
                            break
                    if not nz3:
                        continue
                    w3 = _xor_weight_row(pair_tmp, A, ic, nw)
                    if w3 < min_w:
                        min_w = w3
                        for w in range(nw):
                            shot_cw[w] = pair_tmp[w] ^ A[ic, w]
                        for w in range(nw, nwt):
                            shot_cw[w] = A[ia, w] ^ A[ib, w] ^ A[ic, w]
            else:
                if not nz:
                    continue
                wt = _xor_weight_row(A[ia], A, ib, nw)
                if wt < min_w:
                    min_w = wt
                    for w in range(nwt):
                        shot_cw[w] = A[ia, w] ^ A[ib, w]
    return min_w, n_light


@njit(cache=True)
def _readoff_hashed(A, row_wt, is_log, pivot_col, nw, M, R, L, n_windows,
                    light_idx, light_wt, shot_cw, pair_tmp,
                    keys, bstart, border, win, is_piv, col_cnt):
    """Stern-style read-off (see LB_HASH_L).

    Light list: the M lightest logical rows, selected through a weight
    histogram (the R lightest of them sorted first, they are the triple
    anchors). Exhaustive pairs and triples among the R lightest, exactly as
    `_readoff` with M = R (so this read-off never finds less). Then per window: L free columns on which the basis is
    balanced (density between 0.3 and 0.7 on a sample of rows; a learned
    basis is sparse, and on sparse columns every light row keys to 0 and the
    zero bucket explodes), every row keyed by its bits there (counting sort
    into 2^L buckets), pairs within buckets over all rows, and triples
    (anchor among the R lightest, second row in the light list, third row
    from the bucket whose key cancels the pair's key). Buckets larger than
    four times the expected size are skipped as uninformative. Writes the
    lightest word to `shot_cw`; returns (its weight, size of the light list)."""
    m, nwt = A.shape
    n_cols = is_piv.shape[0]
    # ---- weight histogram over logical rows -> threshold for the M lightest
    for c in range(n_cols + 1):
        col_cnt[c] = 0
    n_logical = 0
    min_w = 1 << 40
    min_i = -1
    for i in range(m):
        if not is_log[i]:
            continue
        wt = row_wt[i]
        col_cnt[wt] += 1
        n_logical += 1
        if wt < min_w:
            min_w = wt
            min_i = i
    if min_i < 0:
        return min_w, 0
    for w in range(nwt):
        shot_cw[w] = A[min_i, w]
    if n_logical < 2:
        return min_w, 1
    acc = 0
    T = n_cols
    for wt in range(n_cols + 1):
        acc += col_cnt[wt]
        if acc >= M:
            T = wt
            break
    n_light = 0
    for i in range(m):                        # rows lighter than T, then ties up to M
        if n_light >= M:
            break
        if is_log[i] and row_wt[i] <= T:
            light_idx[n_light] = i
            light_wt[n_light] = row_wt[i]
            n_light += 1
    nR = R if R < n_light else n_light
    for a in range(nR):                       # the R lightest to the front, sorted
        best = a
        for b in range(a + 1, n_light):
            if light_wt[b] < light_wt[best]:
                best = b
        if best != a:
            ti = light_idx[a]; tw = light_wt[a]
            light_idx[a] = light_idx[best]; light_wt[a] = light_wt[best]
            light_idx[best] = ti; light_wt[best] = tw
    for a in range(nR):                       # exhaustive pairs and triples among the
        ia = light_idx[a]                     # R lightest (the hashed read-off is then
        for b in range(a + 1, nR):            # a superset of `_readoff` with M = R)
            ib = light_idx[b]
            nz = False
            for w in range(nw, nwt):
                if (A[ia, w] ^ A[ib, w]) != 0:
                    nz = True
                    break
            wt = 0
            for w in range(nw):
                x = A[ia, w] ^ A[ib, w]
                pair_tmp[w] = x
                wt += int(_popcount_u64(x))
            if nz and wt < min_w:
                min_w = wt
                for w in range(nwt):
                    shot_cw[w] = A[ia, w] ^ A[ib, w]
            for c in range(b + 1, nR):
                ic = light_idx[c]
                if wt - light_wt[c] >= min_w:
                    continue
                nz3 = False
                for w in range(nw, nwt):
                    if (A[ia, w] ^ A[ib, w] ^ A[ic, w]) != 0:
                        nz3 = True
                        break
                if not nz3:
                    continue
                w3 = _xor_weight_row(pair_tmp, A, ic, nw)
                if w3 < min_w:
                    min_w = w3
                    for w in range(nw):
                        shot_cw[w] = pair_tmp[w] ^ A[ic, w]
                    for w in range(nw, nwt):
                        shot_cw[w] = A[ia, w] ^ A[ib, w] ^ A[ic, w]
    # ---- column densities on a row sample (balanced window columns)
    for c in range(n_cols):
        is_piv[c] = 0
        col_cnt[c] = 0
    for i in range(m):
        is_piv[pivot_col[i]] = 1
    ns = 256 if m > 256 else m
    stride = m // ns
    for q in range(ns):
        i = q * stride
        for w in range(nw):
            word = A[i, w]
            base = w << 6
            while word != 0:
                lowbit = word & (np.uint64(0) - word)
                col_cnt[base + int(_popcount_u64(lowbit - np.uint64(1)))] += 1
                word ^= lowbit
    lo = (3 * ns) // 10
    hi = (7 * ns) // 10
    n_free = n_cols - m                       # every row has its own pivot column
    if L > n_free:
        L = n_free
    if L < 1:
        return min_w, n_light
    nb = 1 << L
    cap = 4 * (m // nb + 1)
    for _win in range(n_windows):
        k = 0
        tries = 0
        while k < L:                          # L distinct balanced free columns
            tries += 1
            c = np.random.randint(0, n_cols)
            if is_piv[c] != 0:
                continue
            if tries < 64 * L and (col_cnt[c] < lo or col_cnt[c] > hi):
                continue
            dup = False
            for j in range(k):
                if win[j] == c:
                    dup = True
                    break
            if dup:
                continue
            win[k] = c
            k += 1
        for i in range(m):                    # keys of all rows on the window
            key = 0
            for j in range(L):
                c = win[j]
                if (A[i, c >> 6] >> np.uint64(c & 63)) & np.uint64(1):
                    key |= 1 << j
            keys[i] = key
        for b in range(nb + 1):               # counting sort into buckets
            bstart[b] = 0
        for i in range(m):
            bstart[keys[i] + 1] += 1
        for b in range(nb):
            bstart[b + 1] += bstart[b]
        for i in range(m):
            kb = keys[i]
            border[bstart[kb]] = i
            bstart[kb] += 1
        # bucket b now occupies border[bstart[b-1] : bstart[b]] (bstart[-1] = 0)
        for b in range(nb):                   # pairs within buckets, all rows
            s = 0 if b == 0 else bstart[b - 1]
            e = bstart[b]
            if e - s > cap:
                continue
            for p in range(s, e):
                i = border[p]
                for q in range(p + 1, e):
                    j = border[q]
                    d = row_wt[i] - row_wt[j]
                    if d < 0:
                        d = -d
                    if d >= min_w:
                        continue
                    nz = False
                    for w in range(nw, nwt):
                        if (A[i, w] ^ A[j, w]) != 0:
                            nz = True
                            break
                    if not nz:
                        continue
                    wt = _xor_weight_row(A[i], A, j, nw)
                    if wt < min_w:
                        min_w = wt
                        for w in range(nwt):
                            shot_cw[w] = A[i, w] ^ A[j, w]
        for a in range(nR):                   # triples through the cancelling bucket
            ia = light_idx[a]
            ka = keys[ia]
            for bb in range(n_light):
                if bb == a:
                    continue
                ib = light_idx[bb]
                kk = ka ^ keys[ib]
                s = 0 if kk == 0 else bstart[kk - 1]
                e = bstart[kk]
                if e <= s or e - s > cap:
                    continue
                wab = 0
                for w in range(nw):
                    x = A[ia, w] ^ A[ib, w]
                    pair_tmp[w] = x
                    wab += int(_popcount_u64(x))
                for p in range(s, e):
                    ic = border[p]
                    if ic == ia or ic == ib:
                        continue
                    if wab - row_wt[ic] >= min_w:
                        continue
                    nz3 = False
                    for w in range(nw, nwt):
                        if (A[ia, w] ^ A[ib, w] ^ A[ic, w]) != 0:
                            nz3 = True
                            break
                    if not nz3:
                        continue
                    w3 = _xor_weight_row(pair_tmp, A, ic, nw)
                    if w3 < min_w:
                        min_w = w3
                        for w in range(nw):
                            shot_cw[w] = pair_tmp[w] ^ A[ic, w]
                        for w in range(nw, nwt):
                            shot_cw[w] = A[ia, w] ^ A[ib, w] ^ A[ic, w]
    return min_w, n_light


@njit(cache=True)
def _readoff_p4(A, row_wt, nw, n_light, R, L4, light_idx, min_w, shot_cw, pair_tmp,
                is_piv, col_cnt, keys, win, pkey, pa, pb, bstart, order):
    """Stern meet-in-the-middle for sums of four light-list rows (LB_HASH_P4),
    run after `_readoff_hashed` (which leaves `is_piv` and the column counts
    in `col_cnt`). Window of L4 balanced free columns; every light row keyed
    on it; all pairs (c, d) of the light list keyed by key[c] ^ key[d] and
    counting-sorted; for each anchor pair (a among the R lightest, b in the
    list) the pairs in the bucket key[a] ^ key[b] give 4-row sums that vanish
    on the window. Returns the (possibly improved) minimum weight; `shot_cw`
    is updated on improvement."""
    m, nwt = A.shape
    n_cols = is_piv.shape[0]
    ns = 256 if m > 256 else m
    lo = (3 * ns) // 10
    hi = (7 * ns) // 10
    n_free = n_cols - m
    if L4 > n_free:
        L4 = n_free
    if L4 < 1:
        return min_w
    k = 0
    tries = 0
    while k < L4:
        tries += 1
        c = np.random.randint(0, n_cols)
        if is_piv[c] != 0:
            continue
        if tries < 64 * L4 and (col_cnt[c] < lo or col_cnt[c] > hi):
            continue
        dup = False
        for j in range(k):
            if win[j] == c:
                dup = True
                break
        if dup:
            continue
        win[k] = c
        k += 1
    for q in range(n_light):                  # keys of the light rows only
        i = light_idx[q]
        key = 0
        for j in range(L4):
            c = win[j]
            if (A[i, c >> 6] >> np.uint64(c & 63)) & np.uint64(1):
                key |= 1 << j
        keys[i] = key
    nb = 1 << L4
    npairs = 0
    for c in range(n_light):                  # pair table over the light list
        ic = light_idx[c]
        for d in range(c + 1, n_light):
            idd = light_idx[d]
            pkey[npairs] = keys[ic] ^ keys[idd]
            pa[npairs] = ic
            pb[npairs] = idd
            npairs += 1
    for b in range(nb + 1):
        bstart[b] = 0
    for q in range(npairs):
        bstart[pkey[q] + 1] += 1
    for b in range(nb):
        bstart[b + 1] += bstart[b]
    for q in range(npairs):
        kb = pkey[q]
        order[bstart[kb]] = q
        bstart[kb] += 1
    cap = 4 * (npairs // nb + 1)
    nR = R if R < n_light else n_light
    for a in range(nR):
        ia = light_idx[a]
        for bb in range(n_light):
            if bb == a:
                continue
            ib = light_idx[bb]
            kk = keys[ia] ^ keys[ib]
            s = 0 if kk == 0 else bstart[kk - 1]
            e = bstart[kk]
            if e <= s or e - s > cap:
                continue
            wab = 0
            for w in range(nw):
                x = A[ia, w] ^ A[ib, w]
                pair_tmp[w] = x
                wab += int(_popcount_u64(x))
            for p in range(s, e):
                q = order[p]
                ic = pa[q]
                idd = pb[q]
                if ic == ia or ic == ib or idd == ia or idd == ib:
                    continue
                if wab - row_wt[ic] - row_wt[idd] >= min_w:
                    continue
                nz = False
                for w in range(nw, nwt):
                    if (A[ia, w] ^ A[ib, w] ^ A[ic, w] ^ A[idd, w]) != 0:
                        nz = True
                        break
                if not nz:
                    continue
                w4 = 0
                for w in range(nw):
                    w4 += int(_popcount_u64(pair_tmp[w] ^ A[ic, w] ^ A[idd, w]))
                if w4 < min_w:
                    min_w = w4
                    for w in range(nw):
                        shot_cw[w] = pair_tmp[w] ^ A[ic, w] ^ A[idd, w]
                    for w in range(nw, nwt):
                        shot_cw[w] = A[ia, w] ^ A[ib, w] ^ A[ic, w] ^ A[idd, w]
    return min_w


@njit(cache=True)
def _descend(A, row_wt, nw, cw, cur, max_passes):
    """Coset-leader descent: repeatedly add a basis row that lowers the weight
    of `cw` while its certification words stay non-zero (stabilizer rows are
    allowed, they do not change the logical class). First improvement, at most
    `max_passes` sweeps. A row of weight >= 2 wt(cw) can never help, so those
    are skipped. Returns the final weight."""
    m, nwt = A.shape
    for _p in range(max_passes):
        improved = False
        lim = 2 * cur
        for i in range(m):
            if row_wt[i] >= lim:
                continue
            wt = _xor_weight_row(cw, A, i, nw)
            if wt < cur:
                nz = False
                for w in range(nw, nwt):
                    if (cw[w] ^ A[i, w]) != 0:
                        nz = True
                        break
                if nz:
                    for w in range(nwt):
                        cw[w] ^= A[i, w]
                    cur = wt
                    lim = 2 * cur
                    improved = True
        if not improved:
            break
    return cur


@njit(cache=True)
def _greedy_reduce(A, row_wt, nw, shot_cw, min_w, max_passes, n_kicks,
                   kick_size, vary_kicks, inc_cw, best_cw, tmp_cw):
    """Descent on `shot_cw`, then optional iterated local search: perturb the
    incumbent by adding a few light-ish rows (lightest of three random rows
    each), re-descend, accept better-or-equal genuine logicals, keep the best.
    Off by default (n_kicks = 0)."""
    m, nwt = A.shape
    min_w = _descend(A, row_wt, nw, shot_cw, min_w, max_passes)
    if n_kicks <= 0 or m < 2:
        return min_w
    best_w = min_w
    inc_w = min_w
    for w in range(nwt):
        best_cw[w] = shot_cw[w]
        inc_cw[w] = shot_cw[w]
    for k in range(n_kicks):
        for w in range(nwt):
            tmp_cw[w] = inc_cw[w]
        ks = kick_size + (k % 3) if vary_kicks else kick_size
        for _s in range(ks):
            ri = np.random.randint(0, m)
            wri = row_wt[ri]
            for _t in range(2):
                rj = np.random.randint(0, m)
                if row_wt[rj] < wri:
                    ri = rj
                    wri = row_wt[rj]
            for w in range(nwt):
                tmp_cw[w] ^= A[ri, w]
        cur = 0
        for w in range(nw):
            cur += int(_popcount_u64(tmp_cw[w]))
        cur = _descend(A, row_wt, nw, tmp_cw, cur, max_passes)
        nz = False
        for w in range(nw, nwt):
            if tmp_cw[w] != 0:
                nz = True
                break
        if nz and cur <= inc_w:
            inc_w = cur
            for w in range(nwt):
                inc_cw[w] = tmp_cw[w]
            if cur < best_w:
                best_w = cur
                for w in range(nwt):
                    best_cw[w] = tmp_cw[w]
    for w in range(nwt):
        shot_cw[w] = best_cw[w]
    return best_w


@njit(cache=True)
def _reinforce_cw(theta, cw, nw, inc):
    """theta += inc on the support of the packed word cw."""
    for w in range(nw):
        word = cw[w]
        base = w << 6
        while word != 0:
            lowbit = word & (np.uint64(0) - word)
            col = base + int(_popcount_u64(lowbit - np.uint64(1)))
            word ^= lowbit
            theta[col] += inc


@njit(cache=True)
def _reinforce_rows(A, theta, row_wt, is_log, best_w, margin, eta, decay,
                    n_cols, nw):
    """Decay theta, then reinforce it on every logical row within `margin` of
    the walker's best weight (lighter rows get a larger increment)."""
    if decay > 0.0:
        for c in range(n_cols):
            theta[c] *= (1.0 - decay)
    thr = best_w + margin
    m = A.shape[0]
    for i in range(m):
        if not is_log[i]:
            continue
        wt = row_wt[i]
        if wt > thr:
            continue
        _reinforce_cw(theta, A[i], nw, eta * float(thr - wt + 1))


@njit(cache=True)
def _mix_rows(A, row_wt, is_log, nw):
    """row i ^= row j for random i != j (experimental drift, see MIX_RATE)."""
    m, nwt = A.shape
    if m < 2:
        return
    i = np.random.randint(0, m)
    j = np.random.randint(0, m - 1)
    if j >= i:
        j += 1
    wt = 0
    for w in range(nw):
        a = A[i, w] ^ A[j, w]
        A[i, w] = a
        wt += int(_popcount_u64(a))
    row_wt[i] = wt
    nz = False
    for w in range(nw, nwt):
        a = A[i, w] ^ A[j, w]
        A[i, w] = a
        if a != 0:
            nz = True
    is_log[i] = nz


# ════════════════════════════════════════════════════════════════════════════
# 6. Island loop (one process)
# ════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _run_island(A0, n_cols, nw, swaps_per_shot, restart_interval, shots, seed,
                cutoff, etas, decays, margins, tourns, epss, migrate_interval,
                migrate_boost, cw_max, lb_M, lb_p, greedy_passes, greedy_kicks,
                kick_size, mix_rate, stall_rounds, kick_rounds, kick_frac, kick_patience,
                lb_hash_l, lb_hash_windows, lb_R, lb_hash_p4, restart_from_h,
                explorer_restart, Hp, colsig, stop):
    """Run W = len(etas) walkers in lockstep for `shots` rounds.

    A0: packed basis of ker(check) with certification words appended.
    stop: int64 array of length >= 1, shared by all islands of a search. The
    island returns at the start of any round with stop[0] != 0, and sets
    stop[0] = 1 itself when it reaches the cutoff.
    mix_rate, stall_rounds: experimental diversification (see MIX_RATE and
    STALL_ROUNDS); with both at 0 the island is unchanged.
    kick_rounds, kick_frac, kick_patience: per-walker ILS kick (see KICK_ROUNDS,
    KICK_PATIENCE); kick_rounds = 0 turns it off.
    lb_hash_l, lb_hash_windows, lb_R: Stern-style hashed read-off (LB_HASH_L);
    lb_hash_l = 0 keeps the exhaustive read-off.
    restart_from_h, explorer_restart, Hp, colsig: restarts from the packed check
    matrix Hp (r x nw) with the per-column signature words colsig (n x ns), and
    the explorer's own restart cadence (see RESTART_FROM_H, EXPLORER_RESTART);
    restart_from_h = 0 ignores Hp and colsig (they may be empty).
    Returns (best_w, best_cw, cw_buf[:cnt], first_hit_round, rounds_done).
    cw_buf collects every hit of the current best weight (up to cw_max)."""
    np.random.seed(seed)
    W = etas.shape[0]
    m, nwt = A0.shape
    big = m * nw > BIG_BASE
    if big:
        g_passes = 2 if greedy_passes > 2 else greedy_passes
        n_kicks = 2 if greedy_kicks > 2 else greedy_kicks
        vary_kicks = False
        greedy_couples = True
    else:
        g_passes = greedy_passes
        n_kicks = greedy_kicks
        vary_kicks = True
        greedy_couples = False
    A_all = np.empty((W, m, nwt), dtype=np.uint64)
    piv_all = np.zeros((W, m), dtype=np.int64)
    wt_all = np.zeros((W, m), dtype=np.int64)
    log_all = np.zeros((W, m), dtype=np.bool_)
    theta_all = np.zeros((W, n_cols), dtype=np.float64)
    walker_best = np.full(W, n_cols + 1, dtype=np.int64)
    wbest_t = np.zeros(W, dtype=np.int64)            # round of the walker's last improvement
    wgap = np.zeros(W, dtype=np.int64)               # longest gap between its improvements
    wbest_cw = np.zeros((W, nwt), dtype=np.uint64)   # the walker's own best word
    kick_now = np.zeros(W, dtype=np.uint8)           # restart requested by a kick
    perm = np.arange(n_cols)
    perm_key = np.empty(n_cols, dtype=np.float64)
    tab = np.zeros((1 << max(swaps_per_shot, M4RI_K), nwt), dtype=np.uint64)
    mu_list = np.empty(swaps_per_shot, dtype=np.int64)
    uc = np.empty(swaps_per_shot, dtype=np.int64)
    u_rows = np.empty((swaps_per_shot, nwt), dtype=np.uint64)
    ws_rows = np.empty(swaps_per_shot, dtype=np.int64)
    in_ws = np.zeros(m, dtype=np.uint8)
    light_idx = np.empty(max(lb_M, 1), dtype=np.int64)
    light_wt = np.empty(max(lb_M, 1), dtype=np.int64)
    shot_cw = np.zeros(nwt, dtype=np.uint64)
    pair_tmp = np.zeros(nwt, dtype=np.uint64)
    hL = lb_hash_l if lb_hash_l > 0 else 1              # hashed read-off scratch
    if lb_hash_l > 0:                                   # window bits grow with the basis
        while (1 << hL) < m and hL < 20:                # so that buckets hold ~1 row
            hL += 1
    h_keys = np.empty(m, dtype=np.int64)
    h_bstart = np.empty((1 << hL) + 1, dtype=np.int64)
    h_border = np.empty(m, dtype=np.int64)
    h_win = np.empty(hL, dtype=np.int64)
    h_ispiv = np.zeros(n_cols, dtype=np.uint8)
    h_cnt = np.zeros(n_cols + 1, dtype=np.int64)
    L4 = min(hL + 4, 20) if (lb_hash_l > 0 and lb_hash_p4 > 0) else 1
    n_pairs = (lb_M * (lb_M - 1)) // 2 if (lb_hash_l > 0 and lb_hash_p4 > 0) else 1
    h4_keys = np.empty(m, dtype=np.int64)
    h4_win = np.empty(L4, dtype=np.int64)
    h4_pkey = np.empty(n_pairs, dtype=np.int64)
    h4_pa = np.empty(n_pairs, dtype=np.int64)
    h4_pb = np.empty(n_pairs, dtype=np.int64)
    h4_bstart = np.empty((1 << L4) + 1, dtype=np.int64)
    h4_order = np.empty(n_pairs, dtype=np.int64)
    inc_cw = np.zeros(nwt, dtype=np.uint64)
    best_tmp = np.zeros(nwt, dtype=np.uint64)
    tmp_cw = np.zeros(nwt, dtype=np.uint64)
    use_h = restart_from_h != 0 and Hp.shape[0] > 0 and colsig.shape[0] == n_cols
    Hs = np.empty((Hp.shape[0] if use_h else 0, nw), dtype=np.uint64)
    pivH = np.empty(Hp.shape[0] if use_h else 0, dtype=np.int64)
    h_idx = np.empty(n_cols, dtype=np.int64)
    perm_h = np.empty(n_cols, dtype=np.int64)
    er = explorer_restart if explorer_restart > 0 else restart_interval

    best_w = n_cols + 1
    best_cw = np.zeros(nwt, dtype=np.uint64)
    cw_buf = np.zeros((cw_max, nwt), dtype=np.uint64)
    cw_cnt = 0
    first_hit = -1
    # current "epoch" of the island: its best word drives migration and the
    # stall test; a stall reseed starts a new epoch (best_w / cw_buf persist)
    ep_w = n_cols + 1
    ep_cw = np.zeros(nwt, dtype=np.uint64)
    ep_t = 0
    reseed = False

    for t in range(shots):
        if stop[0] != 0:
            return best_w, best_cw, cw_buf[:cw_cnt], first_hit, t
        if reseed:                            # a fresh island in place
            for w in range(W):
                walker_best[w] = n_cols + 1
                for c in range(n_cols):
                    theta_all[w, c] = 0.0
            ep_w = n_cols + 1
            ep_t = t
        for w in range(W):
            if t % restart_interval == 0 or reseed or kick_now[w] != 0 \
                    or (epss[w] >= 0.9 and t % er == 0):
                kick_now[w] = 0
                # Restart: rebuild the basis with a fresh column order, either
                # by re-reducing A0 or (restart_from_h) from the check matrix.
                # Learners order columns by theta plus noise (low theta first,
                # so learned supports stay out of the information set);
                # the explorer uses a uniform random order.
                if epss[w] >= 0.9:
                    np.random.shuffle(perm)
                else:
                    tmax = 0.0
                    for c in range(n_cols):
                        if theta_all[w, c] > tmax:
                            tmax = theta_all[w, c]
                    scale = 0.5 * tmax + 1e-9
                    for c in range(n_cols):
                        perm_key[c] = theta_all[w, c] + np.random.random() * scale
                    perm = np.argsort(perm_key)
                built = -1
                if use_h:
                    # H's pivots leave the information set, so the order is
                    # reversed: high theta first (the explorer's is uniform).
                    if epss[w] >= 0.9:
                        for c in range(n_cols):
                            perm_h[c] = perm[c]
                    else:
                        for c in range(n_cols):
                            perm_h[c] = perm[n_cols - 1 - c]
                    built = _restart_from_h(Hp, Hs, perm_h, M4RI_K, tab, pivH, colsig,
                                            A_all[w], piv_all[w], h_idx, nw)
                if built < 0:
                    for r in range(m):
                        for c in range(nwt):
                            A_all[w, r, c] = A0[r, c]
                    _rref_m4ri_order(A_all[w], perm, M4RI_K, tab, piv_all[w])
                _row_stats(A_all[w], nw, wt_all[w], log_all[w])
            else:
                _swap_batch(A_all[w], piv_all[w], wt_all[w], log_all[w],
                            theta_all[w], nw, swaps_per_shot, tourns[w], epss[w],
                            mu_list, u_rows, ws_rows, in_ws, tab, uc)
                if mix_rate > 0.0:
                    k = int(mix_rate)
                    if np.random.random() < mix_rate - k:
                        k += 1
                    for _q in range(k):
                        _mix_rows(A_all[w], wt_all[w], log_all[w], nw)
            if lb_hash_l > 0:
                min_w, _n_light = _readoff_hashed(A_all[w], wt_all[w], log_all[w], piv_all[w],
                                                  nw, lb_M, lb_R, hL, lb_hash_windows,
                                                  light_idx, light_wt, shot_cw, pair_tmp,
                                                  h_keys, h_bstart, h_border, h_win, h_ispiv,
                                                  h_cnt)
                if lb_hash_p4 > 0:
                    min_w = _readoff_p4(A_all[w], wt_all[w], nw, _n_light, lb_R, L4,
                                        light_idx, min_w, shot_cw, pair_tmp, h_ispiv, h_cnt,
                                        h4_keys, h4_win, h4_pkey, h4_pa, h4_pb, h4_bstart,
                                        h4_order)
            else:
                min_w, _n_light = _readoff(A_all[w], wt_all[w], log_all[w], nw, lb_M,
                                           lb_p, light_idx, light_wt, shot_cw, pair_tmp)
            if greedy_couples and g_passes > 0 and min_w <= n_cols:
                min_w = _greedy_reduce(A_all[w], wt_all[w], nw, shot_cw, min_w,
                                       g_passes, n_kicks, kick_size, vary_kicks,
                                       inc_cw, best_tmp, tmp_cw)
            if min_w < walker_best[w]:
                walker_best[w] = min_w
                if t - wbest_t[w] > wgap[w]:
                    wgap[w] = t - wbest_t[w]
                wbest_t[w] = t
                for c in range(nwt):
                    wbest_cw[w, c] = shot_cw[c]
            _reinforce_rows(A_all[w], theta_all[w], wt_all[w], log_all[w],
                            walker_best[w], margins[w], etas[w], decays[w],
                            n_cols, nw)
            if min_w <= walker_best[w] + margins[w]:
                _reinforce_cw(theta_all[w], shot_cw, nw,
                              etas[w] * float(walker_best[w] + margins[w] - min_w + 1))
            if (not greedy_couples) and g_passes > 0 and min_w <= n_cols:
                # On code-sized bases the descended word only improves what is
                # reported; feeding it to theta would flood the learning.
                min_w = _greedy_reduce(A_all[w], wt_all[w], nw, shot_cw, min_w,
                                       g_passes, n_kicks, kick_size, vary_kicks,
                                       inc_cw, best_tmp, tmp_cw)
            if min_w < ep_w:
                ep_w = min_w
                ep_t = t
                for c in range(nwt):
                    ep_cw[c] = shot_cw[c]
            if min_w < best_w:
                best_w = min_w
                for c in range(nwt):
                    best_cw[c] = shot_cw[c]
                cw_cnt = 0
            if min_w == best_w and min_w <= n_cols and cw_cnt < cw_max:
                for c in range(nwt):
                    cw_buf[cw_cnt, c] = shot_cw[c]
                cw_cnt += 1
        if kick_rounds > 0:
            for w in range(W):
                if epss[w] >= 0.9 or t - wbest_t[w] < kick_rounds:
                    continue
                if kick_patience > 0.0 and t - wbest_t[w] < kick_patience * wgap[w]:
                    continue
                # ILS kick: forget a fraction kick_frac of the priorities on the
                # support of this walker's best word (the priorities have collapsed
                # onto it), forget the best weight so learning resumes from the
                # perturbed region, and rebuild the basis at the next round.
                if kick_frac >= 1.0:
                    for c in range(n_cols):
                        theta_all[w, c] = 0.0
                else:
                    for wd in range(nw):
                        word = wbest_cw[w, wd]
                        base = wd << 6
                        while word != 0:
                            lowbit = word & (np.uint64(0) - word)
                            col = base + int(_popcount_u64(lowbit - np.uint64(1)))
                            word ^= lowbit
                            if np.random.random() < kick_frac:
                                theta_all[w, col] = 0.0
                walker_best[w] = n_cols + 1
                wbest_t[w] = t
                wgap[w] = 0
                kick_now[w] = 1
        if cutoff > 0 and best_w <= cutoff and first_hit < 0:
            first_hit = t + 1
            stop[0] = 1
            return best_w, best_cw, cw_buf[:cw_cnt], first_hit, t + 1
        reseed = stall_rounds > 0 and t - ep_t >= stall_rounds
        if migrate_boost > 0.0 and ep_w <= n_cols and not reseed \
                and (t % migrate_interval) == (migrate_interval - 1):
            for w in range(W):
                if epss[w] >= 0.9:
                    continue
                _reinforce_cw(theta_all[w], ep_cw, nw, migrate_boost)
    return best_w, best_cw, cw_buf[:cw_cnt], first_hit, shots


# ════════════════════════════════════════════════════════════════════════════
# 7. Setup and multiprocess harness
# ════════════════════════════════════════════════════════════════════════════

def _kernel_packed(H, n_cols, k):
    """Packed basis of ker(H) for a uint8 matrix H."""
    tab = np.zeros((1 << k, (n_cols + 63) // 64), dtype=np.uint64)
    return _nullspace_packed_u64(pack_gf2_matrix_u64(H), n_cols, k, tab)


def _side_setup(check_H, dual_H, pair_rows, proj_words, seed, m4ri_k,
                G_packed=None, return_h=False):
    """Build A0 = [ker(check) | certification words].

    The certification words pair each kernel row with 64*proj_words random
    elements of the pairing space: ker(dual_H) when `pair_rows` is None,
    otherwise the row space of `pair_rows` (which must be a generating set of
    ker(dual_H), e.g. the other side's kernel, or the observable matrix of a
    detector error model). Returns (A0, n_cols, nw); with return_h also the
    packed check matrix Hp and the per-column signature words colsig that
    restarts from H need (RESTART_FROM_H)."""
    check_u8 = np.ascontiguousarray(check_H, dtype=np.uint8)
    n_cols = check_u8.shape[1]
    nw = (n_cols + 63) // 64
    Hp = pack_gf2_matrix_u64(check_u8)
    if G_packed is None:
        tab = np.zeros((1 << m4ri_k, nw), dtype=np.uint64)
        G = _nullspace_packed_u64(Hp, n_cols, m4ri_k, tab)
    else:
        G = G_packed
    if G.shape[0]:
        s = min(64, G.shape[0])
        idx = np.random.default_rng(0).choice(G.shape[0], size=s, replace=False)
        Gs = unpack_gf2_matrix_u64(G[idx], n_cols)
        assert not (check_u8 @ Gs.T % 2).any(), "kernel basis fails the check"
    if pair_rows is None:
        if dual_H is None:
            raise ValueError("give dual_H or pair_rows")
        P = _kernel_packed(np.ascontiguousarray(dual_H, dtype=np.uint8), n_cols, m4ri_k)
    else:
        P = pack_gf2_matrix_u64(np.ascontiguousarray(pair_rows, dtype=np.uint8))
    ns = max(1, int(proj_words))
    rng = np.random.default_rng(seed)
    coeff = rng.integers(0, 2, size=(64 * ns, P.shape[0]), dtype=np.uint8)
    Rc = _random_combinations(P, coeff)
    Sig = _pairing_words(G, Rc, ns)
    A0 = np.ascontiguousarray(np.hstack([G, Sig]))
    if return_h:
        return A0, n_cols, nw, Hp, _sig_transpose(Rc, n_cols, ns)
    return A0, n_cols, nw


def _side_setup_compact(check_H, dual_H, pair_rows, proj_words, seed, m4ri_k):
    """Inputs of the compact walker: packed check matrix Hp, its rank, and the
    per-column signature words colsig (pairings with 64*proj_words random
    elements of ker(dual_H) or of rowspan(pair_rows)). No kernel basis.
    Returns (Hp, colsig, n_cols, nw, rank)."""
    check_u8 = np.ascontiguousarray(check_H, dtype=np.uint8)
    n_cols = check_u8.shape[1]
    nw = (n_cols + 63) // 64
    Hp = pack_gf2_matrix_u64(check_u8)
    tab = np.zeros((1 << m4ri_k, nw), dtype=np.uint64)
    Hs = Hp.copy()
    pivH = np.empty(Hp.shape[0], dtype=np.int64)
    rank = int(_rref_m4ri_order(Hs, np.arange(n_cols), m4ri_k, tab, pivH))
    if pair_rows is None:
        if dual_H is None:
            raise ValueError("give dual_H or pair_rows")
        Pk = _kernel_packed(np.ascontiguousarray(dual_H, dtype=np.uint8), n_cols, m4ri_k)
    else:
        Pk = pack_gf2_matrix_u64(np.ascontiguousarray(pair_rows, dtype=np.uint8))
    ns = max(1, int(proj_words))
    rng = np.random.default_rng(seed)
    coeff = rng.integers(0, 2, size=(64 * ns, Pk.shape[0]), dtype=np.uint8)
    Rc = _random_combinations(Pk, coeff)
    return Hp, _sig_transpose(Rc, n_cols, ns), n_cols, nw, rank


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


def _island_args_compact(n_cols, nw, rank, rounds, seed, cutoff, p, W=ISLAND_W,
                         restart_interval=None, theta_prior=None):
    """Arguments of _run_island_compact after (Hp, colsig) and before the stop flag."""
    prior = (np.zeros(0, dtype=np.float64) if theta_prior is None
             else np.ascontiguousarray(theta_prior, dtype=np.float64))
    ri = restart_interval
    if ri is None:
        ri = p["restart_interval"]
    if ri is None:
        ri = int(min(2000, max(RESTART_INTERVAL, (n_cols - rank) // 3)))
    etas, decays, margins, tourns, epss = _portfolio_arrays(W)
    return (int(n_cols), int(nw), int(rank), int(p["swaps_per_shot"]), int(ri), int(rounds),
            int(seed), int(cutoff), etas, decays, margins, tourns, epss,
            int(p["migrate_interval"]), float(p["migrate_boost"]), int(p["max_codewords"]),
            int(p["lb_M"]), int(p["greedy_passes"]), int(p["stall_rounds"]),
            int(p["kick_rounds"]), float(p["kick_frac"]), float(p["kick_patience"]),
            int(p["lb_hash_l"]), int(p["lb_hash_windows"]), int(p["lb_R"]),
            int(p["explorer_restart"] or 0), prior)


def _collect_results(res, W, n_cols, nw, collect, wall):
    res = [r for r in res if r[4] > 0]                # islands that actually ran
    if not res:
        return SideResult(distance=n_cols + 1, codewords=np.zeros((0, n_cols), np.uint8),
                          islands=0, wall_time=wall)
    if any(r[1] < 0 for r in res):
        raise RuntimeError("compact walker: rank of the check matrix changed between restarts")
    total_shots = sum(r[4] for r in res) * W
    best_w = min(r[1] for r in res)
    # time at which the best weight was first reached, estimated from the round of the
    # best in the island that holds it (rounds are roughly uniform in time)
    t_best = -1.0
    for r in res:
        if r[1] == best_w and len(r) > 5 and r[5] >= 0 and r[4] > 0:
            est = wall * (r[5] + 1) / r[4]
            t_best = est if t_best < 0 else min(t_best, est)
    codes = np.zeros((0, n_cols), dtype=np.uint8)
    counts = np.zeros(0, dtype=np.int64)
    if best_w <= n_cols:
        bufs = [r[3] for r in res if r[1] == best_w and r[3].shape[0] > 0]
        packed = np.vstack(bufs) if bufs else \
            next(r[2] for r in res if r[1] == best_w).reshape(1, -1)
        if not collect:
            packed = packed[:1]
        codes = unpack_gf2_matrix_u64(np.ascontiguousarray(packed[:, :nw]),
                                      n_cols).astype(np.uint8)
        codes, counts = np.unique(codes, axis=0, return_counts=True)
        assert (codes.sum(axis=1) == best_w).all()
    return SideResult(distance=int(best_w), codewords=codes, counts=counts,
                      shots=int(total_shots), wall_time=wall, islands=len(res), t_best=t_best)


def _search_side_compact(Hp, colsig, n_cols, nw, rank, num_workers, num_trials, island_shots,
                         num_seeds, W, cutoff, rng, collect, pool, stop, p, theta_prior=None):
    """_search_side for the compact walker (Hp, colsig shared with the workers
    through _SHARED_C, set by the caller before the pool was created)."""
    m = n_cols - rank
    if m == 0:
        return SideResult(distance=n_cols + 1, codewords=np.zeros((0, n_cols), np.uint8),
                          islands=0)
    rounds, n_islands, last = _island_plan(num_trials, num_workers, island_shots, num_seeds, W)
    wins = _SHARED_WIN if (p.get("windows", 0) and _SHARED_WIN) else None
    tasks = []
    acc = 0.0
    for i in range(n_islands):
        r_i = rounds if i < n_islands - 1 else last
        acc += float(p.get("window_frac", 0.0)) if wins else 0.0
        if acc >= 1.0 - 1e-9:                         # a window island (interleaved)
            acc -= 1.0
            levels = sorted(set(w[6] for w in wins))        # level first, then a window in it
            lev = levels[int(rng.integers(len(levels)))]
            pool_l = [q for q, w in enumerate(wins) if w[6] == lev]
            widx = pool_l[int(rng.integers(len(pool_l)))]
            cols, _H, _c, n_s, nw_s, rk_s = wins[widx][:6]
            pr = None if theta_prior is None else np.ascontiguousarray(theta_prior[cols])
            # rounds grow with the window's rows (n_s - rank): small windows converge in a
            # few hundred rounds, a code-sized window needs tens of thousands
            r_w = int(p["window_rounds"]) * max(1.0, (n_s - rk_s) / 1000.0)
            tasks.append((i, "cwin", widx, _island_args_compact(
                n_s, nw_s, rk_s, int(min(r_i, r_w)), int(rng.integers(2**31)),
                cutoff, p, W, theta_prior=pr)))
        else:
            tasks.append((i, "compact", _island_args_compact(n_cols, nw, rank, r_i,
                                                             int(rng.integers(2**31)), cutoff, p, W,
                                                             theta_prior=theta_prior)))
    stop[0] = 0
    t0 = time.perf_counter()
    if pool is None:
        from .compact import _run_island_compact
        res = []
        for task in tasks:
            if stop[0] != 0:
                break
            if task[1] == "cwin":
                res.append(_island_worker(task))
                continue
            i, _tag, a = task
            bw, bcw, cwbuf, _fh, rd, br = _run_island_compact(Hp, colsig, *a, stop)
            res.append((i, bw, bcw, cwbuf, rd, br))
    else:
        res = sorted(pool.imap_unordered(_island_worker, tasks, chunksize=1),
                     key=lambda r: r[0])
    return _collect_results(res, W, n_cols, nw, collect, time.perf_counter() - t0)


def _portfolio_arrays(W=ISLAND_W):
    """Walker presets for an island of W walkers (the presets of _PORTFOLIO
    are used in turn)."""
    etas = np.empty(W); decays = np.empty(W)
    margins = np.empty(W, np.int64); tourns = np.empty(W, np.int64)
    epss = np.empty(W)
    for w in range(W):
        eta, dec, mar, tr, ep = _PORTFOLIO[w % len(_PORTFOLIO)]
        etas[w] = eta; decays[w] = dec; margins[w] = mar
        tourns[w] = tr; epss[w] = ep
    return etas, decays, margins, tourns, epss


def _island_args(A0, n_cols, nw, rounds, seed, cutoff, p, W=ISLAND_W, restart_interval=None,
                 Hp=None, colsig=None):
    """Argument tuple of _run_island (without the stop flag) for parameters p.
    Hp, colsig (from _side_setup(..., return_h=True)) enable restart_from_h;
    without them that knob is ignored."""
    if Hp is None or colsig is None:
        Hp = np.zeros((0, nw), dtype=np.uint64)
        colsig = np.zeros((0, 1), dtype=np.uint64)
    ri = restart_interval
    if ri is None:
        ri = p["restart_interval"]
    if ri is None:
        ri = int(min(2000, max(RESTART_INTERVAL, A0.shape[0] // 3)))
    etas, decays, margins, tourns, epss = _portfolio_arrays(W)
    return (A0, n_cols, nw, p["swaps_per_shot"], ri, int(rounds), int(seed), int(cutoff),
            etas, decays, margins, tourns, epss, p["migrate_interval"], p["migrate_boost"],
            p["max_codewords"], p["lb_M"], p["lb_p"], p["greedy_passes"], p["greedy_kicks"],
            p["kick_size"], float(p["mix_rate"]), int(p["stall_rounds"]),
            int(p["kick_rounds"]), float(p["kick_frac"]), float(p["kick_patience"]),
            int(p["lb_hash_l"]), int(p["lb_hash_windows"]), int(p["lb_R"]), int(p["lb_hash_p4"]),
            int(p["restart_from_h"]), int(p["explorer_restart"] or 0),
            np.ascontiguousarray(Hp, dtype=np.uint64), np.ascontiguousarray(colsig, dtype=np.uint64))


_STOP = None   # this process's view of the search's shared stop flag
_SHARED_C = None  # (Hp, colsig) of the current compact search (see _SHARED)
_SHARED_WIN = None  # detector-window sub-instances of the current search (WINDOWS)
_SHARED_NW = 0
_SHARED_NCOLS = 0
_SHARED = None # (A0, Hp, colsig) of the current search, inherited by forked workers
               # so that the basis is not pickled once per island (2026-11-04; a
               # circuit-DEM basis is 80-200 MB)


def _raise_stop(stop):
    """Timer target of side_distance(time_limit=...): stop every island."""
    stop[0] = 1


def _pool_init(raw):
    global _STOP
    _STOP = np.frombuffer(raw, dtype=np.int64)


def _make_pool(num_workers):
    """Worker processes sharing one int64 stop flag. Returns (pool, flag);
    pool is None when the search runs in this process."""
    if num_workers <= 1:
        return None, np.zeros(1, dtype=np.int64)
    raw = mp.RawArray("q", 1)
    pool = mp.Pool(processes=num_workers, initializer=_pool_init, initargs=(raw,))
    return pool, np.frombuffer(raw, dtype=np.int64)


def _island_worker(a):
    """Pool entry point: one island. Returns (index, best_w, best_cw, cw_buf, rounds_done).
    A task whose A0 is None uses the arrays inherited from the parent (_SHARED);
    a 3-tuple (i, "compact", args) runs the compact walker with _SHARED_C."""
    stop = _STOP if _STOP is not None else np.zeros(1, dtype=np.int64)
    if len(a) == 3:
        i, _tag, args = a
        from .compact import _run_island_compact
        Hp, colsig = _SHARED_C
        bw, bcw, cwbuf, _fh, rounds_done, best_round = _run_island_compact(Hp, colsig, *args, stop)
        return i, bw, bcw, cwbuf, rounds_done, best_round
    if len(a) == 4:                                   # (i, "cwin", window index, args)
        i, _tag, widx, args = a
        from .compact import _run_island_compact
        cols, Hp_s, cs, _n_s, _nw_s, _rk, _lev, _wa, _ws = _SHARED_WIN[widx]
        nw = _SHARED_NW
        bw, bcw, cwbuf, _fh, rounds_done, best_round = _run_island_compact(Hp_s, cs, *args, stop)
        if bw > cols.size:                            # nothing found in the window
            return i, _SHARED_NCOLS + 1, np.zeros(nw, np.uint64), np.zeros((0, nw), np.uint64), \
                rounds_done, best_round
        bcw_f = _expand_window_cw(bcw, cols, nw)
        return i, bw, bcw_f, _expand_window_buf(cwbuf, cols, nw), rounds_done, best_round
    i, args = a
    if args[0] is None:
        args = list(args)
        args[0], args[-2], args[-1] = _SHARED
        args = tuple(args)
    bw, bcw, cwbuf, _fh, rounds_done = _run_island(*args, stop)
    return i, bw, bcw, cwbuf, rounds_done


def _available_cores():
    """CPU cores this process may use (respects Slurm and cgroup limits on Linux)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _resolve_workers(num_workers):
    if num_workers is None or num_workers == "auto":
        return _available_cores()
    return max(1, int(num_workers))


def _island_plan(num_trials, num_workers, island_shots, num_seeds, W):
    """(rounds per island, number of islands, rounds of the last island). A
    round is one shot for each of an island's W walkers. Default island
    budget: an equal share of the total per worker, capped at ISLAND_SHOTS.
    `num_seeds` (older interface) means num_seeds waves of num_workers islands."""
    total = max(1, int(num_trials))
    if island_shots is None:
        if num_seeds is not None:
            island_shots = -(-total // (max(1, int(num_seeds)) * num_workers))
        else:
            island_shots = min(ISLAND_SHOTS, -(-total // num_workers))
    rounds = max(1, -(-int(island_shots) // W))
    n_islands = max(1, -(-total // (rounds * W)))
    last = max(1, -(-total // W) - rounds * (n_islands - 1))
    return rounds, n_islands, last


def _search_side(A0, n_cols, nw, num_workers, num_trials, island_shots, num_seeds, W,
                 cutoff, rng, collect, pool, stop, p, Hp=None, colsig=None):
    """Run islands until `num_trials` shots are spent or one island reaches the
    cutoff. At most `num_workers` islands run at once. Each spends at most
    `island_shots` shots and is then replaced by a freshly seeded island. The
    island that reaches the cutoff raises the shared stop flag, and every
    other island returns within one round. Returns a SideResult."""
    m = A0.shape[0]
    if m == 0:
        return SideResult(distance=n_cols + 1, codewords=np.zeros((0, n_cols), np.uint8),
                          islands=0)
    ri = p["restart_interval"]
    if ri is None:
        ri = int(min(2000, max(RESTART_INTERVAL, m // 3)))   # restarts cost O(m^2)
    rounds, n_islands, last = _island_plan(num_trials, num_workers, island_shots,
                                           num_seeds, W)
    tasks = [(i, _island_args(A0, n_cols, nw, rounds if i < n_islands - 1 else last,
                              int(rng.integers(2**31)), cutoff, p, W, ri, Hp, colsig))
             for i in range(n_islands)]
    if pool is not None and _SHARED is not None and _SHARED[0] is A0:
        # the workers were forked after _SHARED was set: send the basis as None
        tasks = [(i, (None,) + a[1:-2] + (None, None)) for i, a in tasks]
    stop[0] = 0
    t0 = time.perf_counter()
    if pool is None:
        res = []
        for i, a in tasks:
            if stop[0] != 0:
                break
            bw, bcw, cwbuf, _fh, rd = _run_island(*a, stop)
            res.append((i, bw, bcw, cwbuf, rd))
    else:
        res = sorted(pool.imap_unordered(_island_worker, tasks, chunksize=1),
                     key=lambda r: r[0])
    wall = time.perf_counter() - t0
    res = [r for r in res if r[4] > 0]                # islands that actually ran
    total_shots = sum(r[4] for r in res) * W
    best_w = min(r[1] for r in res)
    codes = np.zeros((0, n_cols), dtype=np.uint8)
    counts = np.zeros(0, dtype=np.int64)
    if best_w <= n_cols:
        bufs = [r[3] for r in res if r[1] == best_w and r[3].shape[0] > 0]
        packed = np.vstack(bufs) if bufs else \
            next(r[2] for r in res if r[1] == best_w).reshape(1, -1)
        if not collect:
            packed = packed[:1]
        codes = unpack_gf2_matrix_u64(np.ascontiguousarray(packed[:, :nw]),
                                      n_cols).astype(np.uint8)
        codes, counts = np.unique(codes, axis=0, return_counts=True)
        assert (codes.sum(axis=1) == best_w).all()
    return SideResult(distance=int(best_w), codewords=codes, counts=counts,
                      shots=int(total_shots), wall_time=wall, islands=len(res))


# ════════════════════════════════════════════════════════════════════════════
# 8. Public API
# ════════════════════════════════════════════════════════════════════════════

def side_distance(check_H, dual_H, num_trials=20000, num_workers=1, seed=None,
                  cutoff=0, pair_rows=None, collect_codewords=True, num_seeds=None,
                  island_shots=None, walkers_per_island=ISLAND_W, time_limit=None,
                  compact=None, theta_prior=None, **params):
    """Lightest element of ker(check_H) outside rowspan(dual_H).

    check_H       binary matrix; candidates c satisfy check_H c = 0.
    dual_H        binary matrix whose row space is the trivial subspace of
                  ker(check_H) (for X logicals of a CSS code: check_H = Hz,
                  dual_H = Hx). May be None when `pair_rows` is given.
    pair_rows     optional generating set of ker(dual_H) used for
                  certification instead of computing that kernel; for a
                  detector error model pass the observable matrix L.
    num_trials    total shot budget over all islands (each shot = one
                  information-set move plus read-off for one walker).
    num_workers   islands run in parallel (processes); 1 runs in-process;
                  None = all CPU cores this process may use.
    island_shots  shot budget of one island. An island that spends it without
                  reaching the cutoff is replaced by a freshly seeded island
                  (fresh theta and random streams) until `num_trials` is
                  spent. None = an equal share of `num_trials` per worker,
                  capped at ISLAND_SHOTS. On hard instances the spread between
                  islands dominates, so many moderate islands beat a few long
                  ones.
    walkers_per_island  walkers per island (default 4: three independent
                  learners with different learning rates, each with a
                  per-walker kick, plus one uniform explorer; larger
                  islands reuse those presets in turn).
    cutoff        stop at the first codeword of weight <= cutoff; every running
                  island stops within one round (0 = run the whole budget,
                  which is what enumerates the codeword set).
    collect_codewords  keep the set of min-weight codewords found (else one).
    num_seeds     older interface: split the budget into num_seeds waves of
                  num_workers islands (ignored when island_shots is given).
    time_limit    wall-clock budget in seconds (2026-11-04): a timer raises the
                  shared stop flag, every running island returns within one
                  round and the result so far is returned. Needs num_workers
                  >= 2 (with one in-process island the kernel holds the GIL and
                  the timer cannot fire until it returns). None = no limit.
    compact       None = COMPACT; True runs the compact walker (rows stored on
                  the redundant columns, distance_finder_compact.py), False the
                  full-width walker of this module.
    theta_prior   optional array of n_cols initial priorities for the learners
                  (compact walker only): a warm start, e.g. a large value on the
                  support of a known light codeword; the first restart then keeps
                  that support out of the information set and the search explores
                  around it. None = cold start.
    **params      search knobs, see module constants (greedy_kicks, lb_p, ...).

    Returns a SideResult. Every codeword in it is certified: it satisfies the
    check and pairs non-trivially with an element of ker(dual_H).
    """
    p = _params(**params)
    W = max(1, int(walkers_per_island))
    num_workers = _resolve_workers(num_workers)
    num_workers = min(num_workers, _island_plan(num_trials, num_workers, island_shots,
                                                num_seeds, W)[1])
    rng = np.random.default_rng(seed)
    use_compact = COMPACT if compact is None else bool(compact)
    global _SHARED, _SHARED_C, _SHARED_WIN, _SHARED_NW, _SHARED_NCOLS
    if use_compact:
        Hp, colsig, n_cols, nw, rank = _side_setup_compact(
            check_H, dual_H, pair_rows, p["proj_words"], int(rng.integers(2**31)), p["m4ri_k"])
        _SHARED_C = (Hp, colsig)
        _SHARED_WIN, _SHARED_NW, _SHARED_NCOLS = None, nw, n_cols
        n_lev = p["windows"]
        if n_lev == "auto":                           # DEM-shaped instances only
            m_rows = np.asarray(check_H).shape[0]
            n_lev = WINDOW_AUTO_LEVELS if n_cols >= WINDOW_AUTO_RATIO * m_rows else 0
            p["windows"] = n_lev
        if int(n_lev) > 0:
            _SHARED_WIN = _detector_windows(check_H, colsig, int(n_lev),
                                            int(p["window_min"]), p["m4ri_k"]) or None
        if num_workers > 1 and n_cols > rank:
            # load the compiled kernels in this process before the pool is forked:
            # otherwise every worker imports the module and loads the cache itself
            # (2.6 s each, paid by every fresh pool; found 2026-11-04 on small DEMs)
            from .compact import _run_island_compact
            _run_island_compact(Hp, colsig, *_island_args_compact(n_cols, nw, rank, 1, 0, 0, p, 1,
                                                                  theta_prior=theta_prior),
                                np.zeros(1, dtype=np.int64))
    else:
        A0, n_cols, nw, Hp, colsig = _side_setup(check_H, dual_H, pair_rows, p["proj_words"],
                                                 int(rng.integers(2**31)), p["m4ri_k"],
                                                 return_h=True)
        _SHARED = (A0, Hp, colsig)
    pool, stop = _make_pool(num_workers)
    timer = None
    if time_limit is not None and time_limit > 0:
        import threading
        timer = threading.Timer(float(time_limit), _raise_stop, args=(stop,))
        timer.daemon = True
        timer.start()
    try:
        if use_compact:
            res = _search_side_compact(Hp, colsig, n_cols, nw, rank, num_workers, num_trials,
                                       island_shots, num_seeds, W, cutoff, rng,
                                       collect_codewords, pool, stop, p, theta_prior)
        else:
            res = _search_side(A0, n_cols, nw, num_workers, num_trials, island_shots,
                               num_seeds, W, cutoff, rng, collect_codewords, pool, stop, p,
                               Hp, colsig)
    finally:
        if timer is not None:
            timer.cancel()
        if pool is not None:
            pool.close(); pool.join()
        _SHARED = None
        _SHARED_C = None
        _SHARED_WIN = None
    if res.num_codewords:
        import scipy.sparse as _sp
        Hs_ = _sp.csr_matrix(np.ascontiguousarray(check_H, dtype=np.uint8), dtype=np.int32)
        Cs_ = _sp.csr_matrix(res.codewords, dtype=np.int32)
        syn = (Hs_ @ Cs_.T)
        assert not (syn.data % 2).any(), "codeword fails the check"
    return res


def css_distance(Hx, Hz, num_trials=20000, num_workers=1, seed=None, cutoff=0,
                 basis="both", collect_codewords=True, num_seeds=None,
                 island_shots=None, walkers_per_island=ISLAND_W, compact=None, **params):
    """Distance of the CSS code with checks Hx (X type) and Hz (Z type).

    basis   'both' searches Z logicals (ker Hx minus rowspan Hz) first and then
            X logicals; 'z' or 'x' searches one side only. A skipped side is
            returned with distance -1. With cutoff > 0, the X side is skipped
            when the Z side already reached the cutoff (the usual go/no-go use).
    Other arguments as in `side_distance` (`num_trials` applies per side). Returns a CssDistanceResult whose `.distance` is the minimum
    over the sides searched.
    """
    assert basis in ("both", "z", "x")
    use_compact = COMPACT if compact is None else bool(compact)
    if use_compact:
        Hx8 = np.ascontiguousarray(Hx, dtype=np.uint8)
        Hz8 = np.ascontiguousarray(Hz, dtype=np.uint8)
        n = Hx8.shape[1]
        assert Hz8.shape[1] == n
        assert not (Hx8 @ Hz8.T % 2).any(), "Hx Hz^T != 0: not a CSS code"
        empty = SideResult(distance=-1, codewords=np.zeros((0, n), dtype=np.uint8))
        x = z = empty
        rng = np.random.default_rng(seed)
        kw = dict(num_trials=num_trials, num_workers=num_workers, cutoff=cutoff,
                  collect_codewords=collect_codewords, num_seeds=num_seeds,
                  island_shots=island_shots, walkers_per_island=walkers_per_island,
                  compact=True, **params)
        if basis in ("both", "z"):
            z = side_distance(Hx8, Hz8, seed=int(rng.integers(2**31)), **kw)
            if cutoff > 0 and z.distance <= cutoff:
                return CssDistanceResult(x=x, z=z)
        if basis in ("both", "x"):
            x = side_distance(Hz8, Hx8, seed=int(rng.integers(2**31)), **kw)
        return CssDistanceResult(x=x, z=z)
    p = _params(**params)
    W = max(1, int(walkers_per_island))
    num_workers = _resolve_workers(num_workers)
    num_workers = min(num_workers, _island_plan(num_trials, num_workers, island_shots,
                                                num_seeds, W)[1])
    rng = np.random.default_rng(seed)
    Hx8 = np.ascontiguousarray(Hx, dtype=np.uint8)
    Hz8 = np.ascontiguousarray(Hz, dtype=np.uint8)
    n = Hx8.shape[1]
    assert Hz8.shape[1] == n
    assert not (Hx8 @ Hz8.T % 2).any(), "Hx Hz^T != 0: not a CSS code"
    empty = SideResult(distance=-1, codewords=np.zeros((0, n), dtype=np.uint8))
    x = z = empty
    # Each side's kernel is the other side's pairing space, so compute both once.
    ker_Hx = _kernel_packed(Hx8, n, p["m4ri_k"])      # Z-side candidates
    ker_Hz = _kernel_packed(Hz8, n, p["m4ri_k"])      # X-side candidates
    pool, stop = _make_pool(num_workers)
    try:
        if basis in ("both", "z"):
            A0, nc, nw, Hp, cs = _side_setup(Hx8, None, unpack_gf2_matrix_u64(ker_Hz, n),
                                             p["proj_words"], int(rng.integers(2**31)),
                                             p["m4ri_k"], G_packed=ker_Hx, return_h=True)
            z = _search_side(A0, nc, nw, num_workers, num_trials, island_shots,
                             num_seeds, W, cutoff, rng, collect_codewords, pool, stop, p,
                             Hp, cs)
            if cutoff > 0 and z.distance <= cutoff:
                return CssDistanceResult(x=x, z=z)
        if basis in ("both", "x"):
            A0, nc, nw, Hp, cs = _side_setup(Hz8, None, unpack_gf2_matrix_u64(ker_Hx, n),
                                             p["proj_words"], int(rng.integers(2**31)),
                                             p["m4ri_k"], G_packed=ker_Hz, return_h=True)
            x = _search_side(A0, nc, nw, num_workers, num_trials, island_shots,
                             num_seeds, W, cutoff, rng, collect_codewords, pool, stop, p,
                             Hp, cs)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    return CssDistanceResult(x=x, z=z)


def metacheck_distance(M, H, num_trials=20000, num_workers=1, seed=None, cutoff=0,
                       collect_codewords=True, num_seeds=None, island_shots=None,
                       walkers_per_island=ISLAND_W, **params):
    """Syndrome-space distance of a metachecked code: the lightest syndrome
    eta with M eta = 0 that is not a real syndrome (eta not in im H, i.e. not
    in rowspan(H^T)). This is the undetectable measurement-error weight of a
    single-shot code. Returns a SideResult over syndrome bits."""
    return side_distance(M, np.ascontiguousarray(H, dtype=np.uint8).T, num_trials,
                         num_workers, seed, cutoff, None, collect_codewords, num_seeds,
                         island_shots, walkers_per_island, **params)


# ════════════════════════════════════════════════════════════════════════════
# 10. Self-test
# ════════════════════════════════════════════════════════════════════════════

def _bivariate_bicycle(l, m, a_terms, b_terms):
    """Hx = [A | B], Hz = [B^T | A^T] for A, B sums of monomials x^i y^j on
    Z_l x Z_m (Bravyi et al. 2024)."""
    x = np.kron(np.roll(np.eye(l, dtype=np.uint8), 1, axis=1), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), np.roll(np.eye(m, dtype=np.uint8), 1, axis=1))

    def poly(terms):
        P = np.zeros((l * m, l * m), dtype=np.uint8)
        for i, j in terms:
            P ^= (np.linalg.matrix_power(x, i) @ np.linalg.matrix_power(y, j) % 2).astype(np.uint8)
        return P

    A, B = poly(a_terms), poly(b_terms)
    return np.hstack([A, B]), np.hstack([B.T, A.T])


if __name__ == "__main__":
    t0 = time.perf_counter()
    # gross code [[144,12,12]]: A = x^3 + y + y^2, B = y^3 + x + x^2
    Hx, Hz = _bivariate_bicycle(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    res = css_distance(Hx, Hz, num_trials=30000, cutoff=12)
    assert res.z.distance == 12 and res.x.distance == -1, res      # cutoff skips X
    res = css_distance(Hx, Hz, num_trials=30000, cutoff=12, basis="x")
    assert res.x.distance == 12
    print(f"gross: d = ({res.x.distance}, {res.distance})  [{time.perf_counter()-t0:.1f}s]")
    # enumeration: no cutoff, collect the weight-12 logicals that were hit
    res = css_distance(Hx, Hz, num_trials=20000, seed=0, basis="z")
    z = res.z
    assert z.distance == 12 and (z.codewords.sum(axis=1) == 12).all()
    assert not (Hx @ z.codewords.T % 2).any()
    print(f"gross Z side: {z.num_codewords} distinct weight-12 logicals, "
          f"{int(z.counts.sum())} hits, {z.shots} shots, {z.wall_time:.2f}s")
    # seed splitting: 4 populations of 5000 shots; merged set and counts
    res4 = css_distance(Hx, Hz, num_trials=20000, seed=0, basis="z", num_seeds=4)
    assert res4.z.distance == 12 and res4.z.seeds == 4
    assert int(res4.z.counts.sum()) == res4.z.counts.sum() and res4.z.num_codewords > 0
    print(f"gross Z side, 4 seeds: {res4.z.num_codewords} distinct weight-12 logicals, "
          f"{res4.z.shots} shots")
    # rolling islands: 8 islands of 2500 shots on 2 workers, all of them run
    res8 = css_distance(Hx, Hz, num_trials=20000, num_workers=2, seed=0, basis="z",
                        island_shots=2500)
    assert res8.z.distance == 12 and res8.z.islands == 8 and res8.z.shots == 20000
    # early stop: with a cutoff the first hit ends the search well inside the budget
    res_c = css_distance(Hx, Hz, num_trials=2_000_000, num_workers=2, seed=0, basis="z",
                         island_shots=50_000, cutoff=12)
    assert res_c.z.distance == 12 and res_c.z.shots < 200_000, res_c.z.shots
    print(f"gross Z side, rolling islands: {res8.z.islands} islands; with cutoff: "
          f"stopped after {res_c.z.shots} of 2,000,000 shots")
    # half-gross [[72,8,6]] through the tuple-contract entry point
    Hx, Hz = _bivariate_bicycle(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    d, shots, wall, w = find_side_distance_fast(Hz, Hx, 20000, 1, 0, 6, True)
    assert d == 6 and int(w.sum()) == 6 and not (Hz @ w % 2).any()
    print(f"half-gross: d_X = {d} in {shots} shots  [{time.perf_counter()-t0:.1f}s total]  self-test OK")
