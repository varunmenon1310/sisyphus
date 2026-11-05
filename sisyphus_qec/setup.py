"""Per-side setup of a search: packed check matrix, verification parity words,
starting basis, and the argument tuples of the island kernels.
"""
from __future__ import annotations

import numpy as np

from .gf2 import (
    _kernel_packed, _nullspace_packed_u64, _pairing_words, _random_combinations,
    _rref_m4ri_order, _sig_transpose, pack_gf2_matrix_u64, unpack_gf2_matrix_u64)
from .params import ISLAND_W, RESTART_INTERVAL, _portfolio_arrays
from .results import SideResult


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
