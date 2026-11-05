"""Parallel search and the public entry points.

Islands (independent groups of chains, one per process) are scheduled on a
process pool; each runs for a fixed iteration budget and is replaced by a
freshly seeded island until the total budget is spent. The first island to
reach the cutoff weight stops all others. Shared read-only inputs are inherited
by forked workers through module globals instead of being pickled per island.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np

from .gf2 import _kernel_packed, unpack_gf2_matrix_u64
from .params import (
    COMPACT, ISLAND_SHOTS, ISLAND_W, RESTART_INTERVAL, WINDOW_AUTO_LEVELS,
    WINDOW_AUTO_RATIO, _params)
from .results import CssDistanceResult, SideResult
from .chain_full import _run_island
from .windows import _detector_windows, _expand_window_buf, _expand_window_cw
from .setup import (
    _collect_results, _island_args, _island_args_compact, _side_setup,
    _side_setup_compact)


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
        from .chain_compact import _run_island_compact
        Hp, colsig = _SHARED_C
        bw, bcw, cwbuf, _fh, rounds_done, best_round = _run_island_compact(Hp, colsig, *args, stop)
        return i, bw, bcw, cwbuf, rounds_done, best_round
    if len(a) == 4:                                   # (i, "cwin", window index, args)
        i, _tag, widx, args = a
        from .chain_compact import _run_island_compact
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
        from .chain_compact import _run_island_compact
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

def side_distance(check_H, dual_H, num_trials=20000, num_workers=1, seed=None,
                  cutoff=0, pair_rows=None, collect_codewords=True, num_seeds=None,
                  island_shots=None, walkers_per_island=ISLAND_W, time_limit=None,
                  compact=None, **params):
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
                  the redundant columns, chain_compact.py), False the
                  full-width chain (chain_full.py).
    **params      search knobs, see module constants (greedy_kicks, lb_p, ...).

    Returns a SideResult. Every codeword in it is certified: it satisfies the
    check and pairs non-trivially with an element of ker(dual_H).
    """
    p = _params(**params)
    theta_prior = None      # internal hook: initial column weights of the learning chains
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
            from .chain_compact import _run_island_compact
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
