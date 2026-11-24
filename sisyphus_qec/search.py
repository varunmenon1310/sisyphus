"""Parallel search and the public entry points.

Islands (independent groups of chains, one per process) are scheduled on a
process pool; each runs for a fixed iteration budget and is replaced by a
freshly seeded island until the total budget is spent. The first island to
reach the cutoff weight stops all others. Shared read-only inputs are inherited
by the workers through the pool initializer (inherited under fork, pickled once
per worker under spawn) instead of being pickled per island.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np

from .gf2 import _kernel_packed, pack_gf2_matrix_u64, unpack_gf2_matrix_u64
from .params import (
    COMPACT, ISLAND_SHOTS, ISLAND_W, RESTART_INTERVAL, WINDOW_AUTO_LEVELS,
    WINDOW_AUTO_RATIO, _params)
from .results import CssDistanceResult, SideResult
from .chain_full import _run_island
from .windows import (_SparseChecks, _detector_period, _detector_windows, _expand_window_buf,
                      _expand_window_cw, _period_window_specs)
from .setup import (
    _collect_results, _island_args, _island_args_compact, _island_walkers, _side_setup,
    _side_setup_compact)


_STOP = None   # this process's view of the search's shared stop flag
_SHARED_C = None  # (Hp, colsig) of the current compact search (see _SHARED)
_SHARED_WIN = None  # detector-window sub-instances of the current search (WINDOWS)
_SHARED_NW = 0
_SHARED_NCOLS = 0
_SHARED = None # (A0, Hp, colsig) of the current search, handed to the workers by the
               # pool initializer so that the basis is not pickled once per island
               # (2026-11-04; a circuit-DEM basis is 80-200 MB)

def _raise_stop(stop):
    """Timer target of side_distance(time_limit=...): stop every island."""
    stop[0] = 1

def _mp_context():
    """Start method of the worker pools. SISYPHUS_MP_START overrides it. The default is
    'fork' on Linux (zero-copy inheritance of the shared arrays; Python 3.14 changes the
    Linux default to forkserver, so it is requested explicitly) and 'spawn' elsewhere:
    on macOS fork is unsafe once system frameworks have started threads, and Python's
    own default there is spawn."""
    method = os.environ.get("SISYPHUS_MP_START") or \
        ("fork" if os.uname().sysname == "Linux" else "spawn")
    return mp.get_context(method)

def _pool_init(raw, shared=None):
    """Worker initializer: the stop flag and the read-only search inputs. Passing the
    inputs here (not through module globals set before the pool was created) makes the
    workers correct under every start method: fork inherits initargs without pickling,
    spawn/forkserver pickle them once per worker, never once per island."""
    global _STOP, _SHARED, _SHARED_C, _SHARED_WIN, _SHARED_NW, _SHARED_NCOLS
    _STOP = np.frombuffer(raw, dtype=np.int64)
    if shared is not None:
        _SHARED, _SHARED_C, _SHARED_WIN, _SHARED_NW, _SHARED_NCOLS = shared

def _make_pool(num_workers, shared=None):
    """Worker processes sharing one int64 stop flag. Returns (pool, flag);
    pool is None when the search runs in this process. `shared` = the module
    globals (_SHARED, _SHARED_C, _SHARED_WIN, _SHARED_NW, _SHARED_NCOLS) the
    workers need, installed in each worker by _pool_init."""
    if num_workers <= 1:
        return None, np.zeros(1, dtype=np.int64)
    ctx = _mp_context()
    raw = ctx.RawArray("q", 1)
    pool = ctx.Pool(processes=num_workers, initializer=_pool_init, initargs=(raw, shared))
    return pool, np.frombuffer(raw, dtype=np.int64)

def _island_worker(a):
    """Pool entry point: one island. Returns (index, best_w, best_cw, cw_buf, rounds_done).
    A task whose A0 is None uses the arrays inherited from the parent (_SHARED);
    a 3-tuple (i, "compact", args) runs the compact walker with _SHARED_C."""
    stop = _STOP if _STOP is not None else np.zeros(1, dtype=np.int64)
    if a[1] == "compact":                             # (i, "compact", args[, hint, p])
        i, _tag, args = a[:3]
        from .chain_compact import _run_island_compact
        Hp, colsig = _SHARED_C
        bw, bcw, cwbuf, _fh, rounds_done, best_round = _run_island_compact(Hp, colsig, *args, stop)
        if len(a) > 4:
            cwbuf = _trim_buffer(cwbuf, bw, a[3], a[4])
        return i, bw, bcw, cwbuf, rounds_done, best_round
    if a[1] == "cwin":                                # (i, "cwin", window index, args)
        i, _tag, widx, args = a
        from .chain_compact import _run_island_compact
        cols, Hp_s, cs = _SHARED_WIN[widx][:3]
        nw = _SHARED_NW
        bw, bcw, cwbuf, _fh, rounds_done, best_round = _run_island_compact(Hp_s, cs, *args, stop)
        if bw > cols.size:                            # nothing found in the window
            return i, _SHARED_NCOLS + 1, np.zeros(nw, np.uint64), np.zeros((0, nw), np.uint64), \
                rounds_done, best_round
        bcw_f = _expand_window_cw(bcw, cols, nw)
        return i, bw, bcw_f, _expand_window_buf(cwbuf, cols, nw), rounds_done, best_round
    if a[1] == "lwin":                                # (i, "lwin", spec index, rounds, seed, cutoff, p, W)
        return _lazy_window_island(a, stop)
    i, args = a
    if args[0] is None:
        args = list(args)
        args[0], args[-2], args[-1] = _SHARED
        args = tuple(args)
    bw, bcw, cwbuf, _fh, rounds_done = _run_island(*args, stop)
    return i, bw, bcw, cwbuf, rounds_done

_WIN_CACHE = {}
FULL_ISLAND_BYTES = 512 << 20   # see _search_side_compact (limits on full-problem islands)
WIN_CACHE_BYTES = 256 << 20     # per worker, see _lazy_window_island

def _lazy_window_island(a, stop):
    """One island on a lazy detector window: the sub-instance is cut out of the shared
    sparse check matrix by this worker the first time it gets the window (then cached).
    Returns the usual result tuple plus a dict (rows, rank) of the window; a window
    without a possible logical returns 0 rounds and dead=1."""
    i, _tag, widx, r_i, seed, cutoff, p, W = a[:8]
    hint = a[8] if len(a) > 8 else (1 << 40)          # best weight known when submitted
    from .chain_compact import _run_island_compact
    _lazy, sc, specs = _SHARED_WIN
    nw = _SHARED_NW
    if widx in _WIN_CACHE:
        e = _WIN_CACHE.pop(widx)                      # move to the most recent end
    else:
        sp = specs[widx]
        e = sc.build(sp["a"], sp["b"], sp["cols"], sp["level"], p["m4ri_k"])
    _WIN_CACHE[widx] = e
    # bound the cache by size (a large window's sub-instance is tens of MB, and a worker
    # visits many windows): drop the least recently used entries beyond WIN_CACHE_BYTES
    size = sum(0 if v is None else v[0].nbytes + v[1].nbytes + v[2].nbytes
               for v in _WIN_CACHE.values())
    while size > WIN_CACHE_BYTES and len(_WIN_CACHE) > 1:
        k0 = next(iter(_WIN_CACHE))
        v = _WIN_CACHE.pop(k0)
        size -= 0 if v is None else v[0].nbytes + v[1].nbytes + v[2].nbytes
    empty = (i, _SHARED_NCOLS + 1, np.zeros(nw, np.uint64), np.zeros((0, nw), np.uint64), 0, -1)
    if e is None:
        return empty + ({"dead": 1},)
    cols, Hp_s, cs, n_s, nw_s, rk_s = e[:6]
    r_w = int(p["window_rounds"]) * max(1.0, (n_s - rk_s) / 1000.0)
    if int(p.get("window_island_max") or 0) > 0:
        r_w = min(r_w, int(p["window_island_max"]))
    cut_w = cutoff if n_s >= cutoff else 0
    args = _island_args_compact(n_s, nw_s, rk_s, int(min(r_i, r_w)), seed, cut_w, p, W)
    bw, bcw, cwbuf, _fh, rounds_done, best_round = _run_island_compact(Hp_s, cs, *args, stop)
    info = {"rows": n_s - rk_s, "rank": rk_s}
    if bw > cols.size:
        return (i, _SHARED_NCOLS + 1, np.zeros(nw, np.uint64), np.zeros((0, nw), np.uint64),
                rounds_done, best_round, info)
    cwbuf = _trim_buffer(cwbuf, bw, hint, p)
    return (i, bw, _expand_window_cw(bcw, cols, nw), _expand_window_buf(cwbuf, cols, nw),
            rounds_done, best_round, info)

def _trim_buffer(cwbuf, bw, hint, p):
    """Codewords an island sends back to the parent: none if its best weight is worse than
    the best known when it was submitted, else at most RETURN_CODEWORDS (thousands of
    islands finish in a long search; full-width buffers of up to max_codewords words each
    would pile up in the parent faster than it can take them)."""
    if bw > hint:
        return cwbuf[:0]
    cap = int(p.get("return_codewords") or 0)
    return cwbuf[:cap] if cap > 0 else cwbuf

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
        # the workers received _SHARED in their initializer: send the basis as None
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

class _Arm:
    """One sub-problem of the adaptive island scheduler: the full instance, or one class of
    translation-equivalent detector windows (members = indices into _SHARED_WIN)."""
    __slots__ = ("name", "members", "level", "n_cols", "n_rows", "n", "inflight", "best",
                 "stale", "shots", "first_best", "time")

    def __init__(self, name, members, level, n_cols, n_rows):
        self.name, self.members, self.level = name, members, level
        self.n_cols, self.n_rows = n_cols, n_rows
        self.n = 0; self.inflight = 0; self.best = 1 << 40; self.stale = 0; self.shots = 0
        self.first_best = -1; self.time = 0.0


def _make_arms(n_cols, rank, wins, period=0, counts=None):
    arms = [_Arm("full", None, 0, n_cols, n_cols - rank)]
    if wins and wins[0] == "lazy":
        for q, sp in enumerate(wins[2]):
            arms.append(_Arm(f"R{sp['level'] / 4:g}@{sp['a']}:{sp['cols'].size}"
                             + (f"(x{sp['count']})" if sp["count"] > 1 else ""),
                             [q], sp["level"], int(sp["cols"].size), -1))
        return arms
    if wins:
        classes = {}
        for q, w in enumerate(wins):
            classes.setdefault(w[9], []).append(q)
        for key, members in classes.items():
            w = wins[members[0]]
            k = sum(counts[q] for q in members) if counts else len(members)
            tag = f"R{w[6] / 2:g}" if period else f"L{w[6]}"
            arms.append(_Arm(f"{tag}@{w[7]}:{w[3]}x{w[5]}" + (f"(x{k})" if k > 1 else ""),
                             members, w[6], w[3], w[3] - w[5]))
    return arms


def _choose_arm(arms, rng, tau, order, cost_power=1.0, cap_full=True):
    """Next arm: every arm once (in `order`), then sample with probability proportional to
    exp(-(best - global best) / tau) / sqrt(1 + stale + inflight) / cost^cost_power, where
    stale counts the arm's islands since its best last improved and cost is the arm's mean
    island wall time (so equally promising arms get equal time, not equal island counts:
    an island on a large window costs many times one on a small window)."""
    for a in order:
        if arms[a].n + arms[a].inflight == 0 and arms[a].n_rows != 0:
            return a
    bstar = min(x.best for x in arms)
    costs = np.array([x.time / x.n if x.n > 0 else 0.0 for x in arms])
    ref = np.median(costs[costs > 0]) if (costs > 0).any() else 1.0
    costs = np.where(costs > 0, costs, ref) / ref
    # cap_full: at most one island on the full problem at a time (its bases are the
    # largest, about 1 GB per island on a 10k x 150k DEM, and on multi-round circuits it
    # rarely wins once windows exist)
    w = np.array([0.0 if x.n_rows == 0 or (cap_full and x.members is None and x.inflight > 0
                                           and len(arms) > 1)
                  else np.exp(-max(0, x.best - bstar) / tau) / np.sqrt(1.0 + x.stale + x.inflight)
                  / max(c, 1e-3) ** cost_power
                  for x, c in zip(arms, costs)])
    if w.sum() <= 0:
        w = np.array([0.0 if x.n_rows == 0 else 1.0 for x in arms])
    return int(rng.choice(len(arms), p=w / w.sum()))


def _search_side_compact(Hp, colsig, n_cols, nw, rank, num_workers, num_trials, island_shots,
                         num_seeds, W, cutoff, rng, collect, pool, stop, p, theta_prior=None):
    """_search_side for the compact walker (Hp, colsig shared with the workers
    through _SHARED_C, set by the caller before the pool was created).
    p['scheduler'] == 'adaptive' (default): islands are assigned on the fly to sub-problems
    (the full instance and the classes of equivalent detector windows) by _choose_arm;
    'static': the pre-planned interleaving of full and window islands (window_frac)."""
    m = n_cols - rank
    if m == 0:
        return SideResult(distance=n_cols + 1, codewords=np.zeros((0, n_cols), np.uint8),
                          islands=0)
    W_eff = _island_walkers(p, W)
    rounds, n_islands, last = _island_plan(num_trials, num_workers, island_shots, num_seeds, W)
    wins = _SHARED_WIN if (p.get("windows", 0) and _SHARED_WIN) else None
    adaptive = p.get("scheduler", "adaptive") == "adaptive"
    if wins and wins[0] == "lazy":
        adaptive = True

    lazy = bool(wins) and wins[0] == "lazy"
    # Limits on full-problem islands (one at a time, stall stop) apply on multi-round
    # circuits (a round period was found) or when one full island would be large. On a
    # one-round circuit the light solutions span all of its detectors and the full problem
    # is where they are found.
    full_bytes = (n_cols - rank) * ((rank + 63) // 64 + colsig.shape[1]) * 8 * W_eff
    limit_full = bool(p.get("_win_period")) or full_bytes > FULL_ISLAND_BYTES

    def task_for(i, widx, r_i, first=False):
        if lazy and widx is not None and first and int(p.get("first_pass_stall") or 0) > 0:
            pw = dict(p, stall_stop=int(p["first_pass_stall"]))
            return (i, "lwin", widx, int(r_i), int(rng.integers(2**31)), int(cutoff), pw, W)
        if widx is None:
            pf = p
            if adaptive and wins and limit_full and int(p.get("full_stall_stop") or 0) > 0:
                pf = dict(p, stall_stop=int(p["full_stall_stop"]))
            return (i, "compact", _island_args_compact(n_cols, nw, rank, r_i, int(rng.integers(2**31)),
                                                       cutoff, pf, W, theta_prior=theta_prior))
        if lazy:
            return (i, "lwin", widx, int(r_i), int(rng.integers(2**31)), int(cutoff), p, W)
        cols, _H, _c, n_s, nw_s, rk_s = wins[widx][:6]
        pr = None if theta_prior is None else np.ascontiguousarray(theta_prior[cols])
        # rounds grow with the window's rows (n_s - rank): small windows converge in a
        # few hundred rounds, a code-sized window needs tens of thousands
        r_w = int(p["window_rounds"]) * max(1.0, (n_s - rk_s) / 1000.0)
        # a window with fewer than `cutoff` columns starts at its "nothing found" weight
        # n_s + 1 <= cutoff, which the kernel would take for a cutoff hit: it would raise
        # the shared stop flag in the first round and end the whole search without a hit
        cut_w = cutoff if n_s >= cutoff else 0
        return (i, "cwin", widx, _island_args_compact(
            n_s, nw_s, rk_s, int(min(r_i, r_w)), int(rng.integers(2**31)),
            cut_w, p, W, theta_prior=pr))

    stop[0] = 0
    t0 = time.perf_counter()
    if not adaptive or not wins:
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
                tasks.append(task_for(i, pool_l[int(rng.integers(len(pool_l)))], r_i))
            else:
                tasks.append(task_for(i, None, r_i))
        if pool is None:
            res = []
            for task in tasks:
                if stop[0] != 0:
                    break
                res.append(_run_task_inproc(task, Hp, colsig, stop))
        else:
            res = sorted(pool.imap_unordered(_island_worker, tasks, chunksize=1),
                         key=lambda r: r[0])
        return _collect_results(res, W_eff, n_cols, nw, collect, time.perf_counter() - t0)

    # ---- adaptive scheduler
    arms = _make_arms(n_cols, rank, wins, p.get("_win_period", 0), p.get("_win_counts"))
    # first pass: every sub-problem once, smallest first (cheap islands set the reference
    # weight that the sampling below compares against); the full instance last
    order = sorted(range(len(arms)), key=lambda q: (arms[q].members is None, arms[q].n_cols,
                                                    float(rng.random())))
    tau = float(p.get("arm_tau", 1.0))
    budget = int(num_trials)
    committed = 0
    res = []
    arm_of = {}
    planned = {}
    started = {}
    history = []          # (time, weight, shots so far, arm) at each improvement of the best
    done_shots = 0
    gbest = 1 << 40
    stored = 0                                        # codewords kept from islands at gbest
    store_cap = 4 * max(1, int(p.get("max_codewords") or 2000))
    import queue as _queue
    _EMPTY_CW = np.zeros(0, np.uint64)
    _EMPTY_BUF = np.zeros((0, nw), np.uint64)
    results_q = _queue.Queue()
    inflight = 0
    i = 0

    def submit():
        nonlocal committed, inflight, i, done_shots
        a = _choose_arm(arms, rng, tau, order, float(p.get("arm_cost_power", 1.0)), limit_full)
        arm = arms[a]
        widx = None if arm.members is None else arm.members[int(rng.integers(len(arm.members)))]
        r_i = rounds
        task = task_for(i, widx, r_i, first=(arm.n + arm.inflight == 0))
        if task[1] == "lwin":
            task = task + (int(gbest),)
        elif task[1] == "compact":
            task = task + (int(gbest), p)
        r_task = int(task[3]) if task[1] == "lwin" else int(task[2 if task[1] == "compact" else 3][5])
        committed += r_task * W_eff
        planned[i] = r_task
        arm_of[i] = a
        started[i] = (time.perf_counter() - t0, done_shots)
        arm.inflight += 1
        inflight += 1
        if pool is None:
            results_q.put(_run_task_inproc(task, Hp, colsig, stop))
        else:
            pool.apply_async(_island_worker, (task,), callback=results_q.put,
                             error_callback=lambda e: results_q.put(e))
        i += 1

    while True:
        while (stop[0] == 0 and committed < budget
               and inflight < (num_workers if pool is not None else 1)):
            submit()
        if inflight == 0:
            break
        r = results_q.get()
        if isinstance(r, BaseException):
            raise r
        inflight -= 1
        a = arm_of[r[0]]
        arm = arms[a]
        arm.inflight -= 1
        arm.n += 1
        arm.shots += int(r[4]) * W_eff
        if len(r) > 6 and isinstance(r[6], dict):
            if r[6].get("dead"):
                arm.n_rows = 0
            elif "rows" in r[6]:
                arm.n_rows = int(r[6]["rows"])
        committed -= (planned.pop(r[0]) - int(r[4])) * W_eff   # refund an early return
        now = time.perf_counter() - t0
        t_start, sh_start = started.pop(r[0])
        arm.time += now - t_start
        frac = (int(r[5]) + 1) / max(1, int(r[4])) if r[5] >= 0 else 1.0
        done_shots += int(r[4]) * W_eff
        improved = r[1] < gbest and r[1] <= n_cols
        if improved:
            gbest = int(r[1])
            # estimated time and shot count at which the island first reached it: the
            # shots of the islands finished before plus all running islands' progress
            t_hit = t_start + frac * (now - t_start)
            history.append((round(t_hit, 2), gbest, int(sh_start + frac * (done_shots - sh_start)),
                            arm.name))
        if r[1] < arm.best:
            arm.best = int(r[1]); arm.stale = 0
            arm.first_best = len(res)
        else:
            arm.stale += 1
        # keep the stored codewords only for islands at the best weight so far (each buffer
        # holds up to max_codewords full-width words: tens of MB on a large DEM, and a long
        # search runs thousands of islands)
        if r[1] > gbest:                              # only its weight is needed from now on
            r = (r[0], r[1], _EMPTY_CW, _EMPTY_BUF) + tuple(r[4:])
        elif improved:
            stored = 0
            for k_res in range(len(res)):
                x = res[k_res]
                if x[1] > gbest and x[2].shape[0]:
                    res[k_res] = (x[0], x[1], _EMPTY_CW, _EMPTY_BUF) + tuple(x[4:])
        # and at most a few times max_codewords of them in total (when the best weight is
        # the bulk weight of a memory circuit, nearly every island reaches it)
        if r[1] == gbest and r[3].shape[0]:
            room = max(0, store_cap - stored)
            if r[3].shape[0] > room:
                r = r[:3] + (np.ascontiguousarray(r[3][:room]),) + tuple(r[4:])
            stored += r[3].shape[0]
        res.append(r)
    out = _collect_results(sorted([r[:6] for r in res], key=lambda r: r[0]), W_eff, n_cols, nw,
                           collect, time.perf_counter() - t0)
    out.history = history
    out.arms = [dict(name=x.name, rows=x.n_rows, islands=x.n, shots=x.shots, time=round(x.time, 1),
                     best=x.best if x.best < (1 << 40) else -1) for x in arms]
    return out


def _run_task_inproc(task, Hp, colsig, stop):
    """One island in this process (num_workers = 1)."""
    if task[1] in ("cwin", "lwin"):
        return _island_worker(task)
    from .chain_compact import _run_island_compact
    i, _tag, a = task[:3]
    bw, bcw, cwbuf, _fh, rd, br = _run_island_compact(Hp, colsig, *a, stop)
    return (i, bw, bcw, cwbuf, rd, br)


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
                  islands reuse those presets in turn). With col_window != 0
                  and col_window_extra = 1 (the default) every island also
                  runs a column-window walker on top of these.
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
            # levels down to windows of about 64 detectors (at least WINDOW_AUTO_LEVELS, at
            # most 8): the adaptive scheduler drops the levels that do not pay
            lev_auto = WINDOW_AUTO_LEVELS
            if p.get("scheduler", "adaptive") == "adaptive" and m_rows >= 64:
                lev_auto = int(min(8, max(WINDOW_AUTO_LEVELS, np.floor(np.log2(m_rows / 64.0)))))
            n_lev = lev_auto if n_cols >= WINDOW_AUTO_RATIO * m_rows else 0
            p["windows"] = n_lev
        if int(n_lev) > 0:
            T = 0
            if p.get("scheduler", "adaptive") == "adaptive" and p.get("period_windows", 1):
                T = _detector_period(check_H, colsig)
            m_rows = np.asarray(check_H).shape[0]
            if T > 0 and m_rows >= 4 * T:
                # a memory circuit with T detectors per round: windows of whole rounds,
                # equivalent bulk windows merged, sub-instances cut out by the workers
                # (see windows._period_window_specs)
                sc = _SparseChecks(check_H, colsig)
                specs = _period_window_specs(sc, T)
                _SHARED_WIN = ("lazy", sc, specs) if specs else None
                p["_win_period"] = T
            else:
                _SHARED_WIN = _detector_windows(check_H, colsig, int(n_lev),
                                                int(p["window_min"]), p["m4ri_k"]) or None
        if num_workers > 1 and n_cols > rank:
            # load the compiled kernels in this process before the pool is forked:
            # otherwise every worker imports the module and loads the cache itself
            # (2.6 s each, paid by every fresh pool; found 2026-11-04 on small DEMs).
            # Under spawn each worker still loads the cache once per pool.
            # A tiny instance with the same argument types loads the same compiled code
            # (a full-instance warm-up costs one restart: 9 s on a 7k x 110k DEM).
            from .chain_compact import _run_island_compact
            Ht = np.array([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=np.uint8)
            Hpt = pack_gf2_matrix_u64(Ht)
            cst = np.ascontiguousarray(np.full((4, colsig.shape[1]), 1, dtype=np.uint64))
            _run_island_compact(Hpt, cst, *_island_args_compact(4, 1, 2, 1, 0, 0, p, 1,
                                                                theta_prior=None),
                                np.zeros(1, dtype=np.int64))
    else:
        A0, n_cols, nw, Hp, colsig = _side_setup(check_H, dual_H, pair_rows, p["proj_words"],
                                                 int(rng.integers(2**31)), p["m4ri_k"],
                                                 return_h=True)
        _SHARED = (A0, Hp, colsig)
    pool, stop = _make_pool(num_workers, (_SHARED, _SHARED_C, _SHARED_WIN, _SHARED_NW,
                                          _SHARED_NCOLS))
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
