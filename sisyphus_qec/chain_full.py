"""Full-width search chain (compact=False).

A chain keeps a generator matrix of the searched space in systematic form and
moves between neighbouring information sets by single-column pivot swaps
(Canteaut--Chabaud), batched with the Method of Four Russians. Learned column
weights bias which columns leave and enter the information set and the column
order of periodic restarts. `_run_island` runs one island: several chains in
lockstep in one process.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .gf2 import _bit, _popcount_u64, _rref_m4ri_order
from .params import BIG_BASE, M4RI_K
from .extraction import _greedy_reduce, _readoff, _readoff_hashed, _readoff_p4


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
