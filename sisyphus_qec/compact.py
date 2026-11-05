"""Compact walker for distance_finder: every basis row is stored on the redundant
columns only (the [I | P] systematic form), 2026-11-04.

A walker's basis of ker(check) in reduced form has one row per information-set
column (its pivot) with further 1s only on the r = rank(check) redundant columns.
distance_finder.py stores every row at full width (n bits); here row i is
    codeword_i = e_{piv[i]} + sum_{slot j set in P[i]} e_{slot_col[j]}
with P[i] of r bits (plus the certification words), slot_col[j] the column held
by slot j. Everything a shot does (swap batch, hashed read-off, descent, theta
reinforcement) scales with the row length, so this is n/r times cheaper per shot
and in memory: 4x on a rate-1/2 code, 10x on circuit DEMs (bb756: 207 words per
row instead of 2020). The information set, the codewords found and the learning
are the same as in distance_finder.py; a candidate codeword is (P-part XOR,
set S of rows in the sum), weight = |S| + popcount(P-part).

A pivot swap (row r evicts its pivot lambda, slot j_mu with P[r, j_mu] = 1
enters): every other row with bit j_mu gets P[r] with bit j_mu cleared XORed in
(its bit j_mu stays set and now means lambda), row r is unchanged, piv[r] = mu,
slot_col[j_mu] = lambda. Restarts eliminate the check matrix in the chosen column
order and transpose its free columns into P (no kernel basis is ever built).

Not supported in compact mode: mix_rate, greedy_kicks, lb_hash_p4 (ignored).
"""
import numpy as np
from numba import njit

from .finder import (_popcount_u64, _bit, _rref_m4ri_order, _xor_weight_row,
                             M4RI_K, BIG_BASE)


# ---------------------------------------------------------------- restart
@njit(cache=True)
def _restart_compact(Hp, Hs, perm, k, tab, pivH, colsig, P, piv, slot_col, idx, rw):
    """Fresh compact basis with pivot order `perm`: eliminate H (copy in Hs), slot
    rho = pivot column of H row rho, P[row of free column f] has bit rho iff
    H row rho contains f; certification words = XOR of colsig over the support.
    Returns the rank (number of slots), or -1 if n - rank != P.shape[0]."""
    r_rows = Hp.shape[0]
    nwh = Hp.shape[1]
    m, nwc = P.shape
    ns = nwc - rw
    n_cols = idx.shape[0]
    for i in range(r_rows):
        for w in range(nwh):
            Hs[i, w] = Hp[i, w]
    rank = _rref_m4ri_order(Hs, perm, k, tab, pivH)
    if n_cols - rank != m:
        return -1
    for c in range(n_cols):
        idx[c] = -1
    for rho in range(rank):
        idx[pivH[rho]] = -2
        slot_col[rho] = pivH[rho]
    j = 0
    for c in range(n_cols):
        if idx[c] == -2:
            continue
        idx[c] = j
        piv[j] = c
        for w in range(nwc):
            P[j, w] = np.uint64(0)
        for s in range(ns):
            P[j, rw + s] = colsig[c, s]
        j += 1
    for rho in range(rank):
        q = pivH[rho]
        sw = rho >> 6
        sb = np.uint64(1) << np.uint64(rho & 63)
        for w in range(nwh):
            word = Hs[rho, w]
            base = w << 6
            while word != 0:
                lowbit = word & (np.uint64(0) - word)
                f = base + int(_popcount_u64(lowbit - np.uint64(1)))
                word ^= lowbit
                if f == q:
                    continue
                jj = idx[f]
                P[jj, sw] |= sb
                for s in range(ns):
                    P[jj, rw + s] ^= colsig[q, s]
    return rank


@njit(cache=True)
def _row_stats_compact(P, rw, wt, is_log):
    m, nwc = P.shape
    for i in range(m):
        c = 1
        for w in range(rw):
            c += int(_popcount_u64(P[i, w]))
        wt[i] = c
        nz = False
        for w in range(rw, nwc):
            if P[i, w] != 0:
                nz = True
                break
        is_log[i] = nz


# ---------------------------------------------------------------- swaps
@njit(cache=True)
def _swap_batch_compact(P, piv, slot_col, wt, is_log, theta, rw, s, tourn, eps,
                        mu_list, u_rows, ws_rows, in_ws, tab, uc):
    """s pivot swaps applied as one pass (see distance_finder._swap_batch).
    Entering choice: the set slot of the evicted row whose column has the
    lowest theta (uniform with probability eps). Correction vector of a swap:
    the evicted row with the entering slot bit cleared (rows keep that bit,
    it now stands for the evicted pivot column)."""
    m, nwc = P.shape
    n_ops = 0
    n_ws = 0
    for _j in range(s):
        r = np.random.randint(0, m)
        if np.random.random() >= eps:
            best_th = theta[piv[r]]
            for _t in range(tourn - 1):
                rc = np.random.randint(0, m)
                th = theta[piv[rc]]
                if th > best_th:
                    best_th = th
                    r = rc
        if in_ws[r] == 0:                     # bring row r up to date
            for q in range(n_ops):
                if _bit(P, r, mu_list[q]):
                    for w in range(nwc):
                        P[r, w] ^= u_rows[q, w]
            in_ws[r] = 1
            ws_rows[n_ws] = r
            n_ws += 1
        lam = piv[r]
        cnt = 0
        for w in range(rw):
            cnt += int(_popcount_u64(P[r, w]))
        if cnt < 1:
            continue                          # nothing to swap in
        mu = -1
        if np.random.random() < eps:          # uniform entering slot
            pick = np.random.randint(0, cnt)
            seen = 0
            for w in range(rw):
                word = P[r, w]
                base = w << 6
                while word != 0:
                    lowbit = word & (np.uint64(0) - word)
                    j = base + int(_popcount_u64(lowbit - np.uint64(1)))
                    word ^= lowbit
                    if seen == pick:
                        mu = j
                        break
                    seen += 1
                if mu >= 0:
                    break
        else:                                 # lowest-theta entering slot; ties go to
            best_mu_th = 1e18                 # the lowest COLUMN index, exactly as the
            best_col = 1 << 60                # full-width walker's first-bit scan does.
            for w in range(rw):               # (Slots are numbered in the elimination
                word = P[r, w]                # order, so a first-slot or a random
                base = w << 6                 # tie-break changes the dynamics: on
                while word != 0:              # circuit DEMs the column order is the
                    lowbit = word & (np.uint64(0) - word)   # time order and the
                    j = base + int(_popcount_u64(lowbit - np.uint64(1)))  # earliest-
                    word ^= lowbit            # column rule keeps late faults out of the
                    c = slot_col[j]           # information set; measured 2026-11-04.)
                    th = theta[c]
                    if th < best_mu_th or (th == best_mu_th and c < best_col):
                        best_mu_th = th
                        best_col = c
                        mu = j
        if mu < 0:
            continue
        mu_list[n_ops] = mu
        for w in range(nwc):
            u_rows[n_ops, w] = P[r, w]
        wi = mu >> 6
        bm = np.uint64(1) << np.uint64(mu & 63)
        u_rows[n_ops, wi] ^= bm              # entering bit cleared in the correction
        for t in range(n_ws):                 # eager update of the working set
            i = ws_rows[t]
            if i != r and (P[i, wi] & bm):
                for w in range(nwc):
                    P[i, w] ^= u_rows[n_ops, w]
        piv[r] = slot_col[mu]
        slot_col[mu] = lam
        n_ops += 1
    if n_ops > 0:
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
        for w in range(nwc):
            tab[0, w] = 0
        for j in range(n_u):
            idx = 1 << j
            for w in range(nwc):
                tab[idx, w] = 0
            mu = uc[j]
            ebit = np.uint64(1) << np.uint64(mu & 63)
            tab[idx, mu >> 6] |= ebit
            for q in range(n_ops):
                muq = mu_list[q]
                if (tab[idx, muq >> 6] >> np.uint64(muq & 63)) & np.uint64(1):
                    for w in range(nwc):
                        tab[idx, w] ^= u_rows[q, w]
            tab[idx, mu >> 6] ^= ebit
            for p in range(idx + 1, 2 * idx):
                for w in range(nwc):
                    tab[p, w] = tab[p - idx, w] ^ tab[idx, w]
        for i in range(m):                    # the single pass
            if in_ws[i]:
                continue
            p = 0
            for j in range(n_u):
                p |= _bit(P, i, uc[j]) << j
            if p != 0:
                c = 1
                for w in range(rw):
                    a = P[i, w] ^ tab[p, w]
                    P[i, w] = a
                    c += int(_popcount_u64(a))
                wt[i] = c
                nz = False
                for w in range(rw, nwc):
                    a = P[i, w] ^ tab[p, w]
                    P[i, w] = a
                    if a != 0:
                        nz = True
                is_log[i] = nz
    for t in range(n_ws):                     # refresh the working set
        i = ws_rows[t]
        c = 1
        for w in range(rw):
            c += int(_popcount_u64(P[i, w]))
        wt[i] = c
        nz = False
        for w in range(rw, nwc):
            if P[i, w] != 0:
                nz = True
                break
        is_log[i] = nz
        in_ws[i] = 0


# ---------------------------------------------------------------- read-off
@njit(cache=True)
def _readoff_hashed_compact(P, wt, is_log, rw, r_slots, M, R, L, n_windows,
                            light_idx, light_wt, shot_cw, shot_rows, pair_tmp,
                            keys, bstart, border, win, slot_cnt, wt_cnt):
    """distance_finder._readoff_hashed on the compact basis. Sums of k rows weigh
    k + popcount of the XOR of their P parts. Writes the lightest candidate's
    P part and certification words to shot_cw and its rows to shot_rows;
    returns (weight, number of rows in the candidate, size of the light list)."""
    m, nwc = P.shape
    n_max = wt_cnt.shape[0] - 1
    for c in range(n_max + 1):
        wt_cnt[c] = 0
    n_logical = 0
    min_w = 1 << 40
    min_i = -1
    for i in range(m):
        if not is_log[i]:
            continue
        w0 = wt[i]
        wt_cnt[w0] += 1
        n_logical += 1
        if w0 < min_w:
            min_w = w0
            min_i = i
    shot_nr = 0
    if min_i < 0:
        return min_w, shot_nr, 0
    for w in range(nwc):
        shot_cw[w] = P[min_i, w]
    shot_rows[0] = min_i
    shot_nr = 1
    if n_logical < 2:
        return min_w, shot_nr, 1
    acc = 0
    T = n_max
    for w0 in range(n_max + 1):
        acc += wt_cnt[w0]
        if acc >= M:
            T = w0
            break
    n_light = 0
    for i in range(m):
        if n_light >= M:
            break
        if is_log[i] and wt[i] <= T:
            light_idx[n_light] = i
            light_wt[n_light] = wt[i]
            n_light += 1
    nR = R if R < n_light else n_light
    for a in range(nR):
        best = a
        for b in range(a + 1, n_light):
            if light_wt[b] < light_wt[best]:
                best = b
        if best != a:
            ti = light_idx[a]; tw = light_wt[a]
            light_idx[a] = light_idx[best]; light_wt[a] = light_wt[best]
            light_idx[best] = ti; light_wt[best] = tw
    for a in range(nR):                       # exhaustive pairs and triples among the R lightest
        ia = light_idx[a]
        for b in range(a + 1, nR):
            ib = light_idx[b]
            nz = False
            for w in range(rw, nwc):
                if (P[ia, w] ^ P[ib, w]) != 0:
                    nz = True
                    break
            w2 = 2
            for w in range(rw):
                x = P[ia, w] ^ P[ib, w]
                pair_tmp[w] = x
                w2 += int(_popcount_u64(x))
            if nz and w2 < min_w:
                min_w = w2
                for w in range(nwc):
                    shot_cw[w] = P[ia, w] ^ P[ib, w]
                shot_rows[0] = ia; shot_rows[1] = ib; shot_nr = 2
            for c in range(b + 1, nR):
                ic = light_idx[c]
                if w2 - light_wt[c] + 1 >= min_w:     # |a+b+c| >= |a+b| - |c| + ... (triangle)
                    continue
                nz3 = False
                for w in range(rw, nwc):
                    if (P[ia, w] ^ P[ib, w] ^ P[ic, w]) != 0:
                        nz3 = True
                        break
                if not nz3:
                    continue
                w3 = 3 + _xor_weight_row(pair_tmp, P, ic, rw)
                if w3 < min_w:
                    min_w = w3
                    for w in range(rw):
                        shot_cw[w] = pair_tmp[w] ^ P[ic, w]
                    for w in range(rw, nwc):
                        shot_cw[w] = P[ia, w] ^ P[ib, w] ^ P[ic, w]
                    shot_rows[0] = ia; shot_rows[1] = ib; shot_rows[2] = ic; shot_nr = 3
    if L > r_slots:
        L = r_slots
    if L < 1 or n_windows < 1:
        return min_w, shot_nr, n_light
    # ---- slot densities on a row sample (balanced window slots)
    for j in range(r_slots):
        slot_cnt[j] = 0
    nsamp = 256 if m > 256 else m
    stride = m // nsamp
    for q in range(nsamp):
        i = q * stride
        for w in range(rw):
            word = P[i, w]
            base = w << 6
            while word != 0:
                lowbit = word & (np.uint64(0) - word)
                slot_cnt[base + int(_popcount_u64(lowbit - np.uint64(1)))] += 1
                word ^= lowbit
    lo = (3 * nsamp) // 10
    hi = (7 * nsamp) // 10
    nb = 1 << L
    cap = 4 * (m // nb + 1)
    for _win in range(n_windows):
        k = 0
        tries = 0
        while k < L:
            tries += 1
            j = np.random.randint(0, r_slots)
            if tries < 64 * L and (slot_cnt[j] < lo or slot_cnt[j] > hi):
                continue
            dup = False
            for t in range(k):
                if win[t] == j:
                    dup = True
                    break
            if dup:
                continue
            win[k] = j
            k += 1
        for i in range(m):
            key = 0
            for t in range(L):
                j = win[t]
                if (P[i, j >> 6] >> np.uint64(j & 63)) & np.uint64(1):
                    key |= 1 << t
            keys[i] = key
        for b in range(nb + 1):
            bstart[b] = 0
        for i in range(m):
            bstart[keys[i] + 1] += 1
        for b in range(nb):
            bstart[b + 1] += bstart[b]
        for i in range(m):
            kb = keys[i]
            border[bstart[kb]] = i
            bstart[kb] += 1
        for b in range(nb):                   # pairs within buckets, all rows
            s = 0 if b == 0 else bstart[b - 1]
            e = bstart[b]
            if e - s > cap:
                continue
            for p in range(s, e):
                i = border[p]
                for q in range(p + 1, e):
                    jr = border[q]
                    d = wt[i] - wt[jr]
                    if d < 0:
                        d = -d
                    if d + 2 >= min_w:        # |a+b| >= |wt a - wt b| (pivots included)
                        continue
                    nz = False
                    for w in range(rw, nwc):
                        if (P[i, w] ^ P[jr, w]) != 0:
                            nz = True
                            break
                    if not nz:
                        continue
                    w2 = 2 + _xor_weight_row(P[i], P, jr, rw)
                    if w2 < min_w:
                        min_w = w2
                        for w in range(nwc):
                            shot_cw[w] = P[i, w] ^ P[jr, w]
                        shot_rows[0] = i; shot_rows[1] = jr; shot_nr = 2
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
                wab = 2
                for w in range(rw):
                    x = P[ia, w] ^ P[ib, w]
                    pair_tmp[w] = x
                    wab += int(_popcount_u64(x))
                for p in range(s, e):
                    ic = border[p]
                    if ic == ia or ic == ib:
                        continue
                    if wab - wt[ic] + 2 >= min_w:
                        continue
                    nz3 = False
                    for w in range(rw, nwc):
                        if (P[ia, w] ^ P[ib, w] ^ P[ic, w]) != 0:
                            nz3 = True
                            break
                    if not nz3:
                        continue
                    w3 = 3 + _xor_weight_row(pair_tmp, P, ic, rw)
                    if w3 < min_w:
                        min_w = w3
                        for w in range(rw):
                            shot_cw[w] = pair_tmp[w] ^ P[ic, w]
                        for w in range(rw, nwc):
                            shot_cw[w] = P[ia, w] ^ P[ib, w] ^ P[ic, w]
                        shot_rows[0] = ia; shot_rows[1] = ib; shot_rows[2] = ic; shot_nr = 3
    return min_w, shot_nr, n_light


# ---------------------------------------------------------------- descent
@njit(cache=True)
def _descend_compact(P, wt, rw, cw, S, cnt, cur, max_passes):
    """Coset-leader descent on the candidate (cw, S): add a basis row when the
    weight |S'| + popcount(P part) drops and the certification words stay
    non-zero. S is a bitset over rows (toggled), cnt = |S|. Returns (weight, cnt)."""
    m, nwc = P.shape
    for _p in range(max_passes):
        improved = False
        lim = 2 * cur
        for i in range(m):
            if wt[i] >= lim:
                continue
            inS = (S[i >> 6] >> np.uint64(i & 63)) & np.uint64(1)
            w_new = _xor_weight_row(cw, P, i, rw) + cnt + (-1 if inS else 1)
            if w_new < cur:
                nz = False
                for w in range(rw, nwc):
                    if (cw[w] ^ P[i, w]) != 0:
                        nz = True
                        break
                if nz:
                    for w in range(nwc):
                        cw[w] ^= P[i, w]
                    S[i >> 6] ^= np.uint64(1) << np.uint64(i & 63)
                    cnt += -1 if inS else 1
                    cur = w_new
                    lim = 2 * cur
                    improved = True
        if not improved:
            break
    return cur, cnt


@njit(cache=True)
def _expand_compact(cw, S, ms, piv, slot_col, rw, out, nw):
    """Full packed codeword (nw words) of the candidate (cw P part, S rows)."""
    for w in range(nw):
        out[w] = 0
    for w in range(ms):
        word = S[w]
        base = w << 6
        while word != 0:
            lowbit = word & (np.uint64(0) - word)
            i = base + int(_popcount_u64(lowbit - np.uint64(1)))
            word ^= lowbit
            c = piv[i]
            out[c >> 6] ^= np.uint64(1) << np.uint64(c & 63)
    for w in range(rw):
        word = cw[w]
        base = w << 6
        while word != 0:
            lowbit = word & (np.uint64(0) - word)
            j = base + int(_popcount_u64(lowbit - np.uint64(1)))
            word ^= lowbit
            c = slot_col[j]
            out[c >> 6] ^= np.uint64(1) << np.uint64(c & 63)


@njit(cache=True)
def _reinforce_row_compact(theta, P, i, piv, slot_col, rw, inc):
    theta[piv[i]] += inc
    for w in range(rw):
        word = P[i, w]
        base = w << 6
        while word != 0:
            lowbit = word & (np.uint64(0) - word)
            theta[slot_col[base + int(_popcount_u64(lowbit - np.uint64(1)))]] += inc
            word ^= lowbit


@njit(cache=True)
def _reinforce_rows_compact(P, theta, wt, is_log, piv, slot_col, rw, best_w, margin,
                            eta, decay, n_cols):
    if decay > 0.0:
        for c in range(n_cols):
            theta[c] *= (1.0 - decay)
    thr = best_w + margin
    m = P.shape[0]
    for i in range(m):
        if not is_log[i]:
            continue
        w0 = wt[i]
        if w0 > thr:
            continue
        _reinforce_row_compact(theta, P, i, piv, slot_col, rw, eta * float(thr - w0 + 1))


@njit(cache=True)
def _reinforce_full(theta, cw_full, nw, inc):
    for w in range(nw):
        word = cw_full[w]
        base = w << 6
        while word != 0:
            lowbit = word & (np.uint64(0) - word)
            theta[base + int(_popcount_u64(lowbit - np.uint64(1)))] += inc
            word ^= lowbit


# ---------------------------------------------------------------- island
@njit(cache=True)
def _run_island_compact(Hp, colsig, n_cols, nw, rank, swaps_per_shot, restart_interval,
                        shots, seed, cutoff, etas, decays, margins, tourns, epss,
                        migrate_interval, migrate_boost, cw_max, lb_M, greedy_passes,
                        stall_rounds, kick_rounds, kick_frac, kick_patience,
                        lb_hash_l, lb_hash_windows, lb_R, explorer_restart, theta_prior, stop):
    """distance_finder._run_island on the compact representation. Hp: packed
    check matrix (rows x nw), colsig: per-column signature words (n_cols x ns),
    rank = rank(Hp) = number of slots, m = n_cols - rank rows.
    theta_prior (n_cols floats, all zero = none): initial priorities of the learners,
    e.g. a large value on the support of a known light codeword (warm start, 2026-11-05).
    Returns (best_w, best_cw_full, cw_buf_full[:cnt], first_hit_round, rounds, best_round)."""
    np.random.seed(seed)
    W = etas.shape[0]
    m = n_cols - rank
    ns = colsig.shape[1]
    rw = (rank + 63) // 64
    nwc = rw + ns
    ms = (m + 63) // 64
    big = m * nw > BIG_BASE
    g_passes = (2 if greedy_passes > 2 else greedy_passes) if big else greedy_passes
    greedy_couples = big
    P_all = np.empty((W, m, nwc), dtype=np.uint64)
    piv_all = np.zeros((W, m), dtype=np.int64)
    slot_all = np.zeros((W, max(rank, 1)), dtype=np.int64)
    wt_all = np.zeros((W, m), dtype=np.int64)
    log_all = np.zeros((W, m), dtype=np.bool_)
    theta_all = np.zeros((W, n_cols), dtype=np.float64)
    walker_best = np.full(W, n_cols + 1, dtype=np.int64)
    wbest_t = np.zeros(W, dtype=np.int64)
    wgap = np.zeros(W, dtype=np.int64)
    wbest_cw = np.zeros((W, nw), dtype=np.uint64)
    kick_now = np.zeros(W, dtype=np.uint8)
    perm = np.arange(n_cols)
    perm_h = np.empty(n_cols, dtype=np.int64)
    perm_key = np.empty(n_cols, dtype=np.float64)
    Hs = np.empty(Hp.shape, dtype=np.uint64)
    pivH = np.empty(Hp.shape[0], dtype=np.int64)
    h_idx = np.empty(n_cols, dtype=np.int64)
    tabw = max(nwc, Hp.shape[1])
    tab = np.zeros((1 << max(swaps_per_shot, M4RI_K), tabw), dtype=np.uint64)
    mu_list = np.empty(swaps_per_shot, dtype=np.int64)
    uc = np.empty(swaps_per_shot, dtype=np.int64)
    u_rows = np.empty((swaps_per_shot, nwc), dtype=np.uint64)
    ws_rows = np.empty(swaps_per_shot, dtype=np.int64)
    in_ws = np.zeros(m, dtype=np.uint8)
    light_idx = np.empty(max(lb_M, 1), dtype=np.int64)
    light_wt = np.empty(max(lb_M, 1), dtype=np.int64)
    shot_cw = np.zeros(nwc, dtype=np.uint64)
    shot_rows = np.zeros(3, dtype=np.int64)
    shot_S = np.zeros(ms, dtype=np.uint64)
    pair_tmp = np.zeros(nwc, dtype=np.uint64)
    hL = lb_hash_l if lb_hash_l > 0 else 0
    if lb_hash_l > 0:
        while (1 << hL) < m and hL < 20:
            hL += 1
    hLa = hL if hL > 0 else 1
    h_keys = np.empty(m, dtype=np.int64)
    h_bstart = np.empty((1 << hLa) + 1, dtype=np.int64)
    h_border = np.empty(m, dtype=np.int64)
    h_win = np.empty(hLa, dtype=np.int64)
    h_slot_cnt = np.zeros(max(rank, 1), dtype=np.int64)
    h_wt_cnt = np.zeros(n_cols + 2, dtype=np.int64)
    R_eff = lb_R if lb_hash_l > 0 else lb_M
    full_tmp = np.zeros(nw, dtype=np.uint64)

    best_w = n_cols + 1
    best_round = -1
    best_cw = np.zeros(nw, dtype=np.uint64)
    cw_buf = np.zeros((cw_max, nw), dtype=np.uint64)
    cw_cnt = 0
    first_hit = -1
    ep_w = n_cols + 1
    ep_cw = np.zeros(nw, dtype=np.uint64)
    ep_t = 0
    reseed = False
    er = explorer_restart if explorer_restart > 0 else restart_interval
    use_prior = theta_prior.shape[0] == n_cols
    if use_prior:
        for w in range(W):
            if epss[w] < 0.9:
                for c in range(n_cols):
                    theta_all[w, c] = theta_prior[c]

    for t in range(shots):
        if stop[0] != 0:
            return best_w, best_cw, cw_buf[:cw_cnt], first_hit, t, best_round
        if reseed:
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
                if epss[w] >= 0.9:
                    np.random.shuffle(perm)
                    for c in range(n_cols):
                        perm_h[c] = perm[c]
                else:
                    tmax = 0.0
                    for c in range(n_cols):
                        if theta_all[w, c] > tmax:
                            tmax = theta_all[w, c]
                    scale = 0.5 * tmax + 1e-9
                    for c in range(n_cols):
                        perm_key[c] = theta_all[w, c] + np.random.random() * scale
                    perm = np.argsort(perm_key)
                    for c in range(n_cols):           # H's pivots leave the information set
                        perm_h[c] = perm[n_cols - 1 - c]
                rk = _restart_compact(Hp, Hs, perm_h, M4RI_K, tab, pivH, colsig, P_all[w],
                                      piv_all[w], slot_all[w], h_idx, rw)
                if rk != rank:
                    return -1, best_cw, cw_buf[:cw_cnt], first_hit, t, best_round   # inconsistent rank
                _row_stats_compact(P_all[w], rw, wt_all[w], log_all[w])
            else:
                _swap_batch_compact(P_all[w], piv_all[w], slot_all[w], wt_all[w], log_all[w],
                                    theta_all[w], rw, swaps_per_shot, tourns[w], epss[w],
                                    mu_list, u_rows, ws_rows, in_ws, tab, uc)
            min_w, nr, _nl = _readoff_hashed_compact(P_all[w], wt_all[w], log_all[w], rw, rank,
                                                     lb_M, R_eff, hL, lb_hash_windows,
                                                     light_idx, light_wt, shot_cw, shot_rows,
                                                     pair_tmp, h_keys, h_bstart, h_border,
                                                     h_win, h_slot_cnt, h_wt_cnt)
            have = min_w <= n_cols
            if have:
                for q in range(ms):
                    shot_S[q] = 0
                for q in range(nr):
                    i = shot_rows[q]
                    shot_S[i >> 6] |= np.uint64(1) << np.uint64(i & 63)
                cnt = nr
                if greedy_couples and g_passes > 0:
                    min_w, cnt = _descend_compact(P_all[w], wt_all[w], rw, shot_cw, shot_S,
                                                  cnt, min_w, g_passes)
            if have and min_w < walker_best[w]:
                walker_best[w] = min_w
                if t - wbest_t[w] > wgap[w]:
                    wgap[w] = t - wbest_t[w]
                wbest_t[w] = t
                _expand_compact(shot_cw, shot_S, ms, piv_all[w], slot_all[w], rw, wbest_cw[w], nw)
            _reinforce_rows_compact(P_all[w], theta_all[w], wt_all[w], log_all[w], piv_all[w],
                                    slot_all[w], rw, walker_best[w], margins[w], etas[w],
                                    decays[w], n_cols)
            if have and min_w <= walker_best[w] + margins[w]:
                _expand_compact(shot_cw, shot_S, ms, piv_all[w], slot_all[w], rw, full_tmp, nw)
                _reinforce_full(theta_all[w], full_tmp, nw,
                                etas[w] * float(walker_best[w] + margins[w] - min_w + 1))
            if have and (not greedy_couples) and g_passes > 0:
                min_w, cnt = _descend_compact(P_all[w], wt_all[w], rw, shot_cw, shot_S,
                                              cnt, min_w, g_passes)
            if have and min_w < ep_w:
                ep_w = min_w
                ep_t = t
                _expand_compact(shot_cw, shot_S, ms, piv_all[w], slot_all[w], rw, ep_cw, nw)
            if have and min_w < best_w:
                best_w = min_w
                best_round = t
                _expand_compact(shot_cw, shot_S, ms, piv_all[w], slot_all[w], rw, best_cw, nw)
                cw_cnt = 0
            if have and min_w == best_w and cw_cnt < cw_max:
                _expand_compact(shot_cw, shot_S, ms, piv_all[w], slot_all[w], rw, cw_buf[cw_cnt], nw)
                cw_cnt += 1
        if kick_rounds > 0:
            for w in range(W):
                if epss[w] >= 0.9 or t - wbest_t[w] < kick_rounds:
                    continue
                if kick_patience > 0.0 and t - wbest_t[w] < kick_patience * wgap[w]:
                    continue
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
            return best_w, best_cw, cw_buf[:cw_cnt], first_hit, t + 1, best_round
        reseed = stall_rounds > 0 and t - ep_t >= stall_rounds
        if migrate_boost > 0.0 and ep_w <= n_cols and not reseed \
                and (t % migrate_interval) == (migrate_interval - 1):
            for w in range(W):
                if epss[w] >= 0.9:
                    continue
                _reinforce_full(theta_all[w], ep_cw, nw, migrate_boost)
    return best_w, best_cw, cw_buf[:cw_cnt], first_hit, shots, best_round
