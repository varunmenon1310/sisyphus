"""Candidate extraction and greedy weight reduction for the full-width chain.

At every iteration the rows of the systematic generator matrix are scanned for
low-weight codewords: single rows (Prange), sums of two or three rows
(Lee--Brickell), with pairs and triples found by a collision search on random
subsets of redundant columns (Stern). Each candidate is then reduced greedily
by adding basis rows that lower its weight.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .gf2 import _popcount_u64, _xor_weight_row


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
