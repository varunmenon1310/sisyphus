"""Default search parameters and their validation.

Every default was chosen by equal-wall-clock comparisons on small, medium
and large codes and on circuit-level detector error models; see README.md.
Each can be overridden per call as a keyword argument.
"""
from __future__ import annotations

import numpy as np


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
REINFORCE_FRESH = 0       # 1: a learner rewards only light rows that the last swap batch
                          # changed (or a restart built), not every light row still in the
                          # basis; that repeated reward is a positive feedback (columns of a
                          # light row stay redundant, so the row stays light and is rewarded
                          # again) that collapses the walker onto one local optimum
SWAP_BATCHES = 1          # swap batches (of SWAPS_PER_SHOT swaps each) per read-off (compact)
SLOT_SAMPLE = 256         # rows sampled for the slot densities of the hashed read-off
SLOT_RECOUNT = 1          # hashed read-off: the slot densities that pick balanced window
                          # slots are re-sampled every SLOT_RECOUNT shots and after a restart.
                          # Keep 1: a swap XORs a correction into about half of the rows, so
                          # the densities are re-randomized every shot; recounting every 32
                          # shots (2026-11-21 test) saved 9% per shot but picked unbalanced
                          # slots and needed 2-3x more shots on the m630 end window
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
COMPACT = 1               # 1: walkers store each basis row on the r redundant columns
                          # only (chain_compact.py, 2026-11-04): same search,
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
COL_WINDOW = 0.67         # column-window walker: when nonzero every compact island runs a
                          # window walker (a 5th walker, or in place of the uniform explorer,
                          # see COL_WINDOW_EXTRA). At each of its restarts
                          # it puts a random window of consecutive columns (wrapping) first
                          # in the elimination of H, so they leave the information set, and
                          # its uniform swaps never move a window column back in. Light words
                          # concentrated in a stretch of the column order (block-structured
                          # codes, time-ordered circuits) then need far fewer of their other
                          # columns outside the information set. >= 1: window size in columns;
                          # 0 < COL_WINDOW < 1: a fraction of rank(H); < 0: a random fraction
                          # in [0.33, 0.78] per window; 0: off.
COL_WINDOW_EXTRA = 1      # 1: the window walker is a 5th walker of the island (the uniform
                          # explorer stays); 0: it replaces the uniform explorer.
COL_WINDOW_ROUNDS = 50    # rounds per window of the window walker (a fresh window each time)
SCHEDULER = "adaptive"    # island scheduler (2026-11-21). 'static': islands planned in advance,
                          # a fraction WINDOW_FRAC on detector windows drawn uniformly (level,
                          # then window). 'adaptive': every sub-problem (the full instance and
                          # each class of equivalent detector windows: windows whose check
                          # matrices and signature words coincide in local coordinates, i.e.
                          # time-translates in a periodic circuit) is an arm; each arm gets
                          # one island, then islands go to arm a with probability
                          # proportional to exp(-(best_a - best) / ARM_TAU) / sqrt(1 + s_a),
                          # s_a = the arm's islands since its best improved, plus its
                          # running islands. Without windows both are the same.
PERIOD_WINDOWS = 1        # adaptive scheduler on circuit-level models: if the detectors have a
                          # period T (a memory circuit with T detectors per round, found as the
                          # smallest shift that leaves the middle half of the check matrix and
                          # signature words invariant), use windows of 1, 2, 3, 4, 6, 8, ...
                          # whole rounds, aligned to round boundaries, plus the windows at
                          # both time boundaries, and merge equivalent bulk windows into one
                          # sub-problem; 0 = the dyadic windows of WINDOWS
ARM_COST_POWER = 1.0      # adaptive scheduler: arm weights are divided by (mean island wall time
                          # of the arm)^ARM_COST_POWER (1 = equal time for equally good arms)
ARM_TAU = 1.0             # temperature (in weight units) of the adaptive scheduler
FULL_STALL_STOP = 3000    # adaptive scheduler with windows: stall stop of the full-instance
                          # islands (on circuit-level models they are rarely competitive and
                          # one of them would otherwise hold a worker for hours)
RETURN_CODEWORDS = 256    # adaptive scheduler: codewords an island sends back (only if it
                          # reached the best weight known when it started); 0 = all
FIRST_PASS_STALL = 1000   # adaptive scheduler: the first island of every window is a screening
                          # run and stops after this many rounds without improvement (and at
                          # least as long as it took to reach its best); later islands of the
                          # window run their full budget. 0 = off.
WINDOW_ISLAND_MAX = 40000 # adaptive scheduler: at most this many rounds per window island (large
                          # windows would otherwise get 100k+ rounds each and hold workers for
                          # hours during the first pass over the sub-problems)
STALL_STOP = 0            # an island returns once its best weight has not improved for
                          # STALL_STOP rounds and for as long as it took to reach it
                          # (0 = run the full island budget)

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
             window_min=WINDOW_MIN, col_window=COL_WINDOW,
             col_window_rounds=COL_WINDOW_ROUNDS, col_window_extra=COL_WINDOW_EXTRA,
             scheduler=SCHEDULER, arm_tau=ARM_TAU, arm_cost_power=ARM_COST_POWER,
             stall_stop=STALL_STOP,
             period_windows=PERIOD_WINDOWS, full_stall_stop=FULL_STALL_STOP,
             reinforce_fresh=REINFORCE_FRESH, swap_batches=SWAP_BATCHES,
             window_island_max=WINDOW_ISLAND_MAX, first_pass_stall=FIRST_PASS_STALL,
             return_codewords=RETURN_CODEWORDS)
    unknown = set(overrides) - set(p)
    if unknown:
        raise TypeError(f"unknown finder parameters: {sorted(unknown)}")
    p.update(overrides)
    return p

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
