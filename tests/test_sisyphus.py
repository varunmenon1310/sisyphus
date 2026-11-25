"""Basic correctness tests: python -m pytest tests/"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from codes import bb72_code, gross_code  # noqa: E402

from sisyphus_qec import css_distance, side_distance  # noqa: E402


def _is_logical(c, check, dual):
    """c in ker(check) and not in rowspan(dual) (brute-force rank test)."""
    if (check @ c % 2).any():
        return False
    M = np.vstack([dual, c]).astype(np.uint8) % 2
    return _rank(M) > _rank(dual)


def _rank(M):
    M = M.copy() % 2
    r = 0
    for col in range(M.shape[1]):
        piv = [i for i in range(r, M.shape[0]) if M[i, col]]
        if not piv:
            continue
        M[[r, piv[0]]] = M[[piv[0], r]]
        for i in range(M.shape[0]):
            if i != r and M[i, col]:
                M[i] ^= M[r]
        r += 1
    return r


def _brute_force_distance(check, dual, n):
    for w in range(1, n + 1):
        for supp in itertools.combinations(range(n), w):
            c = np.zeros(n, dtype=np.uint8)
            c[list(supp)] = 1
            if _is_logical(c, check, dual):
                return w
    return -1


def test_bb72_distance_6():
    Hx, Hz = bb72_code()
    res = css_distance(Hx, Hz, num_trials=50_000, num_workers=1, seed=0)
    assert res.x.distance == 6 and res.z.distance == 6
    for side, check, dual in ((res.x, Hz, Hx), (res.z, Hx, Hz)):
        assert (side.codewords.sum(axis=1) == 6).all()
        for c in side.codewords[:5]:
            assert _is_logical(c, check, dual)


def test_gross_distance_12_with_cutoff():
    Hx, Hz = gross_code()
    res = css_distance(Hx, Hz, num_trials=200_000, num_workers=2, seed=0, cutoff=12, basis="z")
    assert res.z.distance == 12


@pytest.mark.parametrize("seed", range(6))
def test_random_small_codes_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 12
    # random CSS code: Hx random, Hz drawn from the kernel of Hx
    Hx = rng.integers(0, 2, size=(3, n), dtype=np.uint8)
    ker = _kernel(Hx)
    Hz = (rng.integers(0, 2, size=(3, ker.shape[0]), dtype=np.uint8) @ ker % 2).astype(np.uint8)
    if _rank(np.vstack([Hx, Hz])) == n:
        pytest.skip("no logical qubits")
    exact = _brute_force_distance(Hx, Hz, n)          # Z side: ker(Hx) minus rowspan(Hz)
    res = side_distance(Hx, Hz, num_trials=5_000, num_workers=1, seed=seed)
    assert res.distance == exact


def _kernel(H):
    H = H.copy() % 2
    m, n = H.shape
    piv_cols, r = [], 0
    for col in range(n):
        piv = [i for i in range(r, m) if H[i, col]]
        if not piv:
            continue
        H[[r, piv[0]]] = H[[piv[0], r]]
        for i in range(m):
            if i != r and H[i, col]:
                H[i] ^= H[r]
        piv_cols.append(col)
        r += 1
    free = [c for c in range(n) if c not in piv_cols]
    K = np.zeros((len(free), n), dtype=np.uint8)
    for k, f in enumerate(free):
        K[k, f] = 1
        for i, pc in enumerate(piv_cols):
            K[k, pc] = H[i, f]
    return K


def test_circuit_distance_surface_code():
    stim = pytest.importorskip("stim")
    from sisyphus_qec import dem_to_matrices, lift_columns, verify_fault_set
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3,
                                     after_clifford_depolarization=0.001,
                                     before_measure_flip_probability=0.001,
                                     after_reset_flip_probability=0.001)
    dem = circuit.detector_error_model(decompose_errors=False)
    H, L, columns = dem_to_matrices(dem)
    res = side_distance(H, None, pair_rows=L, num_trials=20_000, num_workers=1, seed=0)
    assert res.distance == 3
    assert verify_fault_set(dem, lift_columns(res.witness, columns))


def test_cutoff_with_small_detector_windows():
    """Regression: a detector window with fewer columns than the cutoff starts at its
    'nothing found' weight n_s + 1 <= cutoff; it must not count that as a cutoff hit (which
    raised the shared stop flag and ended the whole search without a codeword)."""
    stim = pytest.importorskip("stim")
    from sisyphus_qec import dem_to_matrices
    import sisyphus_qec.search as S
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5,
                                     after_clifford_depolarization=0.001,
                                     before_measure_flip_probability=0.001,
                                     after_reset_flip_probability=0.001)
    H, L, _ = dem_to_matrices(circuit.detector_error_model(decompose_errors=False))
    calls = []
    orig = S._island_args_compact

    def spy(n_cols, nw, rank, rounds, seed, cutoff, *a, **kw):
        calls.append((n_cols, cutoff))
        return orig(n_cols, nw, rank, rounds, seed, cutoff, *a, **kw)

    S._island_args_compact = spy
    try:
        cutoff = H.shape[1]                        # above every window's column count
        res = side_distance(H, None, pair_rows=L, num_trials=20_000, num_workers=1, seed=0,
                            cutoff=cutoff, windows=2, island_shots=1_000)
    finally:
        S._island_args_compact = orig
    windows = [(n, c) for n, c in calls if n < H.shape[1]]
    assert windows, "no detector-window islands were created"
    assert all(c == 0 or c <= n for n, c in windows)
    assert 5 <= res.distance <= cutoff


def test_adaptive_scheduler_on_memory_circuit():
    """Adaptive scheduler with period-aligned windows and several workers: the result
    holds a sub-problem log and an improvement history, and the witness is a genuine
    undetectable logical fault set of the DEM."""
    stim = pytest.importorskip("stim")
    from sisyphus_qec import dem_to_matrices, lift_columns, verify_fault_set
    from sisyphus_qec.setup import _side_setup_compact
    from sisyphus_qec.windows import _detector_period
    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=10,
                                     after_clifford_depolarization=0.001,
                                     before_measure_flip_probability=0.001,
                                     after_reset_flip_probability=0.001)
    dem = circuit.detector_error_model(decompose_errors=False)
    H, L, columns = dem_to_matrices(dem)
    _Hp, colsig, _n, _nw, _rank = _side_setup_compact(H, None, L, 1, 1, 8)
    T = _detector_period(H, colsig)
    assert T > 0 and H.shape[0] >= 4 * T
    res = side_distance(H, None, pair_rows=L, num_trials=60_000, num_workers=2, seed=3,
                        island_shots=4_000)
    assert res.distance == 5
    assert verify_fault_set(dem, lift_columns(res.witness, columns))
    assert any(a["name"].startswith("R") and a["islands"] > 0 for a in res.arms)
    ws = [h[1] for h in res.history]
    assert ws and ws[-1] == 5 and all(a > b for a, b in zip(ws, ws[1:]))


@pytest.mark.parametrize("compact", [True, False])
def test_time_limit_single_worker(compact):
    """time_limit stops an in-process search (num_workers=1) inside a long island: the
    island kernels release the GIL, so the timer thread can raise the stop flag. Before
    this fix the single island below ran its full budget (minutes) and ignored the limit."""
    stim = pytest.importorskip("stim")
    import time
    from sisyphus_qec import dem_to_matrices
    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=10,
                                     after_clifford_depolarization=0.001,
                                     before_measure_flip_probability=0.001,
                                     after_reset_flip_probability=0.001)
    H, L, _ = dem_to_matrices(circuit.detector_error_model(decompose_errors=False))
    side_distance(H, None, pair_rows=L, num_trials=200, num_workers=1, seed=1,
                  compact=compact)                              # compile outside the timing
    t0 = time.perf_counter()
    res = side_distance(H, None, pair_rows=L, num_trials=10**9, island_shots=10**8,
                        num_workers=1, seed=2, time_limit=1.0, compact=compact)
    assert time.perf_counter() - t0 < 15.0
    assert res.distance == 5
