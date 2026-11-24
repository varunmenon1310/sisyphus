"""Tests of the syndrome-extraction schedule search (syndrome_extraction/) on the [[672,336,12]]
GALA code, which needs no external files: python -m pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pytest

stim = pytest.importorskip("stim")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syndrome_extraction import codes as C  # noqa: E402
from syndrome_extraction.circuits import reduced_keys  # noqa: E402
from syndrome_extraction.filter import Tiler, staged_one, verify  # noqa: E402
from syndrome_extraction.search import anneal_search  # noqa: E402


@pytest.fixture(scope="module")
def gala672():
    return C.load_code("gala672")


@pytest.fixture(scope="module")
def schedule(gala672):
    return anneal_search(gala672, 1, seed=3)[0]


def test_code(gala672):
    assert gala672.n == 672 and gala672.Lz.shape[0] == gala672.Lx.shape[0] == 336
    assert not (gala672.Hx @ gala672.Lz.T % 2).any() and not (gala672.Hz @ gala672.Lx.T % 2).any()
    assert C.rank(gala672.Lx @ gala672.Lz.T % 2) == 336            # logicals pair up
    seq = np.zeros(gala672.M, np.int64)                              # all X moves before all Z moves
    seq[[i for i, m in enumerate(gala672.moves) if m["kind"] == "Z"]] = 1
    assert gala672.violations(seq) == 0


def test_interleaved_schedule_is_valid(gala672, schedule):
    assert schedule.max() + 1 == 12 and gala672.violations(schedule) == 0
    for basis in ("Z", "X"):           # Stim: noiseless detectors are deterministic (checked inside)
        gala672.circuit(gala672.layers_from_times(schedule), basis, 2)


def test_tiled_model_equals_direct(gala672, schedule):
    layers = gala672.layers_from_times(schedule)
    tl = Tiler(gala672, layers, "Z")
    for r in (1, 2, 3):
        keys, exact = reduced_keys(gala672, layers, "Z", r)
        assert exact and tl.keys(r) == keys


def test_rejection_has_verified_witness(gala672, schedule):
    # random depth-12 schedules of this code lose distance (fault sets of weight <= 11 within seconds)
    rec = staged_one(gala672, gala672.layers_from_times(schedule), budget=40, workers=2, seed=1)
    assert rec["rejected"] and rec["rejected_at"]["w"] < gala672.d
    assert verify(gala672, schedule, rec["rejected_at"]["basis"], rec["witness"])
