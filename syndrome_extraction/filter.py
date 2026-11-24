"""Staged round filter: reject schedules that are not distance preserving.

A schedule is distance preserving if its d-round memory circuit has no undetectable logical
fault set of weight below the code distance d (in both bases). Any fault set of weight <= d-1
that Sisyphus finds is a checkable witness that the schedule is NOT distance preserving: a
rigorous rejection. A schedule for which nothing is found is only a candidate.

Instead of searching the d-round circuit at once, the filter searches the r-round circuit for
r = 1, 2, 3, 4, 6, 8, 12, 16, ..., d-1 (both bases at each stage) and stops at the first fault
set of weight <= d-1. This is sound and complete (memory basis Z; X is symmetric):
  (a) lifting: a fault set of the r-round circuit is one of every longer circuit (the first r
      rounds are the same, and a flip before the final readout is a data error after round
      r-1). Checked per witness against the bulk / readout columns of the reduced model.
  (b) span: a minimum-weight logical fault set has no fault-free round strictly inside its time
      span (otherwise the syndrome vanishes at that cut and one part alone is a lighter logical
      fault set), so a weight-w fault set fits in w rounds: D(R) >= min(D(r), r + 1). A d-round
      circuit has a fault set of weight <= d-1 iff the (d-1)-round circuit has one.
Most rejections happen at r = 1 or 2, in seconds.

The reduced model is periodic in time, so the r-round model is tiled from the start and
readout columns of the 1-round model and the bulk columns of the 2-round model (`validate`
compares tiled and directly built models).
"""
import json
import time

import numpy as np

from .circuits import matrices, reduced_keys, write_dem
from .search import out_dir, side


class Tiler:
    """Reduced model of one (schedule, basis) for any number of rounds."""

    def __init__(self, code, layers, basis):
        t0 = time.time()
        self.code, self.layers, self.basis = code, layers, basis
        self.ns = (code.Hz if basis == "Z" else code.Hx).shape[0]
        self.nodet = set()
        k1, self.exact = reduced_keys(code, layers, basis, 1)
        sh = self._split(k1, 1)
        self.S0, self.F = sh[0], sh[1]
        self.B = None
        self.consistent = None
        self.t_build = time.time() - t0

    def _split(self, keys, R):
        sh = {b: set() for b in range(R + 1)}
        for d, o in keys:
            if not d:
                self.nodet.add((d, o))
                continue
            b = d[0] // self.ns
            sh[b].add((tuple(x - b * self.ns for x in d), o))
        return sh

    def ensure_B(self):
        if self.B is None:
            t0 = time.time()
            k2, ex2 = reduced_keys(self.code, self.layers, self.basis, 2)
            sh = self._split(k2, 2)
            self.B = sh[1]
            self.exact = self.exact and ex2
            self.consistent = sh[0] == self.S0 and sh[2] == self.F
            self.t_build += time.time() - t0

    def keys(self, r):
        ns = self.ns
        out = set(self.nodet) | self.S0
        if r >= 2:
            self.ensure_B()
            for b in range(1, r):
                out.update((tuple(x + b * ns for x in d), o) for d, o in self.B)
        out.update((tuple(x + r * ns for x in d), o) for d, o in self.F)
        return out

    def member(self, key, R):
        """Is `key` a column of the R-round model? None = undecided (bulk not built yet)."""
        d, o = key
        if not d:
            return key in self.nodet
        b = d[0] // self.ns
        shape = (tuple(x - b * self.ns for x in d), o)
        if b == 0:
            return shape in self.S0
        if b < R:
            return None if self.B is None else shape in self.B
        return shape in self.F if b == R else False


def check_witness(wkeys, tiler, R):
    d, o = set(), set()
    for dd, oo in wkeys:
        d ^= set(dd)
        o ^= set(oo)
    valid = (not d) and bool(o) and len(set(wkeys)) == len(wkeys)
    ms = [tiler.member(k, R) for k in wkeys]
    return bool(valid), (False if False in ms else None if None in ms else True)


def ladder(d, spec="auto"):
    if spec == "auto":
        return [r for r in (1, 2, 3, 4, 6, 8, 12, 16) if r <= d - 2] + [d - 1]
    return [d if t == "d" else d - 1 if t == "d-1" else int(t) for t in spec.split(",")]


def budgets(lad, B):
    """The last stage gets B/2; the earlier ones share B/2 in proportion to r."""
    if len(lad) == 1:
        return [B]
    s = sum(lad[:-1])
    return [B / 2 * r / s for r in lad[:-1]] + [B / 2]


def staged_one(code, layers, budget, workers, seed=1, lad=None, goal=None, bases=("Z", "X"), dem_dir=None):
    """Reject at the first fault set of weight <= goal-1 (goal = d: distance preserving;
    goal = d-1: within one unit). dem_dir: also write every stage's reduced model as a .dem file."""
    goal = code.d if goal is None else goal
    lad = lad or ladder(goal)
    rec = dict(rejected=False, stages=[])
    t_start = time.time()
    tilers = {b: Tiler(code, layers, b) for b in ("Z", "X")}
    for r, secs in zip(lad, budgets(lad, budget)):
        for basis in bases:
            tl = tilers[basis]
            t0 = time.time()
            n_obs = (code.Lz if basis == "Z" else code.Lx).shape[0]
            H, L, klist = matrices(tl.keys(r), (r + 1) * tl.ns, n_obs)
            if dem_dir:
                write_dem(klist, n_obs, f"{dem_dir}/{code.tag}_{basis}_r{r}.dem")
            t_b = time.time() - t0
            w, cw, t_hit = side(H, L, workers, secs, seed, cutoff=goal - 1)
            if w > H.shape[1]:
                w = -1
            st = dict(r=r, basis=basis, secs=round(secs, 1), w=w, t_build=round(t_b, 2), t_hit=round(t_hit, 2),
                      columns=len(klist))
            if 0 < w < goal and cw is not None:
                wkeys = [klist[j] for j in np.nonzero(cw)[0]]
                valid, lift = check_witness(wkeys, tl, code.d)
                st.update(valid=valid, lift_ok=lift)
                if valid and lift is not False and tl.exact:
                    rec["stages"].append(st)
                    rec.update(rejected=True, rejected_at=dict(r=r, basis=basis, w=w,
                                                               t=round(t0 - t_start + t_b + t_hit, 2)),
                               witness=[[list(a), list(b)] for a, b in wkeys])
                    break
            rec["stages"].append(st)
        if rec["rejected"]:
            break
    rec["wall"] = round(time.time() - t_start, 2)
    rec["exact"] = all(t.exact for t in tilers.values())
    return rec


def verify(code, cand_T, basis, witness):
    """Re-check a logged witness against the directly built d-round model."""
    layers = code.layers_from_times(np.asarray(cand_T))
    keys, exact = reduced_keys(code, layers, basis, code.d)
    wk = [(tuple(a), tuple(b)) for a, b in witness]
    d, o = set(), set()
    for a, b in wk:
        d ^= set(a)
        o ^= set(b)
    return bool(exact and all(k in keys for k in wk) and not d and o and len(set(wk)) == len(wk))


def log_path(tag):
    return out_dir() / f"{tag}_staged.jsonl"


def append_log(tag, rec):
    with open(log_path(tag), "a") as f:
        f.write(json.dumps(rec) + "\n")
