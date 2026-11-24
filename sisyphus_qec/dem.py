"""Detector error models (DEMs) to check matrices, and back.

A circuit-level distance problem is given by a Stim detector error model. Each
error mechanism ("fault") flips a set of detectors and a set of logical
observables. With one column per fault,

    H[i, j] = 1  if fault j flips detector i,
    L[k, j] = 1  if fault j flips observable k,

the circuit distance is the smallest number of faults whose combined effect
flips no detector and at least one observable: the minimum weight of c with
H c = 0 and L c != 0. Pass these matrices to
``side_distance(H, None, pair_rows=L)``.

Faults with identical effects (same detectors and same observables) are
interchangeable for the distance, so by default they are merged into a single
column; ``lift_columns`` maps a solution back to DEM error indices.

This helper does not drop detectors. Dropping the detectors of the basis that
is not being protected, when it can be done exactly, shrinks circuit instances
by a further factor of about 2 (dist-m4ri's ``drop=auto`` option implements
an exact version); the benchmark instances in the README were reduced that way.
"""
from __future__ import annotations

import numpy as np


def _error_effects(dem):
    """(detector tuple, observable tuple) of every error instruction of the
    flattened DEM, in order. Decomposed errors (``^`` separators) contribute the
    XOR of their components."""
    effects = []
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        d, o = set(), set()
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                d ^= {t.val}
            elif t.is_logical_observable_id():
                o ^= {t.val}
        effects.append((tuple(sorted(d)), tuple(sorted(o))))
    return effects


def dem_to_matrices(dem, merge_duplicates=True):
    """Check matrices of a ``stim.DetectorErrorModel``.

    Returns ``(H, L, columns)``: ``H`` (detectors x faults) and ``L``
    (observables x faults) as dense uint8 arrays, and ``columns``, a list whose
    entry j is the list of DEM error indices (positions among the error
    instructions of ``dem.flattened()``) represented by column j. Faults that
    flip nothing are dropped. With ``merge_duplicates`` faults with identical
    effects share one column.
    """
    effects = _error_effects(dem)
    groups = {}
    order = []
    for idx, eff in enumerate(effects):
        if not eff[0] and not eff[1]:
            continue
        key = eff if merge_duplicates else (eff, idx)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(idx)
    n = len(order)
    H = np.zeros((dem.num_detectors, n), dtype=np.uint8)
    L = np.zeros((max(1, dem.num_observables), n), dtype=np.uint8)
    columns = []
    for j, key in enumerate(order):
        dets, obs = key if merge_duplicates else key[0]
        H[list(dets), j] = 1
        L[list(obs), j] = 1
        columns.append(groups[key])
    return H, L, columns


def lift_columns(codeword, columns):
    """DEM error indices of a solution: one representative fault per selected
    column (``codeword`` is a 0/1 vector over the columns)."""
    return [columns[j][0] for j in np.nonzero(np.asarray(codeword))[0]]


def verify_fault_set(dem, error_indices):
    """Independent check on the DEM itself: True iff the faults are distinct,
    flip no detector in total, and flip at least one observable."""
    effects = _error_effects(dem)
    idx = [int(i) for i in error_indices]
    if len(set(idx)) != len(idx):
        return False
    d, o = set(), set()
    for i in idx:
        d ^= set(effects[i][0])
        o ^= set(effects[i][1])
    return (not d) and bool(o)
