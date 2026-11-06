"""Sisyphus: a fast circuit-level distance estimator for quantum codes (circuit-level detector
error models and CSS codes). See README.md.

Package layout:
    gf2            bit-packed GF(2) linear algebra (M4RI row reduction, kernels)
    params         default search parameters
    results        result types
    extraction     candidate extraction (Prange, Lee-Brickell, Stern) and greedy reduction
    chain_full     full-width search chain (column swaps, learned column weights, restarts)
    chain_compact  compact [I | P] search chain (default)
    windows        detector time windows for circuit-level models
    setup          per-side setup and island arguments
    search         parallel island search and the public entry points
    dem            Stim detector error model helpers
"""
from .search import css_distance, side_distance
from .results import SideResult, CssDistanceResult
from .dem import dem_to_matrices, lift_columns, verify_fault_set

__all__ = ["css_distance", "side_distance", "SideResult", "CssDistanceResult",
           "dem_to_matrices", "lift_columns", "verify_fault_set"]
__version__ = "0.1.0"
