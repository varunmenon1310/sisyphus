"""Sisyphus: fast randomized search for the minimum distance of CSS quantum codes and of
circuit-level detector error models. See README.md."""
from .finder import (css_distance, side_distance, metacheck_distance, SideResult,
                     CssDistanceResult)
from .dem import dem_to_matrices, lift_columns, verify_fault_set

__all__ = ["css_distance", "side_distance", "metacheck_distance", "SideResult",
           "CssDistanceResult", "dem_to_matrices", "lift_columns", "verify_fault_set"]
__version__ = "0.1.0"
