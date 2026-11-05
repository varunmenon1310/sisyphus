"""Result types returned by the search.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SideResult:
    """Outcome of one search (one side of a CSS code, or a circuit-level
    instance). `distance` is the lightest certified logical found (an upper
    bound on the true distance), or -1 if the side was skipped. `codewords`
    holds the distinct logicals of that weight found during the run, one per
    row, and `counts[i]` is how often row i was hit (each island stores at
    most `max_codewords` hits, so the set is complete only up to that cap)."""
    distance: int
    codewords: np.ndarray
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    shots: int = 0
    wall_time: float = 0.0
    islands: int = 1          # islands that ran (each with fresh seeds and learning state)
    t_best: float = -1.0      # estimated wall time at which `distance` was first reached
                              # (compact walker; -1 = unknown)

    @property
    def seeds(self) -> int:
        """Older name of `islands`."""
        return self.islands

    @property
    def num_codewords(self) -> int:
        return int(self.codewords.shape[0])

    @property
    def witness(self):
        """One certified codeword of weight `distance`, or None."""
        return self.codewords[0].astype(np.uint8) if self.num_codewords else None

@dataclass
class CssDistanceResult:
    """`x`: X-type logicals (ker Hz minus rowspan Hx); `z`: Z-type logicals."""
    x: SideResult
    z: SideResult

    @property
    def distance(self) -> int:
        ds = [s.distance for s in (self.x, self.z) if s.distance >= 0]
        return min(ds) if ds else -1
