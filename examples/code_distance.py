"""Distance of a CSS code: the [[144,12,12]] gross code.

    python examples/code_distance.py
"""
import time

from codes import gross_code
from sisyphus_qec import css_distance

if __name__ == "__main__":
    Hx, Hz = gross_code()

    # 1. Quick distance: stop as soon as a weight-12 logical is found on each side.
    t0 = time.perf_counter()
    res = css_distance(Hx, Hz, num_trials=200_000, num_workers=4, cutoff=12)
    print(f"distance <= {res.distance}   (Z side {res.z.distance}; the X side is skipped "
          f"once the Z side reaches the cutoff)   {time.perf_counter() - t0:.1f} s")

    # 2. Both sides, and the set of minimum-weight logicals hit (no cutoff).
    res = css_distance(Hx, Hz, num_trials=40_000, num_workers=4, seed=1)
    for name, side in (("X", res.x), ("Z", res.z)):
        print(f"{name} side: distance <= {side.distance}, {side.num_codewords} distinct "
              f"minimum-weight logicals found, {side.shots} iterations, {side.wall_time:.1f} s")
    print("one Z logical (qubit indices):", res.z.witness.nonzero()[0].tolist())
