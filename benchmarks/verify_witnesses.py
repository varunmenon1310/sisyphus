"""Independent check of the fault sets in benchmarks/witnesses/ with Stim.

For each witness: build the detector error model of the original circuit
(from the dist-m4ri benchmark folder), check that every listed fault is an
error mechanism of that model, that the faults are distinct, that together
they flip no detector, and that they flip at least one logical observable.

usage: python benchmarks/verify_witnesses.py <path to dist-m4ri/benchmark> [witness.json ...]
"""
import json
import sys
from pathlib import Path

import stim


def effects(dem):
    out = set()
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        d, o = set(), set()
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                d ^= {t.val}
            elif t.is_logical_observable_id():
                o ^= {t.val}
        out.add((tuple(sorted(d)), tuple(sorted(o))))
    return out


def main():
    bench = Path(sys.argv[1])
    files = [Path(f) for f in sys.argv[2:]] or sorted((Path(__file__).parent / "witnesses").glob("*.json"))
    ok_all = True
    for f in files:
        w = json.loads(f.read_text())
        circ = stim.Circuit.from_file(str(bench / w["circuit"]))
        known = effects(circ.detector_error_model(decompose_errors=False))
        faults = [(tuple(x["detectors"]), tuple(x["observables"])) for x in w["faults"]]
        present = all(fl in known for fl in faults)
        distinct = len(set(faults)) == len(faults)
        d, o = set(), set()
        for dets, obs in faults:
            d ^= set(dets)
            o ^= set(obs)
        ok = present and distinct and not d and bool(o)
        ok_all &= ok
        print(f"{w['instance']:14s} weight {w['weight']:3d}  faults in DEM: {present}  distinct: {distinct}  "
              f"detectors left: {len(d)}  observables flipped: {sorted(o)}  -> {'VALID' if ok else 'INVALID'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
