"""Command line: python -m syndrome_extraction <command> ... (see README.md in this directory).

Candidate schedules live in $SE_RUNS (default ./se_runs) as <tag>_cands.json; logs next to them.
"""
import argparse
import json
import sys
import time

from pathlib import Path

import numpy as np

from .codes import load_code
from .search import anneal_search, hooktable, load_candidates, out_dir, sat_search, save_candidates


def _which(spec, n):
    if "-" in spec:
        a, b = map(int, spec.split("-"))
        return list(range(a, min(b, n)))
    return [int(x) for x in spec.split(",")]


CIRCUITS = Path(__file__).resolve().parent / "schedules" / "circuits.json"


def _layers(code, cand):
    if cand == "best":                               # the circuit of this code in schedules/circuits.json
        ent = [c for c in json.load(open(CIRCUITS)) if c["code"].replace(":", "-") == code.tag]
        assert ent, f"no circuit for {code.tag} in {CIRCUITS}"
        T = np.asarray(ent[0]["T"])
        assert code.violations(T) == 0, "invalid schedule"
        return code.layers_from_times(T)
    if cand == "ref":
        return code.published_layers()
    T = np.asarray(load_candidates(code)[int(cand)]["T"])
    assert code.violations(T) == 0, "invalid schedule"
    return code.layers_from_times(T)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m syndrome_extraction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="code, moves and validity constraints")
    p.add_argument("code")
    p = sub.add_parser("anneal", help="valid schedules by simulated annealing")
    p.add_argument("code"); p.add_argument("num", type=int); p.add_argument("seed", type=int)
    p.add_argument("--depth", type=int, default=12)
    p = sub.add_parser("sat", help="SAT search guided by hook patterns")
    p.add_argument("code"); p.add_argument("depth", type=int); p.add_argument("hours", type=float)
    p.add_argument("workers", type=int)
    p.add_argument("--hook-secs", type=float, default=10.0); p.add_argument("--verify-secs", type=float, default=120.0)
    p.add_argument("--max-cands", type=int, default=20); p.add_argument("--floor", type=int)
    p.add_argument("--r-secs", type=float, default=60.0); p.add_argument("--d-secs", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--n-assume", type=int, default=0)
    p.add_argument("--goal", type=int)
    p = sub.add_parser("hooktable", help="exact single-hook patterns for the SAT search")
    p.add_argument("code"); p.add_argument("basis"); p.add_argument("smax", type=int)
    p.add_argument("shots", type=int); p.add_argument("procs", type=int)
    p = sub.add_parser("filter", help="staged round filter (Sisyphus)")
    p.add_argument("code"); p.add_argument("cands", help="first-last or i,j,k")
    p.add_argument("budget", type=float, help="seconds per schedule"); p.add_argument("workers", type=int)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--goal", type=int)
    p.add_argument("--ladder", default="auto"); p.add_argument("--bases", default="ZX")
    p.add_argument("--dem-dir", help="also write each stage's reduced model as .dem")
    p = sub.add_parser("verify", help="re-check logged rejections on the d-round model")
    p.add_argument("code")
    p = sub.add_parser("validate", help="tiled vs directly built reduced models")
    p.add_argument("code"); p.add_argument("cand"); p.add_argument("rounds", help="r,r,...")
    p = sub.add_parser("check", help="validity of the circuits in schedules/circuits.json")
    p.add_argument("--rounds", type=int, default=2)
    p = sub.add_parser("build", help="write the noiseless memory circuits (.stim)")
    p.add_argument("code"); p.add_argument("cand", help="index, 'ref' or 'best'"); p.add_argument("rounds", type=int)
    p.add_argument("--noisy", type=float, default=0.0, help="add uniform noise of this strength")
    a = ap.parse_args(argv)

    if a.cmd == "check":
        bad = 0
        for c in json.load(open(CIRCUITS)):
            try:
                code = load_code(c["code"])
            except RuntimeError as e:                    # Mitten codes need MITTEN_CODES
                print(f"{c['name']}: skipped ({e})")
                continue
            T = np.asarray(c["T"])
            v = code.violations(T)
            for basis in ("Z", "X"):                     # Stim: noiseless detectors deterministic
                code.circuit(code.layers_from_times(T), basis, a.rounds)
            print(f"{c['name']}: depth {c['depth']}, violated constraints {v}", flush=True)
            bad += v != 0
        print(f"CHECK_DONE invalid={bad}")
        sys.exit(1 if bad else 0)
    if a.cmd == "sat":
        sat_search(a.code, a.depth, a.hours, a.workers, a.hook_secs, a.verify_secs, a.max_cands, a.floor,
                   r_secs=a.r_secs, d_secs=a.d_secs, seed=a.seed, n_assume=a.n_assume, goal=a.goal)
        return
    if a.cmd == "hooktable":
        hooktable(a.code, a.basis, a.smax, a.shots, a.procs)
        return

    code = load_code(a.code)
    print(f"{code.tag}: n={code.n} k={code.Lz.shape[0]} d={code.d} group={code.group} moves={code.M} "
          f"move pairs={len(code.pairs)} constraints={code.A_full.shape[0]} (rank {code.A.shape[0]})", flush=True)
    if a.cmd == "info":
        return
    if a.cmd == "anneal":
        t0 = time.time()
        sols = anneal_search(code, a.num, a.seed, depth=a.depth)
        path, new, tot = save_candidates(code, sols, seed=a.seed)
        print(f"ANNEAL_DONE {new} new, {tot} total -> {path} ({time.time() - t0:.0f} s)")
    elif a.cmd == "filter":
        from .filter import append_log, ladder, staged_one
        cands = load_candidates(code)
        goal = a.goal or code.d
        lad = ladder(goal, a.ladder)
        for i in _which(a.cands, len(cands)):
            r = staged_one(code, code.layers_from_times(np.asarray(cands[i]["T"])), a.budget, a.workers, a.seed,
                           lad, goal, tuple(a.bases), a.dem_dir)
            rec = dict(cand=i, budget=a.budget, seed=a.seed, d=code.d, goal=goal, ladder=lad, **r)
            print(json.dumps({k: v for k, v in rec.items() if k not in ("stages", "witness")}), flush=True)
            append_log(code.tag, rec)
        print("FILTER_DONE")
    elif a.cmd == "verify":
        from .filter import log_path, verify
        cands = load_candidates(code)
        ok = bad = 0
        for line in open(log_path(code.tag)):
            rec = json.loads(line)
            if rec.get("rejected"):
                good = verify(code, cands[rec["cand"]]["T"], rec["rejected_at"]["basis"], rec["witness"])
                ok += good
                bad += not good
                print(f"cand {rec['cand']} {rec['rejected_at']}: {'ok' if good else 'NOT VERIFIED'}", flush=True)
        print(f"VERIFY_DONE ok={ok} bad={bad}")
        sys.exit(1 if bad else 0)
    elif a.cmd == "validate":
        from .circuits import reduced_keys
        from .filter import Tiler
        layers = _layers(code, a.cand)
        for basis in ("Z", "X"):
            tl = Tiler(code, layers, basis)
            tl.ensure_B()
            for r in (code.d if x == "d" else int(x) for x in a.rounds.split(",")):
                dk, ex = reduced_keys(code, layers, basis, r)
                print(json.dumps(dict(basis=basis, r=r, equal=dk == tl.keys(r), exact=ex, columns=len(dk))), flush=True)
    elif a.cmd == "build":
        from .circuits import add_noise
        layers = _layers(code, a.cand)
        for basis in ("Z", "X"):
            circ, k = code.circuit(layers, basis, a.rounds)
            if a.noisy:
                circ = add_noise(circ, a.noisy)
            path = out_dir() / f"{code.tag}_{a.cand}_r{a.rounds}_{basis}.stim"
            circ.to_file(str(path))
            print(f"{path}: {len(layers)} layers, {circ.num_qubits} qubits, {circ.num_detectors} detectors, k={k}")


if __name__ == "__main__":
    main()
