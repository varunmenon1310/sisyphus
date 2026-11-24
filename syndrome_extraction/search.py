"""Searches for valid interleaved schedules.

A schedule is a vector T: move -> layer (depth 12 by default) with at most one move per check
block and per data block in each layer and all parity constraints A o(T) = 0 (codes.BlockCode).

anneal_search   simulated annealing on 4 * (block conflicts) + (violated constraints) (+ an
                optional penalty, e.g. a cap on inter-block hops). Fast: the constraints have a
                large solution space (about 100 distinct depth-12 schedules per code in seconds).

sat_search      exact CNF model of the same rules (one-hot layers, Tseitin XOR chains for the
                parity constraints) and a counterexample-guided loop over *hook patterns*:

  hook code     for memory basis Z the X part of any single circuit fault is one data qubit or
                a hook = the support of an X check restricted to the moves after some layer (a
                fault on an X-check ancilla). Every undetectable logical fault set of the circuit
                projects onto a logical of the hook code H_hook = Hz [I | hooks],
                L_hook = Lz [I | hooks] of no greater weight, so d_circuit >= d_hook for any
                number of rounds. Only the order of the moves within each check block matters.
  patterns      a light hook-code logical uses a set of hooks (check block, set of moves after
                the fault); it stays a logical of the same weight in every schedule in which each
                of those move sets is still a suffix of its block's order. The SAT model forbids
                each learned pattern with one clause ("some move set is not a suffix").
  loop          solve -> search the hook code of both bases with Sisyphus -> add the pattern of
                any light logical -> repeat. With floor < d the loop also searches r-round and
                d-round circuits (heuristic patterns from circuit witnesses). Surviving
                schedules are written to <tag>_sat_cands_*.json; UNSAT with exact patterns only
                proves that no schedule of that depth has hook distance >= floor.
"""
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

from sisyphus_qec.search import _mp_context

from .codes import load_code


def out_dir():
    d = Path(os.environ.get("SE_RUNS", "se_runs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def side(H, L, workers, secs, seed, cutoff=0):
    """Lightest column set of H with zero syndrome and a nonzero logical effect L (Sisyphus).
    Returns (weight, witness or None, time to the best weight)."""
    from sisyphus_qec import side_distance
    if workers >= 2:
        res = side_distance(H, None, pair_rows=L, num_trials=10**9, num_workers=workers, seed=seed,
                            cutoff=cutoff, collect_codewords=False, time_limit=secs)
    else:
        res = side_distance(H, None, pair_rows=L, num_trials=int(secs * 2000), num_workers=1, seed=seed,
                            cutoff=cutoff, collect_codewords=False)
    w = int(res.distance)
    wit = res.witness
    t = float(res.t_best if res.t_best >= 0 else res.wall_time)
    return w, (np.asarray(wit) if wit is not None else None), t


# ───────────────────────────── simulated annealing ─────────────────────────────

def anneal_search(code, num, seed, depth=12, iters=200000, max_tries=None, extra=None):
    """Distinct valid schedules (time vectors) found by simulated annealing."""
    rng = np.random.default_rng(seed)
    M = code.M
    cgroup = np.array([code.cblocks.index((m["kind"], m["cb"])) for m in code.moves])
    dgroup = np.array([code.dblocks.index(m["db"]) for m in code.moves])
    A = code.A.astype(np.int64)
    ps, pt = code.ps, code.pt

    def conflicts(T):
        c = 0
        for grp, ng in ((cgroup, len(code.cblocks)), (dgroup, len(code.dblocks))):
            occ = np.zeros((ng, depth), np.int64)
            np.add.at(occ, (grp, T), 1)
            c += int(np.maximum(occ - 1, 0).sum())
        return c

    def cost(T):
        o = (T[ps] < T[pt]).astype(np.int64)
        return 4 * conflicts(T) + int(((A @ o) % 2).sum()) + (extra(T) if extra else 0)

    sols, seen, tries = [], set(), 0
    while len(sols) < num and (max_tries is None or tries < max_tries):
        tries += 1
        T = rng.integers(0, depth, M)
        c = cost(T)
        temp = 2.0
        for it in range(iters):
            if c == 0:
                break
            i = int(rng.integers(M))
            if rng.random() < 0.5:                              # move one
                old = T[i]
                T[i] = rng.integers(depth)
                c2 = cost(T)
                if c2 <= c or rng.random() < np.exp((c - c2) / temp):
                    c = c2
                else:
                    T[i] = old
            else:                                               # swap two
                j = int(rng.integers(M))
                T[i], T[j] = T[j], T[i]
                c2 = cost(T)
                if c2 <= c or rng.random() < np.exp((c - c2) / temp):
                    c = c2
                else:
                    T[i], T[j] = T[j], T[i]
            temp = max(0.05, temp * 0.9995)
        if c == 0:
            key = tuple(int(x) for x in T)
            if key not in seen:
                seen.add(key)
                sols.append(T.copy())
                print(f"  schedule {len(sols)} after try {tries} ({it} steps)", flush=True)
    return sols


def save_candidates(code, sols, **meta):
    path = out_dir() / f"{code.tag}_cands.json"
    old = json.load(open(path)) if path.exists() else []
    have = {tuple(c["T"]) for c in old}
    new = 0
    for T in sols:
        key = tuple(int(x) for x in T)
        if key not in have:
            have.add(key)
            old.append(dict(T=list(key), **meta))
            new += 1
    json.dump(old, open(path, "w"))
    return path, new, len(old)


def load_candidates(code):
    return json.load(open(out_dir() / f"{code.tag}_cands.json"))


# ───────────────────────────── hook model ─────────────────────────────

class HookModel:
    def __init__(self, code):
        self.code = code
        self.blocks = {}                       # (kind, check block) -> move ids
        for i, m in enumerate(code.moves):
            self.blocks.setdefault((m["kind"], m["cb"]), []).append(i)
        self.c2q = {}                          # move -> data qubit of each check of the block
        for i, m in enumerate(code.moves):
            g = m["gates"]
            self.c2q[i] = g[np.argsort(g[:, 0]), 1]
            self.rows = len(g)
        for ids in self.blocks.values():       # every move of a block covers all its checks
            assert len({tuple(np.sort(code.moves[i]["gates"][:, 0]).tolist()) for i in ids}) == 1
        self.block_rows = {k: np.sort(code.moves[v[0]]["gates"][:, 0]) for k, v in self.blocks.items()}

    def order(self, T, blk):
        ids = self.blocks[blk]
        return [ids[j] for j in np.argsort([T[i] for i in ids], kind="stable")]

    def hook_code(self, T, basis):
        """(H, L, meta): columns = the n data qubits, then for every hook-emitting block and suffix
        size s in 2..w-2 one column per check (meta[g] = (block, s, moves) of column group g)."""
        code = self.code
        hk = "X" if basis == "Z" else "Z"
        Hd = (code.Hz if basis == "Z" else code.Hx).astype(np.uint8)
        Ld = (code.Lz if basis == "Z" else code.Lx).astype(np.uint8)
        Hs, Ls, meta = [Hd], [Ld], []
        for blk in sorted(self.blocks):
            if blk[0] != hk:
                continue
            od = self.order(T, blk)
            w = len(od)
            GH = np.stack([Hd[:, self.c2q[m]] for m in od])
            GL = np.stack([Ld[:, self.c2q[m]] for m in od])
            SH = np.bitwise_xor.accumulate(GH[::-1], axis=0)[::-1]       # SH[k] = XOR_{j >= k} GH[j]
            SL = np.bitwise_xor.accumulate(GL[::-1], axis=0)[::-1]
            for s in range(2, w - 1):
                Hs.append(SH[w - s])
                Ls.append(SL[w - s])
                meta.append((blk, s, frozenset(od[w - s:])))
        return np.ascontiguousarray(np.hstack(Hs)), np.ascontiguousarray(np.hstack(Ls)), meta

    def hook_pattern(self, cw, meta):
        """Hook-code codeword -> frozenset of the (block, move set) hooks it uses."""
        n, b = self.code.n, self.rows
        idx = np.nonzero(np.asarray(cw))[0]
        return frozenset((meta[(j - n) // b][0], meta[(j - n) // b][2]) for j in idx if j >= n)


def circuit_eval(code, hm, T, basis, rounds, workers, secs, seed, cutoff):
    """Search the r-round circuit (reduced DEM); the witness is explained with Stim and mapped to
    the hooks it uses (ancilla faults that spread to the data)."""
    import stim
    from .circuits import add_noise, twin_check
    layers = code.layers_from_times(np.asarray(T))
    circ = add_noise(code.circuit(layers, basis, rounds)[0])
    dem = circ.detector_error_model(decompose_errors=False).flattened()
    lines = [ins for ins in dem if ins.type == "error"]
    ns = (code.Hz if basis == "Z" else code.Hx).shape[0]
    no = (code.Hx if basis == "Z" else code.Hz).shape[0]
    M = set()
    for t in range(1, rounds):
        s = ns + (t - 1) * (ns + no) + ns
        M.update(range(s, s + no))
    first = {}
    for i, ins in enumerate(lines):
        d, o = set(), set()
        for tg in ins.targets_copy():
            if tg.is_relative_detector_id():
                d ^= {int(tg.val)}
            elif tg.is_logical_observable_id():
                o ^= {int(tg.val)}
        if d or o:
            first.setdefault((tuple(sorted(d)), tuple(sorted(o))), i)
    keys = list(first)
    ok = twin_check(keys, M)
    if not ok:
        M = set()
    kept_det = sorted(set(range(dem.num_detectors)) - M)
    row = {x: i for i, x in enumerate(kept_det)}
    kept = [k for k in keys if not M.intersection(k[0])]
    H = np.zeros((len(kept_det), len(kept)), np.uint8)
    L = np.zeros((dem.num_observables, len(kept)), np.uint8)
    for j, (d, o) in enumerate(kept):
        H[[row[x] for x in d], j] = 1
        L[list(o), j] = 1
    w, cw, t = side(H, L, workers, secs, seed, cutoff)
    if w > H.shape[1]:                         # nothing found
        w = -1
    out = dict(w=w, t=round(t, 1), exact=bool(ok))
    if cw is not None and 0 < w < code.d:
        err = [first[kept[j]] for j in np.nonzero(cw)[0]]
        filt = stim.DetectorErrorModel("\n".join(str(lines[i]) for i in err))
        times = sorted(set(int(x) for x in T))
        depth = len(layers)
        n, mx = code.n, code.Hx.shape[0]
        pat = set()
        for e in circ.explain_detector_error_model_errors(dem_filter=filt, reduce_to_one_representative_error=True):
            loc = e.circuit_error_locations[0]
            ell = (loc.tick_offset - 1) % (depth + 2)     # the gates of layer ell came before the fault
            if ell >= depth:
                continue
            for pp in loc.flipped_pauli_product:
                q, pt = pp.gate_target.value, pp.gate_target.pauli_type
                if q < n:
                    continue
                kind, c = ("X", q - n) if q < n + mx else ("Z", q - n - mx)
                if (kind == "X" and pt not in "XY") or (kind == "Z" and pt not in "ZY"):
                    continue
                blk = next(k for k, rows in hm.block_rows.items() if k[0] == kind and c in set(rows.tolist()))
                S = frozenset(i for i in hm.blocks[blk] if T[i] > times[ell])
                if 1 < len(S) < len(hm.blocks[blk]) - 1:
                    pat.add((blk, S))
        out["pattern"] = frozenset(pat)
    return out


# ───────────────────────────── pattern bank ─────────────────────────────

def _pat_to_json(p):
    return [[blk[0], blk[1] if isinstance(blk[1], str) else int(blk[1]), sorted(int(i) for i in S)]
            for blk, S in sorted(p, key=lambda x: (x[0][0], str(x[0][1]), sorted(x[1])))]


def _pat_from_json(j):
    return frozenset(((k, r), frozenset(S)) for k, r, S in j)


class Bank:
    """Failure patterns of one code, shared by all searches (append-only jsonl per writer)."""

    def __init__(self, tag, writer):
        self.tag, self.path = tag, out_dir() / f"{tag}_bank_{writer}.jsonl"
        self.pats, self.circ, self.circw, self.seen_lines = {}, set(), {}, {}
        self.reload()

    def reload(self):
        for f in sorted(out_dir().glob(f"{self.tag}_bank_*.jsonl")):
            lines = open(f).read().split("\n")[:-1]
            k = self.seen_lines.get(f, 0)
            while k < len(lines):
                try:
                    r = json.loads(lines[k])
                except json.JSONDecodeError:
                    break
                self._add(_pat_from_json(r["pattern"]), r["w"], r["src"])
                k += 1
            self.seen_lines[f] = k

    def _add(self, p, w, src):
        if not p:
            return False
        new_circ = src.startswith("circ") and p not in self.circ
        if src.startswith("circ"):
            self.circ.add(p)
            self.circw[p] = min(w, self.circw.get(p, 10**9))
        old = self.pats.get(p)
        if old is None or w < old[0]:
            self.pats[p] = (w, src)
            return True
        return new_circ

    def add(self, p, w, src):
        if self._add(p, w, src):
            with open(self.path, "a") as f:
                f.write(json.dumps(dict(pattern=_pat_to_json(p), w=int(w), src=src)) + "\n")
            self.seen_lines[self.path] = self.seen_lines.get(self.path, 0) + 1


EXACT_SRC = ("hook", "table")          # sources whose patterns are exact for the hook code


# ───────────────────────────── single-hook table ─────────────────────────────

_HT = {}


def _ht_init(codename, basis):
    code = load_code(codename)
    _HT.update(code=code, hm=HookModel(code), Hd=(code.Hz if basis == "Z" else code.Hx).astype(np.uint8),
               Ld=(code.Lz if basis == "Z" else code.Lx).astype(np.uint8))


def _ht_one(job):
    from sisyphus_qec import side_distance
    blk, S, shots, seed = job
    hm, Hd, Ld, d = _HT["hm"], _HT["Hd"], _HT["Ld"], _HT["code"].d
    GH = np.zeros((Hd.shape[0], hm.rows), np.uint8)
    GL = np.zeros((Ld.shape[0], hm.rows), np.uint8)
    for m in S:
        GH ^= Hd[:, hm.c2q[m]]
        GL ^= Ld[:, hm.c2q[m]]
    res = side_distance(np.hstack([Hd, GH]), None, pair_rows=np.hstack([Ld, GL]), num_trials=shots, num_workers=1,
                        seed=seed, cutoff=d - 1, collect_codewords=False)
    return blk, sorted(S), int(res.distance)


def hooktable(codename, basis, smax, shots, procs):
    """Every hook set S (size 2..smax) of every hook-emitting block: if [Hd | hooks(S)] has a
    logical of weight < d, S may be neither a suffix nor a prefix of its block's order (a prefix is
    the suffix of the complement up to a stabilizer); added to the bank as exact patterns."""
    code = load_code(codename)
    hm = HookModel(code)
    hk = "X" if basis == "Z" else "Z"
    rng = np.random.default_rng(7)
    jobs = [(blk, frozenset(S), shots, int(rng.integers(2**31)))
            for blk in sorted(hm.blocks) if blk[0] == hk
            for s in range(2, smax + 1) for S in itertools.combinations(hm.blocks[blk], s)]
    print(f"hook table {code.tag} basis {basis}: {len(jobs)} hook sets", flush=True)
    bank = Bank(code.base_tag, f"table{basis}")
    t0, nbad = time.time(), 0
    with _mp_context().Pool(procs, initializer=_ht_init, initargs=(codename, basis)) as pool:
        for k, (blk, S, w) in enumerate(pool.imap_unordered(_ht_one, jobs, chunksize=4)):
            if 0 < w < code.d:
                nbad += 1
                comp = frozenset(i for i in hm.blocks[tuple(blk)] if i not in set(S))
                bank.add(frozenset([(tuple(blk), frozenset(S))]), w, "table")
                bank.add(frozenset([(tuple(blk), comp)]), w, "table")
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(jobs)}, {nbad} light, {time.time() - t0:.0f} s", flush=True)
    print(f"HOOKTABLE_DONE {nbad}/{len(jobs)} light ({time.time() - t0:.0f} s)")


# ───────────────────────────── SAT model ─────────────────────────────

class ScheduleSAT:
    """CNF over one-hot move layers x[m, t] (depth D): one layer per move, at most one move per
    check block and per data block in each layer, order literals [T_a < T_b] (ladder encoding),
    the parity constraints A o = 0 as Tseitin XOR chains (pipelined: on both order vectors).
    A failure pattern becomes the clause OR over its hooks of 'this move set is not a suffix'."""

    def __init__(self, code, hm, D, solver="cadical195"):
        from pysat.card import CardEnc, EncType
        from pysat.formula import IDPool
        from pysat.solvers import Solver
        self.code, self.hm, self.D = code, hm, D
        self.pool = IDPool()
        self.S = Solver(name=solver)
        M = code.M
        x = lambda m, t: self.pool.id(("x", m, t))
        y = lambda m, t: self.pool.id(("y", m, t))          # y[m, t] = [T_m <= t]
        self.x, self.y = x, y
        for m in range(M):
            for cl in CardEnc.equals([x(m, t) for t in range(D)], 1, vpool=self.pool, encoding=EncType.seqcounter).clauses:
                self.S.add_clause(cl)
        groups = {}
        for i, mv in enumerate(code.moves):
            groups.setdefault(("c", mv["kind"], mv["cb"]), []).append(i)
            groups.setdefault(("d", mv["db"]), []).append(i)
        for ids in groups.values():
            if len(ids) > 1:
                for t in range(D):
                    for cl in CardEnc.atmost([x(m, t) for m in ids], 1, vpool=self.pool, encoding=EncType.seqcounter).clauses:
                        self.S.add_clause(cl)
        for m in range(M):
            for t in range(D):
                self.S.add_clause([-x(m, t), y(m, t)])
                if t + 1 < D:
                    self.S.add_clause([-y(m, t), y(m, t + 1)])
                self.S.add_clause([-y(m, t)] + [x(m, k) for k in range(t + 1)])
        self._lt = {}
        self.o = [self.lt(int(s), int(t)) for s, t in code.pairs]
        z = code.zphase
        if not z:
            for row in code.A:
                self._xor_zero([self.o[p] for p in np.nonzero(row)[0]])
        else:
            o1, o2 = [], []
            for (s_, t_), l in zip(code.pairs, self.o):
                w = y(int(t_), z - 1)
                a = self.pool.id(("o1", int(s_), int(t_)))
                b = self.pool.id(("o2", int(s_), int(t_)))
                self.S.add_clause([-a, w, l]); self.S.add_clause([a, -w]); self.S.add_clause([a, -l])
                self.S.add_clause([b, -w, -l]); self.S.add_clause([-b, w]); self.S.add_clause([-b, l])
                o1.append(a)
                o2.append(b)
            for row in code.A:
                idx = np.nonzero(row)[0]
                self._xor_zero([o1[p] for p in idx])
                self._xor_zero([o2[p] for p in idx])

    def lt(self, a, b):
        """Literal [T_a < T_b] (a and b never share a layer)."""
        if (a, b) in self._lt:
            return self._lt[(a, b)]
        if (b, a) in self._lt:
            return -self._lt[(b, a)]
        v = self.pool.id(("lt", a, b))
        for t in range(self.D):
            self.S.add_clause([-v, -self.x(b, t)] + ([self.y(a, t - 1)] if t > 0 else []))
            self.S.add_clause([v, -self.x(a, t)] + ([self.y(b, t - 1)] if t > 0 else []))
        self._lt[(a, b)] = v
        return v

    def _xor_zero(self, lits):
        if not lits:
            return
        acc = lits[0]
        for l in lits[1:]:
            z = self.pool.id(("xor", len(self.pool.obj2id)))
            self.S.add_clause([-z, acc, l]); self.S.add_clause([-z, -acc, -l])
            self.S.add_clause([z, -acc, l]); self.S.add_clause([z, acc, -l])
            acc = z
        self.S.add_clause([-acc])

    def forbid(self, pattern):
        lits = []
        for blk, S in pattern:
            lits += [self.lt(a, b) for a in S for b in self.hm.blocks[blk] if b not in S]
        self.S.add_clause(lits)

    def solve(self, rng=None, n_assume=0):
        ok = False
        if rng is not None and n_assume > 0:                # diversify: pin a few random moves
            ms = rng.choice(self.code.M, size=n_assume, replace=False)
            ok = self.S.solve(assumptions=[self.x(int(m), int(rng.integers(self.D))) for m in ms])
        if not ok and not self.S.solve():
            return None
        mod = set(l for l in self.S.get_model() if l > 0)
        return np.array([next(t for t in range(self.D) if self.x(m, t) in mod) for m in range(self.code.M)], np.int64)

    def block_solution(self, T):
        """Forbid these block orders (all of them at once) to get a different schedule."""
        lits = []
        for ids in self.hm.blocks.values():
            od = sorted(ids, key=lambda i: T[i])
            lits += [-self.lt(a, b) for a, b in zip(od, od[1:])]
        self.S.add_clause(lits)


def sat_search(codename, D, hours, workers, hook_secs=10.0, verify_secs=120.0, max_cands=20, floor=None,
               r=2, r_secs=60.0, d_secs=300.0, seed=0, n_assume=0, goal=None):
    """Counterexample-guided SAT search (module docstring)."""
    code = load_code(codename)
    hm = HookModel(code)
    d = code.d
    floor = d if floor is None else floor
    goal = d if goal is None else goal
    gtag = "" if goal == d else f"_g{goal}"
    bank = Bank(code.base_tag, f"sat{D}_f{floor}" + (f"_s{seed}" if seed else "") + gtag)
    t0 = time.time()
    sat = ScheduleSAT(code, hm, D)
    print(f"SAT {code.tag} D={D}: {sat.pool.top} variables, {sat.S.nof_clauses()} clauses ({time.time() - t0:.1f} s)", flush=True)
    added = set()

    def sync():
        bank.reload()
        k = 0
        for p, (w, src) in bank.pats.items():
            if p not in added and ((src in EXACT_SRC and w < floor) or (floor < d and bank.circw.get(p, 10**9) < goal)):
                sat.forbid(p)
                added.add(p)
                k += 1
        return k

    print(f"  patterns from the bank: {sync()}", flush=True)
    log = out_dir() / f"{code.tag}_sat.jsonl"
    surv_path = out_dir() / (f"{code.tag}_sat_cands_D{D}_f{floor}" + (f"_s{seed}" if seed else "") + gtag + ".json")
    t_end = time.time() + 3600 * hours
    n_iter = n_cand = 0
    heuristic = False
    rng = np.random.default_rng(1000 * seed + D)
    while time.time() < t_end and n_cand < max_cands:
        sync()
        ts = time.time()
        T = sat.solve(rng, n_assume)
        n_iter += 1
        heuristic = heuristic or any(p in bank.circ for p in added)
        if T is None:
            print(f"UNSAT after {n_iter} iterations and {len(added)} patterns ({time.time() - t0:.0f} s; "
                  f"circuit-derived clauses used: {heuristic})", flush=True)
            with open(log, "a") as f:
                f.write(json.dumps(dict(iter=n_iter, D=D, floor=floor, unsat=True, heuristic=heuristic,
                                        patterns=len(added))) + "\n")
            break
        assert code.violations(T) == 0
        rec = dict(iter=n_iter, D=D, floor=floor, goal=goal, t_solve=round(time.time() - ts, 2),
                   patterns=len(added), T=[int(v) for v in T])
        verdict = "hook_clean"
        for basis in ("Z", "X"):
            H, L, meta = hm.hook_code(T, basis)
            w, cw, _ = side(H, L, workers, hook_secs, seed=int(rng.integers(2**31)), cutoff=floor - 1)
            rec[f"hook_{basis}"] = w
            if cw is not None and 0 < w < floor:
                bank.add(hm.hook_pattern(cw, meta), w, "hook")
                verdict = f"hook_{basis}"
                break
        if verdict == "hook_clean" and floor < d:          # circuit tiers: r rounds, then d rounds
            for rounds, secs, name in ((r, r_secs, f"r{r}"), (d, d_secs, "d")):
                for basis in ("Z", "X"):
                    ce = circuit_eval(code, hm, T, basis, rounds, workers, secs, int(rng.integers(2**31)), goal - 1)
                    rec[f"{name}_{basis}"] = ce["w"]
                    if ce["w"] < goal:
                        if ce.get("pattern"):
                            bank.add(ce["pattern"], ce["w"], f"circ_{name}")
                        verdict = f"{name}_{basis}" if ce["w"] > 0 else f"{name}_{basis}_nofind"
                        break
                if verdict != "hook_clean":
                    break
            if verdict == "hook_clean":
                verdict = "circuit_clean"
        elif verdict == "hook_clean":                       # longer confirmation of both sides
            for basis in ("Z", "X"):
                H, L, meta = hm.hook_code(T, basis)
                w, cw, _ = side(H, L, workers, verify_secs, seed=int(rng.integers(2**31)), cutoff=d - 1)
                rec[f"hookv_{basis}"] = w
                if cw is not None and 0 < w < d:
                    bank.add(hm.hook_pattern(cw, meta), w, "hook")
                    verdict = f"hookv_{basis}"
                    break
        rec["verdict"] = verdict
        print(json.dumps({k: v for k, v in rec.items() if k != "T"}), flush=True)
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if verdict in ("hook_clean", "circuit_clean"):
            n_cand += 1
            old = json.load(open(surv_path)) if surv_path.exists() else []
            old.append(dict(T=rec["T"], D=D, rec={k: v for k, v in rec.items() if k != "T"}))
            json.dump(old, open(surv_path, "w"))
            sat.block_solution(T)
    print(f"SAT_DONE {n_iter} iterations, {n_cand} schedules, {len(added)} patterns ({time.time() - t0:.0f} s)")
