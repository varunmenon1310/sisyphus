"""Memory-experiment circuits (Stim), circuit-level noise, and the reduced detector error model.

memory_circuit builds a noiseless r-round memory experiment for any schedule (a list of gate
layers, X and Z gates may share a layer). Detectors (Z memory): Z checks every round (round 0
against the |0> preparation, then consecutive rounds, then the final data readout) and X checks
from round 1 on; observables: the logical Z operators read from the final data measurement.
X memory is the mirror image. Noiseless detectors are checked to be deterministic.

add_noise is uniform circuit-level noise of strength p (the model of the dist-m4ri benchmark
circuits): DEPOLARIZE2(p) after every CX, a flip with probability p after every reset and
before every measurement, and DEPOLARIZE1(p/10) on every qubit idle in a layer.

For the distance only the supports of the fault mechanisms matter. reduced_keys returns them
as (detectors, observables) pairs with equal columns merged, the other-basis detectors
removed when that is exact (twin_check), and the detectors renumbered by round.
"""
import numpy as np


# ───────────────────────────── schedules ─────────────────────────────

def edge_colouring(H):
    """Greedy proper edge colouring of the Tanner graph of H: list of layers of (check, qubit)."""
    H = np.asarray(H, np.uint8)
    m, n = H.shape
    used_c = [set() for _ in range(m)]
    used_q = [set() for _ in range(n)]
    layers = []
    for c in sorted(range(m), key=lambda c: -int(H[c].sum())):    # heaviest checks first
        for q in np.nonzero(H[c])[0]:
            col = 0
            while col in used_c[c] or col in used_q[q]:
                col += 1
            used_c[c].add(col)
            used_q[q].add(col)
            while len(layers) <= col:
                layers.append([])
            layers[col].append((c, int(q)))
    return layers


# ───────────────────────────── memory circuits ─────────────────────────────

def memory_circuit(Hx, Hz, basis="Z", rounds=1, layers=None, logicals=None, zphase=0, other_dets=True):
    """Noiseless Stim circuit of the memory experiment; returns (circuit, k).

    layers      list of entangling layers, each a list of (kind, check, qubit), kind 'X' or 'Z'
                (row of Hx / Hz, data column). None = edge colouring, X checks then Z checks.
    logicals    logical operators of the memory basis (rows); None = computed.
    zphase      pipelined schedule: the Z round starts at layer zphase of the period and ends at
                layer zphase - 1 of the next one. Z gates of layers < zphase in period 0 are
                dropped and a final partial period runs them. Detectors are declared in the same
                order as for zphase = 0 (round t: memory basis, then the other basis).
    other_dets  False: declare only the memory-basis detectors."""
    import stim
    from .codes import logical_basis
    Hx = np.asarray(Hx, np.uint8)
    Hz = np.asarray(Hz, np.uint8)
    mx, n = Hx.shape
    mz = Hz.shape[0]
    ax = [n + c for c in range(mx)]
    az = [n + mx + c for c in range(mz)]
    if layers is None:
        layers = ([[("X", c, q) for c, q in L] for L in edge_colouring(Hx)]
                  + [[("Z", c, q) for c, q in L] for L in edge_colouring(Hz)])
    if logicals is None:
        logicals = logical_basis(Hx, Hz) if basis == "Z" else logical_basis(Hz, Hx)
    logicals = np.asarray(logicals, np.uint8)
    same, other = (Hz, Hx) if basis == "Z" else (Hx, Hz)
    skind, okind = ("Z", "X") if basis == "Z" else ("X", "Z")
    lines, meas, count = [], {}, [0]

    def measure(gate, qubits, kind, rnd):
        lines.append(f"{gate} " + " ".join(str(q) for q in qubits))
        for c in range(len(qubits)):
            meas[(kind, c, rnd)] = count[0]
            count[0] += 1

    lines.append(("R " if basis == "Z" else "RX ") + " ".join(str(q) for q in range(n)))
    if not zphase:
        for t in range(rounds):
            lines.append("RX " + " ".join(str(q) for q in ax))
            lines.append("R " + " ".join(str(q) for q in az))
            lines.append("TICK")
            for layer in layers:
                lines.append("CX " + " ".join(f"{ax[c]} {q}" if k == "X" else f"{q} {az[c]}" for k, c, q in layer))
                lines.append("TICK")
            measure("MX", ax, "X", t)
            measure("M", az, "Z", t)
            rec = lambda i: f"rec[{i - count[0]}]"
            for c in range(same.shape[0]):
                lines.append(f"DETECTOR {rec(meas[(skind, c, 0)])}" if t == 0 else
                             f"DETECTOR {rec(meas[(skind, c, t)])} {rec(meas[(skind, c, t - 1)])}")
            if t > 0 and other_dets:
                for c in range(other.shape[0]):
                    lines.append(f"DETECTOR {rec(meas[(okind, c, t)])} {rec(meas[(okind, c, t - 1)])}")
            lines.append("TICK")
        measure("M" if basis == "Z" else "MX", list(range(n)), "D", 0)
    else:
        D = len(layers)
        assert 0 < zphase < D
        for p in range(rounds + 1):
            if p < rounds:
                lines.append("RX " + " ".join(str(q) for q in ax))
            if p == 0:
                lines.append("R " + " ".join(str(q) for q in az))
            lines.append("TICK")
            for ell, layer in enumerate(layers):
                if ell == zphase:
                    if p >= 1:
                        measure("M", az, "Z", p - 1)
                    if p == rounds:
                        break
                    if p >= 1:
                        lines.append("R " + " ".join(str(q) for q in az))
                        lines.append("TICK")
                gates = []
                for k, c, q in layer:
                    if k == "X" and p < rounds:
                        gates.append(f"{ax[c]} {q}")
                    elif k == "Z" and ((ell >= zphase and p < rounds) or (ell < zphase and p >= 1)):
                        gates.append(f"{q} {az[c]}")
                if gates:
                    lines.append("CX " + " ".join(gates))
                lines.append("TICK")
            if p < rounds:
                measure("MX", ax, "X", p)
                lines.append("TICK")
        measure("M" if basis == "Z" else "MX", list(range(n)), "D", 0)
        rec = lambda i: f"rec[{i - count[0]}]"
        for t in range(rounds):
            for c in range(same.shape[0]):
                lines.append(f"DETECTOR {rec(meas[(skind, c, 0)])}" if t == 0 else
                             f"DETECTOR {rec(meas[(skind, c, t)])} {rec(meas[(skind, c, t - 1)])}")
            if t > 0 and other_dets:
                for c in range(other.shape[0]):
                    lines.append(f"DETECTOR {rec(meas[(okind, c, t)])} {rec(meas[(okind, c, t - 1)])}")
    rec = lambda i: f"rec[{i - count[0]}]"
    for c in range(same.shape[0]):
        lines.append("DETECTOR " + " ".join(rec(meas[("D", int(q), 0)]) for q in np.nonzero(same[c])[0])
                     + f" {rec(meas[(skind, c, rounds - 1)])}")
    for k, lg in enumerate(logicals):
        lines.append(f"OBSERVABLE_INCLUDE({k}) " + " ".join(rec(meas[("D", int(q), 0)]) for q in np.nonzero(lg)[0]))
    circ = stim.Circuit("\n".join(lines))
    assert not circ.compile_detector_sampler().sample(shots=4).any(), "non-deterministic detector"
    return circ, len(logicals)


# ───────────────────────────── noise ─────────────────────────────

def add_noise(circ, p=0.001):
    """Uniform circuit-level noise for circuits of R, RX, M, MX, CX and TICK (see module docstring)."""
    import stim
    out = stim.Circuit()
    nq = circ.num_qubits
    active = set()
    for inst in circ:
        name = inst.name
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if name == "TICK":
            idle = sorted(set(range(nq)) - active)
            if idle:
                out.append("DEPOLARIZE1", idle, p / 10)
            out.append(inst)
            active.clear()
        elif name in ("R", "RX"):
            out.append(inst)
            out.append("X_ERROR" if name == "R" else "Z_ERROR", qs, p)
            active.update(qs)
        elif name in ("M", "MX"):
            out.append("X_ERROR" if name == "M" else "Z_ERROR", qs, p)
            out.append(inst)
            active.update(qs)
        elif name == "CX":
            out.append(inst)
            out.append("DEPOLARIZE2", qs, p)
            active.update(qs)
        elif name in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"):
            out.append(inst)
        else:
            raise ValueError(f"unsupported instruction {name}")
    if active:
        idle = sorted(set(range(nq)) - active)
        if idle:
            out.append("DEPOLARIZE1", idle, p / 10)
    return out


# ───────────────────────────── reduced DEM ─────────────────────────────

def dem_keys(dem):
    """Distinct (detectors, observables) supports of the error mechanisms of a DEM (parsed from
    the text of each instruction, much faster than targets_copy() with many observables)."""
    bodies = set()
    for ins in dem.flattened():
        if ins.type == "error":
            line = str(ins)
            bodies.add(line[line.index(")") + 1:])
    keys = set()
    for body in bodies:
        d, o = set(), set()
        for tok in body.split():
            if tok[0] == "D":
                d ^= {int(tok[1:])}
            elif tok[0] == "L":
                o ^= {int(tok[1:])}
        if d or o:
            keys.add((tuple(sorted(d)), tuple(sorted(o))))
    return keys


def twin_check(keys, M):
    """Dropping the detectors M is exact iff for every column touching M its twin (S \\ M, O) is
    also a column (or S lies inside M and O is empty): then every fault set without a
    syndrome outside M has one with no syndrome at all and the same logical effect."""
    keyset, Ms = set(keys), set(M)
    for dets, obs in keys:
        if not Ms.intersection(dets):
            continue
        kept = tuple(x for x in dets if x not in Ms)
        if (kept or obs) and (kept, obs) not in keyset:
            return False
    return True


def reduced_keys(code, layers, basis, rounds, p=0.001):
    """Columns of the reduced DEM of the noisy r-round memory circuit: detector t*ns + c = memory-
    basis check c compared at round t (t = rounds: final readout). The other basis's detectors
    are dropped when twin_check allows it. Returns (set of keys, exact)."""
    circ, _ = code.circuit(layers, basis, rounds)
    dem = add_noise(circ, p).detector_error_model(decompose_errors=False)
    ns = (code.Hz if basis == "Z" else code.Hx).shape[0]
    no = (code.Hx if basis == "Z" else code.Hz).shape[0]
    per = ns + no
    assert dem.num_detectors == ns * (rounds + 1) + no * (rounds - 1)

    def red(i):
        if i < ns:
            return i
        t, off = divmod(i - ns, per)
        if t < rounds - 1:
            return (t + 1) * ns + off if off < ns else None
        return rounds * ns + (i - ns - (rounds - 1) * per)

    keys = dem_keys(dem)
    M = {i for i in range(dem.num_detectors) if red(i) is None}
    exact = twin_check(keys, M)
    out = {(tuple(sorted(red(x) for x in d)), o) for d, o in keys if not M.intersection(d)}
    return out, exact


def matrices(keys, n_det, n_obs):
    """Dense (H, L, sorted keys) of a set of reduced DEM columns."""
    keys = sorted(keys)
    H = np.zeros((n_det, len(keys)), np.uint8)
    L = np.zeros((n_obs, len(keys)), np.uint8)
    for j, (d, o) in enumerate(keys):
        H[list(d), j] = 1
        L[list(o), j] = 1
    return H, L, keys


def write_dem(keys, n_obs, path):
    """Reduced DEM as a Stim .dem file (for other distance tools)."""
    lines = ["error(0.001) " + " ".join([f"D{x}" for x in d] + [f"L{x}" for x in o]) for d, o in sorted(keys)]
    nd = 1 + max((d[-1] for d, _ in keys if d), default=0)
    lines.append(f"detector D{nd - 1}")
    lines += [f"logical_observable L{k}" for k in range(n_obs)]
    open(path, "w").write("\n".join(lines) + "\n")
