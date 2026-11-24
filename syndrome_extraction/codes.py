"""Codes and their block structure ("moves").

A code is given by parity-check matrices Hx, Hz whose checks and data qubits come in blocks,
and every block of checks meets a block of data qubits through one or more permutations
(one group-algebra term each). We call one such permutation, applied by a whole check block
to a whole data block, a *move*: |block| two-qubit gates that can run in the same layer. A
schedule assigns a layer to every move.

Families:
  Mitten codes      [[150,30,10]] ... [[975,195,24]] (arXiv:2607.28795), read from the public
                    repository github.com/a7b/yarn (directory processor_codes; set the
                    environment variable MITTEN_CODES to it). The repository also supplies the
                    published sequential schedule (an X-check round of 12 layers, then a Z-check
                    round of 12 layers).
  APM codes         affine-permutation-matrix codes of Kasai (arXiv:2601.08824) with the
                    instances of Zhao et al. (arXiv:2604.16209): [[1152,580,12]] (apm96),
                    [[2304,1156,14]] (apm192), [[4608,2308,20]] (apm384).
  GALA codes        Yang, Duckering, Dua (arXiv:2608.07431, supplement Tables S3/S4):
                    [[672,336,12]], [[1752,880,14]], [[1764,886,14]], [[2232,1120,16]].
For APM and GALA codes the reference schedule is a greedy edge colouring (X checks, then Z).
"""
import json
import os
import re
from pathlib import Path

import numpy as np

# ───────────────────────────── GF(2) linear algebra ─────────────────────────────


def _pack(A):
    return np.packbits(np.asarray(A, np.uint8) & 1, axis=1)


def _bit(P, c):
    return (P[:, c >> 3] >> (7 - (c & 7))) & 1


def rref(A):
    """Reduced row echelon form over GF(2): (rows, pivot columns)."""
    A = np.asarray(A, np.uint8)
    n = A.shape[1]
    P = _pack(A)
    piv, r = [], 0
    for c in range(n):
        if r == P.shape[0]:
            break
        col = _bit(P[r:], c)
        nz = np.nonzero(col)[0]
        if len(nz) == 0:
            continue
        p = r + nz[0]
        if p != r:
            P[[r, p]] = P[[p, r]]
        hit = np.nonzero(_bit(P, c))[0]
        hit = hit[hit != r]
        if len(hit):
            P[hit] ^= P[r]
        piv.append(c)
        r += 1
    return np.unpackbits(P[:r], axis=1, count=n), piv


def rank(A):
    return len(rref(A)[1])


def nullspace(A):
    """Basis (rows) of {x : A x = 0}."""
    A = np.asarray(A, np.uint8)
    n = A.shape[1]
    R, piv = rref(A)
    free = [c for c in range(n) if c not in set(piv)]
    N = np.zeros((len(free), n), np.uint8)
    N[np.arange(len(free)), free] = 1
    if piv:
        N[:, piv] = R[:, free].T
    return N


def logical_basis(H_commuting, H_same):
    """Logical operators of one type: ker(H_commuting) modulo rowspan(H_same).
    Z logicals: logical_basis(Hx, Hz); X logicals: logical_basis(Hz, Hx)."""
    N = nullspace(H_commuting)
    R, piv = rref(H_same)
    P = _pack(N)
    RP = _pack(R)
    for i, c in enumerate(piv):                    # clear the pivot columns of rowspan(H_same)
        hit = np.nonzero(_bit(P, c))[0]
        if len(hit):
            P[hit] ^= RP[i]
    Nr = np.unpackbits(P, axis=1, count=N.shape[1])
    L, _ = rref(Nr)
    return L


# ───────────────────────────── explicit constructions ─────────────────────────────

def _apm_matrix(a, b, P):
    """P x P affine permutation x -> a x + b (mod P): M[y, x] = 1 iff y = (a x + b) mod P."""
    M = np.zeros((P, P), np.uint8)
    xs = np.arange(P)
    M[(a * xs + b) % P, xs] = 1
    return M


# (P, F, G, d): coefficients (a, b) of the six F_i and six G_i (Zhao et al. Table A1; d = 20 for
# P = 384 is the minimum distance certified by an exact method)
APM = {
    "apm96": (96, [(5, 41), (85, 77), (73, 66), (1, 0), (1, 72), (37, 9)],
              [(61, 15), (1, 24), (89, 62), (25, 22), (85, 93), (25, 78)], 12),
    "apm192": (192, [(71, 127), (97, 80), (67, 117), (163, 165), (25, 60), (187, 33)],
               [(163, 165), (55, 183), (167, 79), (139, 41), (109, 78), (31, 27)], 14),
    "apm384": (384, [(23, 321), (205, 178), (217, 324), (223, 237), (301, 66), (67, 3)],
               [(91, 231), (265, 204), (197, 294), (289, 176), (85, 318), (139, 111)], 20),
}


def apm_code(name):
    """(Hx, Hz): J = 3 block rows of the 6 x 12 block-circulant parent [F | G] / [G^T | F^T]."""
    P, F, G, _ = APM[name]
    Fm = [_apm_matrix(a, b, P) for a, b in F]
    Gm = [_apm_matrix(a, b, P) for a, b in G]
    Hx = np.vstack([np.hstack([Fm[(c - r) % 6] for c in range(6)] + [Gm[(c - r) % 6] for c in range(6)])
                    for r in range(3)]).astype(np.uint8)
    Hz = np.vstack([np.hstack([Gm[(r - c) % 6].T for c in range(6)] + [Fm[(r - c) % 6].T for c in range(6)])
                    for r in range(3)]).astype(np.uint8)
    return Hx, Hz


# GALA: two-block lift over Z_{L/2} x (H_k * C_m), H_k = S_3 (direct product) or S_2 (wreath).
_S3 = {"e": (0, 1, 2), "s0": (1, 2, 0), "s1": (2, 0, 1), "t0": (0, 2, 1), "t1": (2, 1, 0), "t2": (1, 0, 2)}

# generators from Tables S3/S4 of the supplement. Direct products: (h, exponents);
# wreath products: (h, [exponents of block 0, exponents of block 1]).
GALA = {
    "gala672": dict(n=672, k=336, d=12, L=8, J=2, kind="direct", ms=(2, 3, 14),
        F=[[(None, (1, 1, 5))], [(None, (0, 2, 7)), (None, (0, 0, 10))], [(None, (1, 2, 5))], [(None, (1, 0, 6)), (None, (0, 0, 8))]],
        G=[[(None, (0, 1, 13))], [(None, (1, 0, 10)), (None, (0, 2, 10))], [(None, (0, 0, 6))], [(None, (1, 0, 8)), (None, (0, 1, 12))]]),
    "gala1752": dict(n=1752, k=880, d=14, L=12, J=3, kind="wreath", ms=(73,),
        F=[[("e", [(17,), (17,)])], [("e", [(44,), (44,)])], [("e", [(14,), (14,)])], [("e", [(6,), (6,)])],
           [("t", [(8,), (14,)])], [("t", [(42,), (40,)])]],
        G=[[("e", [(36,), (36,)])], [("e", [(20,), (20,)])], [("e", [(61,), (61,)])], [("e", [(10,), (10,)])],
           [("t", [(72,), (5,)])], [("t", [(22,), (20,)])]]),
    "gala1764": dict(n=1764, k=886, d=14, L=12, J=3, kind="direct", ms=(49,),
        F=[[("e", (18,))], [("e", (6,))], [("e", (22,))], [("s0", (21,))], [("t0", (7,))], [("e", (33,))]],
        G=[[("t0", (17,))], [("e", (48,))], [("e", (20,))], [("e", (32,))], [("e", (47,))], [("s0", (11,))]]),
    "gala2232": dict(n=2232, k=1120, d=16, L=12, J=3, kind="direct", ms=(62,),
        F=[[("s0", (48,))], [("t1", (28,))], [("e", (61,))], [("e", (30,))], [("e", (3,))], [("e", (39,))]],
        G=[[("e", (38,))], [("e", (51,))], [("s0", (8,))], [("t1", (13,))], [("e", (5,))], [("e", (48,))]]),
}


def _perm_matrix(p, inverse=False):
    k = len(p)
    M = np.zeros((k, k), np.uint8)
    for i in range(k):
        if inverse:
            M[i, p[i]] = 1
        else:
            M[p[i], i] = 1
    return M


def _shift(m, s):
    M = np.zeros((m, m), np.uint8)
    for i in range(m):
        M[(i + s) % m, i] = 1
    return M


def _cyc(ms, exps):
    M = np.ones((1, 1), np.uint8)
    for m, s in zip(ms, exps):
        M = np.kron(M, _shift(m, s))
    return M


def _gala_term(h, e, kind, ms, inverse):
    if kind == "direct":
        C = _cyc(ms, e)
        return C if h is None else np.kron(_perm_matrix(_S3[h], inverse), C)
    k, m = len(e), int(np.prod(ms))                # wreath: (direct sum_i C(c_i)) . (P(h) (x) I_m)
    D = np.zeros((k * m, k * m), np.uint8)
    for i, ei in enumerate(e):
        D[i * m:(i + 1) * m, i * m:(i + 1) * m] = _cyc(ms, ei)
    p = (0, 1) if h in (None, "e") else (1, 0)
    Pm = np.kron(_perm_matrix(p, inverse), np.eye(m, dtype=np.uint8))
    return (D.astype(np.int64) @ Pm.astype(np.int64) % 2).astype(np.uint8)


def _gala_block(terms, kind, ms, inverse):
    M = None
    for h, e in terms:
        T = _gala_term(h, e, kind, ms, inverse)
        M = T if M is None else (M ^ T)
    return M


def _gala_build(inst, inverse):
    half, J = inst["L"] // 2, inst["J"]
    Fb = [_gala_block(t, inst["kind"], inst["ms"], inverse) for t in inst["F"]]
    Gb = [_gala_block(t, inst["kind"], inst["ms"], inverse) for t in inst["G"]]
    Hx = np.vstack([np.hstack([Fb[(c - r) % half] for c in range(half)] + [Gb[(c - r) % half] for c in range(half)])
                    for r in range(J)]).astype(np.uint8)
    Hz = np.vstack([np.hstack([Gb[(r - c) % half].T for c in range(half)] + [Fb[(r - c) % half].T for c in range(half)])
                    for r in range(J)]).astype(np.uint8)
    return Hx, Hz


def gala_code(name):
    """(Hx, Hz, inverse): the orientation of the permutation action is the one for which
    H_X H_Z^T = 0 and k matches the paper (checked)."""
    inst = GALA[name]
    for inverse in (False, True):
        Hx, Hz = _gala_build(inst, inverse)
        if ((Hx.astype(np.int64) @ Hz.T.astype(np.int64)) % 2).any():
            continue
        if Hx.shape[1] - rank(Hx) - rank(Hz) == inst["k"]:
            return Hx, Hz, inverse
    raise RuntimeError(f"{name}: no orientation gives a valid code")


# ───────────────────────────── block structure ─────────────────────────────

MITTEN = {"m150": "mitten/[[150,30,10]]", "m200": "mitten/[[200,40,12]]",
          "m300": "mitten/[[300,60,14]]", "m500": "mitten/[[500,100,16]]",
          "m540": "mitten/[[540,108,18]]", "m630": "mitten/[[630,126,20]]",
          "m780": "mitten/[[780,156,22]]", "m975": "mitten/[[975,195,24]]",
          "s300": "structured_mitten/[[300,60,9]]", "s330": "structured_mitten/[[330,66,12]]",
          "s600a": "structured_mitten/[[600,120,14]]", "s600b": "structured_mitten/[[600,120,16]]",
          "s840": "structured_mitten/[[840,168,18]]"}
LIFTED = tuple(APM) + tuple(GALA)


def mitten_dir():
    d = os.environ.get("MITTEN_CODES")
    if not d:
        raise RuntimeError("set MITTEN_CODES to the processor_codes directory of github.com/a7b/yarn")
    return Path(d)


class BlockCode:
    """Common part: moves, the X/Z move pairs and the validity constraints of a schedule.

    Validity. Measuring X and Z checks in the same round gives deterministic detectors iff, for
    every X check x and Z check z, the number of shared qubits on which x's gate comes before
    z's gate is even. With block moves the order bit of a shared qubit depends only on the pair
    (X move s, Z move t) that meet there: o_st = [T_s < T_t]. For every check pair the condition
    is a parity constraint on the o_st of the move pairs meeting at their shared qubits; the
    distinct constraints are collected exactly from the gate lists and row-reduced (A o = 0).
    The sequential schedule (all o = 1) satisfies them (checked)."""

    zphase = 0

    def _finish(self):
        assert not ((self.Hx.astype(np.int64) @ self.Hz.T.astype(np.int64)) % 2).any()
        self.M = len(self.moves)
        self.cblocks = sorted({(m["kind"], m["cb"]) for m in self.moves})
        self.dblocks = sorted({m["db"] for m in self.moves})
        X = [i for i, m in enumerate(self.moves) if m["kind"] == "X"]
        Z = [i for i, m in enumerate(self.moves) if m["kind"] == "Z"]
        self.pairs = [(s, t) for s in X for t in Z if self.moves[s]["db"] == self.moves[t]["db"]]
        entry = {}
        for p, (s, t) in enumerate(self.pairs):
            qs = self.moves[s]["gates"][:, 1]
            for x, z in zip(self.moves[s]["chk"][qs].tolist(), self.moves[t]["chk"][qs].tolist()):
                entry.setdefault((x, z), set()).symmetric_difference_update({p})
        rows = sorted({frozenset(v) for v in entry.values() if v}, key=sorted)
        A = np.zeros((len(rows), len(self.pairs)), np.uint8)
        for r, S in enumerate(rows):
            A[r, list(S)] = 1
        self.A_full = A
        self.A = rref(A)[0]
        assert not ((self.A.astype(np.int64) @ np.ones(len(self.pairs), np.int64)) % 2).any()
        self.ps = np.array([s for s, _ in self.pairs])
        self.pt = np.array([t for _, t in self.pairs])

    @staticmethod
    def _chk(moves, n):
        for m in moves:
            chk = np.full(n, -1, np.int64)
            chk[m["gates"][:, 1]] = m["gates"][:, 0]
            m["chk"] = chk

    def violations(self, T, zphase=None):
        """Number of violated parity constraints of the schedule T (move -> layer).
        Pipelined schedules (zphase z > 0: the Z round starts at layer z and ends at layer z-1 of
        the next period) overlap an X round with two Z rounds: order bits against the Z round of
        the same period, o1 = 1 if the Z move wraps (T_t < z) else [T_s < T_t], and against the
        earlier one, o2 = [T_s < T_t] if it wraps else 0; both must satisfy A o = 0."""
        z = self.zphase if zphase is None else zphase
        A = self.A.astype(np.int64)
        T = np.asarray(T)
        lt = (T[self.ps] < T[self.pt]).astype(np.int64)
        if not z:
            return int(((A @ lt) % 2).sum())
        wrap = T[self.pt] < z
        o1 = np.where(wrap, 1, lt)
        o2 = np.where(wrap, lt, 0)
        return int(((A @ o1) % 2).sum() + ((A @ o2) % 2).sum())

    def layers_from_times(self, T):
        """Gate layers [(kind, check row, data qubit), ...] of the schedule T."""
        T = np.asarray(T)
        layers = [[] for _ in range(int(T.max()) + 1)]
        for i, m in enumerate(self.moves):
            for c, q in m["gates"].tolist():
                layers[int(T[i])].append((m["kind"], c, q))
        return layers if self.zphase else [L for L in layers if L]

    def circuit(self, layers, basis, rounds, other_dets=True):
        from .circuits import memory_circuit
        return memory_circuit(self.Hx, self.Hz, basis, rounds, layers=layers,
                              logicals=self.Lz if basis == "Z" else self.Lx, zphase=self.zphase,
                              other_dets=other_dets)


class MittenCode(BlockCode):
    """n = 5|G| data qubits in blocks D1..D5, check blocks X0, X1, Z0, Z1; every check block makes
    9 moves. Moves and the published schedule come from hook_free_SE_cycle_schedule.json."""

    def __init__(self, name):
        sub = MITTEN.get(name, name)
        self.tag = name
        d = mitten_dir() / sub
        self.Hx = np.load(d / "Hx.npy").astype(np.uint8)
        self.Hz = np.load(d / "Hz.npy").astype(np.uint8)
        self.Lx = np.load(d / "Lx.npy").astype(np.uint8)
        self.Lz = np.load(d / "Lz.npy").astype(np.uint8)
        self.n = self.Hx.shape[1]
        self.d = int(sub.split(",")[-1].rstrip("]"))
        sched = json.load(open(d / "hook_free_SE_cycle_schedule.json"))
        self.group = sched["code"].get("group")
        self.moves, self.published = [], {}
        for kind, key in (("X", "X_layers"), ("Z", "Z_layers")):
            self.published[kind] = []
            for layer in sched[key]["layers"]:
                ids = []
                for mv in layer["moves"]:
                    g = np.asarray(mv["cz_gates"], np.int64)
                    assert len(np.unique(g[:, 1])) == len(g) and len(np.unique(g[:, 0])) == len(g)
                    self.moves.append(dict(kind=kind, cb=mv["check_block"], db=mv["data_block"], gates=g))
                    ids.append(len(self.moves) - 1)
                self.published[kind].append(ids)
        self._chk(self.moves, self.n)
        for kind, H in (("X", self.Hx), ("Z", self.Hz)):        # the moves tile Hx and Hz exactly
            R = np.zeros(H.shape, np.int64)
            for m in self.moves:
                if m["kind"] == kind:
                    np.add.at(R, (m["gates"][:, 0], m["gates"][:, 1]), 1)
            assert (R == H).all(), f"moves do not reproduce H{kind}"
        self._finish()

    def published_layers(self):
        return [[(kind, c, q) for i in ids for c, q in self.moves[i]["gates"].tolist()]
                for kind in ("X", "Z") for ids in self.published[kind]]


class LiftedCode(BlockCode):
    """Two-block lifted codes in the Kasai / GALA layout: H_X block (r, c) = F_{(c-r) mod h}
    (c < h) or G_{(c-h-r) mod h}; H_Z block (r, c) = G_{(r-c) mod h}^T (c < h) or
    F_{(r-c+h) mod h}^T; r < J, h = L/2. Every F_i, G_i is a sum of permutation matrices (one per
    group-algebra term); each term in each block row is one move."""

    def __init__(self, name):
        self.tag = name
        if name in APM:
            P, F, G, self.d = APM[name]
            Fterms = [[_apm_matrix(a, b, P)] for a, b in F]
            Gterms = [[_apm_matrix(a, b, P)] for a, b in G]
            J, self.group = 3, f"Aff(Z_{P})"
            Hx0, Hz0 = apm_code(name)
        else:
            inst = GALA[name]
            Hx0, Hz0, inv = gala_code(name)
            term = lambda t: _gala_term(t[0], t[1], inst["kind"], inst["ms"], inv)
            Fterms = [[term(t) for t in ts] for ts in inst["F"]]
            Gterms = [[term(t) for t in ts] for ts in inst["G"]]
            J, self.d, self.group = inst["J"], inst["d"], f"{inst['kind']} ms={inst['ms']}"
        h = len(Fterms)
        b = Fterms[0][0].shape[0]
        self.moves = []
        for kind in ("X", "Z"):
            for r in range(J):
                for c in range(2 * h):
                    if kind == "X":
                        side, idx = ("F", (c - r) % h) if c < h else ("G", (c - h - r) % h)
                        terms = (Fterms if side == "F" else Gterms)[idx]
                    else:
                        side, idx = ("G", (r - c) % h) if c < h else ("F", (r - c + h) % h)
                        terms = [T.T for T in (Gterms if side == "G" else Fterms)[idx]]
                    for t_i, T in enumerate(terms):
                        i, j = np.nonzero(T)
                        assert len(i) == b and len(set(i)) == b and len(set(j)) == b, "term is not a permutation"
                        g = np.stack([r * b + i, c * b + j], 1).astype(np.int64)
                        self.moves.append(dict(kind=kind, cb=r, db=c, gates=g, term=(side, idx, t_i)))
        self.n = 2 * h * b
        self._chk(self.moves, self.n)
        self.Hx = np.zeros((J * b, self.n), np.uint8)
        self.Hz = np.zeros((J * b, self.n), np.uint8)
        for m in self.moves:
            H = self.Hx if m["kind"] == "X" else self.Hz
            assert not H[m["gates"][:, 0], m["gates"][:, 1]].any(), "overlapping terms"
            H[m["gates"][:, 0], m["gates"][:, 1]] = 1
        assert (Hx0 == self.Hx).all() and (Hz0 == self.Hz).all(), "moves do not reproduce the code"
        self.Lx = logical_basis(self.Hz, self.Hx)
        self.Lz = logical_basis(self.Hx, self.Hz)
        self.published = None
        self._finish()

    def published_layers(self):
        from .circuits import edge_colouring
        return ([[("X", c, q) for c, q in L] for L in edge_colouring(self.Hx)]
                + [[("Z", c, q) for c, q in L] for L in edge_colouring(self.Hz)])


def load_code(name):
    """`name` or `name:variant`. A variant (e.g. apm192:mvrowsz6) is the same code under its own
    tag <name>-<variant>, so that candidate files and logs stay apart; a variant ending in z<k>
    selects pipelined schedules whose Z rounds start at layer k."""
    base, _, variant = name.partition(":")
    code = LiftedCode(base) if base in LIFTED else MittenCode(base)
    code.base_tag = code.tag
    if variant:
        m = re.search(r"z(\d+)$", variant)
        if m:
            code.zphase = int(m.group(1))
        code.tag = f"{code.tag}-{variant}"
    return code
