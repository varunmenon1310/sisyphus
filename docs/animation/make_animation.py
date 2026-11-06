"""Animated GIF for the README: Sisyphus pushes a boulder up a terraced hill whose ledges are
lighter and lighter logical operators of the [[360,12,24]] bivariate bicycle code.

The codewords are real: a plain random information-set search (one fresh information set per
step) on the code records every improvement of the lightest X logical, and Sisyphus itself supplies
the last ledge, a logical of weight 24 = d. The panel below shows the current codeword on the
360 qubits (two 6 x 30 blocks, x^i y^j at row j, column i; accent = support) over a grey
schematic of the learned column weights: reinforced by every light codeword, partially
forgotten when a climb stalls (the boulder rolls back), as in the search itself.

    python docs/animation/make_animation.py [out.gif]      (numpy, matplotlib, pillow)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle, Polygon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.codes import bivariate_bicycle  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "docs" / "sisyphus_climb.gif")
rng = np.random.default_rng(7)


# ----------------------------------------------------------------- GF(2) helpers (small sizes)
def rref(M, order=None):
    """Row-reduce M (copy) with columns visited in `order`; returns (R, pivots)."""
    A = M.copy() % 2
    order = np.arange(A.shape[1]) if order is None else order
    r, piv = 0, []
    for c in order:
        rows = np.nonzero(A[r:, c])[0]
        if len(rows) == 0:
            continue
        p = r + rows[0]
        A[[r, p]] = A[[p, r]]
        others = np.nonzero(A[:, c])[0]
        others = others[others != r]
        A[others] ^= A[r]
        piv.append(c)
        r += 1
        if r == A.shape[0]:
            break
    return A[:r], piv


def kernel(H):
    R, piv = rref(H)
    n = H.shape[1]
    free = [c for c in range(n) if c not in set(piv)]
    K = np.zeros((len(free), n), np.uint8)
    for t, f in enumerate(free):
        K[t, f] = 1
        for i, p in enumerate(piv):
            K[t, p] = R[i, f]
    return K


# [[360,12,24]] bivariate bicycle code: A = x^9 + y + y^2, B = y^3 + x^25 + x^26 on Z_30 x Z_6
L_, M_, D_ = 30, 6, 24
Hx, Hz = bivariate_bicycle(L_, M_, [(9, 0), (0, 1), (0, 2)], [(0, 3), (25, 0), (26, 0)])
n = Hx.shape[1]
G = kernel(Hz)                      # X-type: ker Hz
Lz = kernel(Hx)                     # c in ker Hz is a nontrivial X logical iff it has odd
# overlap with some element of ker Hx (the Z stabilizers in ker Hx contribute nothing)


def nontrivial(c):
    return bool((Lz.astype(np.int64) @ c % 2).any())


# ----------------------------------------------------------------- a real improvement trace
trace = []
best = n + 1
for step in range(6000):
    order = rng.permutation(n)
    R, _ = rref(G, order)
    w = R.sum(1)
    for i in np.argsort(w):
        if w[i] >= best:
            break
        if nontrivial(R[i]):
            best = int(w[i])
            trace.append(R[i].copy())
            break
    if best <= D_:
        break
if best > D_:                       # finish the climb with Sisyphus itself
    from sisyphus_qec import side_distance
    res = side_distance(Hz, Hx, num_trials=200000, num_workers=1, seed=3, cutoff=D_)
    trace.append(np.asarray(res.witness, np.uint8))
print("trace weights:", [int(c.sum()) for c in trace], "steps:", step + 1, flush=True)
if len(trace) > 8:                  # keep the first, the last and evenly spaced ledges
    keep = np.unique(np.round(np.linspace(0, len(trace) - 1, 8)).astype(int))
    trace = [trace[i] for i in keep]
W = [int(c.sum()) for c in trace]
K = len(W)

# ----------------------------------------------------------------- style
INK, PAPER, HILL, MUTED, ACCENT = "#232323", "#fafaf7", "#ecebe6", "#9b9a95", "#e0603a"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.antialiased": True,
                     "lines.antialiased": True, "patch.antialiased": True})

# ----------------------------------------------------------------- the hill: smooth terraces
PITCH, SLOPE_W, RISE = 1.55, 0.95, 0.62
X_LEDGE = np.arange(K) * PITCH


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def height(x):
    x = np.asarray(x, float)
    y = np.zeros_like(x)
    for i in range(1, K):
        c = (i - 0.5) * PITCH
        y += RISE * smoothstep((x - c) / SLOPE_W + 0.5)
    return y


XS = np.linspace(-1.2, X_LEDGE[-1] + 1.2, 6000)
YS = height(XS)
ARC = np.concatenate([[0], np.cumsum(np.hypot(np.diff(XS), np.diff(YS)))])


def at_s(s):
    """Ground point and tangent angle at arc length s."""
    x = np.interp(s, ARC, XS)
    dx = 1e-3
    ang = np.arctan2(height(x + dx) - height(x - dx), 2 * dx)
    return np.array([x, float(height(x))]), float(ang)


S_LEDGE = [float(np.interp(x, XS, ARC)) for x in X_LEDGE]
R_BALL = 0.17

# ----------------------------------------------------------------- timeline
frames = []                       # (s, ledge shown, event)
STALL_AT = max(2, K - 3)
BACK_TO = max(1, STALL_AT - 2)


def ease(u):
    return 0.5 - 0.5 * np.cos(np.pi * u)


def hold(m, s, lv, ev=None):
    for t in range(m):
        frames.append((s, lv, ev if t == 0 else None))


def move(m, s0, s1, lv, curve=ease):
    for t in range(1, m + 1):
        frames.append((s0 + (s1 - s0) * curve(t / m), lv, None))


hold(18, S_LEDGE[0], 0, "learn")
lvl = 0
stalled = False
while lvl < K - 1:
    if lvl == STALL_AT and not stalled:
        stalled = True
        mid = S_LEDGE[lvl] + 0.55 * (S_LEDGE[lvl + 1] - S_LEDGE[lvl])
        move(24, S_LEDGE[lvl], mid, lvl)
        for t in range(16):                       # straining, no progress
            frames.append((mid - 0.015 * (1 - np.cos(t * 0.8)), lvl, None))
        frames.append((mid, lvl, "kick"))
        move(26, mid, S_LEDGE[BACK_TO], lvl, curve=lambda u: u * u)   # rolls back, accelerating
        hold(12, S_LEDGE[BACK_TO], lvl)
        move(40, S_LEDGE[BACK_TO], S_LEDGE[lvl], lvl)
    move(28, S_LEDGE[lvl], S_LEDGE[lvl + 1], lvl)
    lvl += 1
    hold(14 if lvl < K - 1 else 48, S_LEDGE[lvl], lvl, "learn")

theta = np.zeros(n)
thetas = []
for s, lv, ev in frames:
    if ev == "learn":
        theta = 0.6 * theta + trace[lv]
    elif ev == "kick":
        sup = np.nonzero(trace[lv])[0]
        forget = rng.choice(sup, size=max(1, int(0.3 * len(sup))), replace=False)
        theta[forget] *= 0.15
        theta *= 0.7
    thetas.append(theta / max(theta.max(), 1e-9))

# ----------------------------------------------------------------- figure
fig = plt.figure(figsize=(8.0, 4.4), dpi=110, facecolor=PAPER)
ax = fig.add_axes([0.03, 0.30, 0.94, 0.68], facecolor=PAPER)
axg = fig.add_axes([0.07, 0.04, 0.86, 0.20], facecolor=PAPER)
xmin, xmax = XS[0], XS[-1]
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.45, (K - 1) * RISE + 0.85)
ax.set_aspect("equal")
ax.axis("off")
ax.fill_between(XS, YS, -1, color=HILL, lw=0, zorder=1)
ax.plot(XS, YS, color=INK, lw=1.4, solid_capstyle="round", zorder=2)
for i in range(K):
    ax.text(X_LEDGE[i], i * RISE - 0.13, f"{W[i]}", ha="center", va="top", fontsize=8.5,
            color=MUTED, zorder=2)
readout = ax.text(xmax - 0.05, (K - 1) * RISE + 0.78, "", ha="right", va="top", fontsize=11,
                  color=INK, family="DejaVu Sans Mono", zorder=6)
ball = Circle((0, 0), R_BALL, fc=INK, ec="none", zorder=5)
ax.add_patch(ball)
spot = Circle((0, 0), 0.032, fc=PAPER, ec="none", zorder=6)
ax.add_patch(spot)
stroke = dict(color=INK, solid_capstyle="round", solid_joinstyle="round", zorder=4)
torso, = ax.plot([], [], lw=2.8, **stroke)
legs, = ax.plot([], [], lw=2.2, **stroke)
arm, = ax.plot([], [], lw=2.0, **stroke)
head = Circle((0, 0), 0.065, fc=INK, ec="none", zorder=4)
ax.add_patch(head)
summit = Circle((X_LEDGE[-1] + 0.45, (K - 1) * RISE + 0.0), 0.05, fc=ACCENT, ec="none", zorder=3,
                visible=False)
ax.add_patch(summit)

# qubit panel: two 6 x 30 blocks
axg.set_xlim(-0.6, 2 * L_ + 0.6)
axg.set_ylim(-0.6, M_ - 0.4)
axg.set_aspect("equal")
axg.axis("off")
base = np.array(matplotlib.colors.to_rgb(HILL))
dark = np.array(matplotlib.colors.to_rgb(MUTED))
acc = np.array(matplotlib.colors.to_rgb(ACCENT))
cells = []
for q in range(n):
    b, r = divmod(q, L_ * M_)
    i, j = divmod(r, M_)
    cx = i + (L_ + 1 if b else 0)
    cells.append(axg.add_patch(Rectangle((cx - 0.42, M_ - 1 - j - 0.42), 0.84, 0.84,
                                         fc=HILL, ec="none")))


def draw(k):
    s, lv, _ = frames[k]
    p, ang = at_s(s)
    u = np.array([np.cos(ang), np.sin(ang)])
    nrm = np.array([-np.sin(ang), np.cos(ang)])
    c = p + R_BALL * nrm
    ball.center = c
    roll = -s / R_BALL
    spot.center = c + 0.6 * R_BALL * np.array([np.cos(roll), np.sin(roll)])
    # the pusher: hands on the upper back of the boulder, body leaning into it
    hand = c + R_BALL * (-0.72 * u + 0.70 * nrm)
    lean = np.deg2rad(46) + 0.5 * ang            # torso angle above the horizontal
    tdir = np.array([np.cos(lean), np.sin(lean)])
    sh = hand + np.array([-0.15, 0.11])
    hip = sh - 0.30 * tdir
    hd = sh + 0.095 * tdir + np.array([0.0, 0.035])
    s_hip = float(np.interp(hip[0], XS, ARC))
    phase = 9.0 * s
    f1, _ = at_s(s_hip - 0.24 + 0.08 * np.sin(phase))
    f2, _ = at_s(s_hip + 0.04 - 0.08 * np.sin(phase))
    knee = 0.5 * (hip + f2) + 0.07 * u + 0.02 * nrm
    torso.set_data([hip[0], sh[0]], [hip[1], sh[1]])
    legs.set_data([f1[0], hip[0], knee[0], f2[0]], [f1[1], hip[1], knee[1], f2[1]])
    arm.set_data([sh[0], hand[0]], [sh[1], hand[1]])
    head.center = hd
    done = lv == K - 1 and abs(s - S_LEDGE[-1]) < 1e-9
    readout.set_text(f"w = {W[lv]}" + (" = d" if done else ""))
    readout.set_color(ACCENT if done else INK)
    summit.set_visible(done)
    th = thetas[k]
    cw = trace[lv]
    for q in range(n):
        cells[q].set_facecolor(acc if cw[q] else base + (dark - base) * (0.85 * th[q]))
    return []


anim = FuncAnimation(fig, draw, frames=len(frames), interval=40, blit=False)
anim.save(OUT, writer=PillowWriter(fps=25), savefig_kwargs={"facecolor": PAPER})
print("wrote", OUT, len(frames), "frames", flush=True)
