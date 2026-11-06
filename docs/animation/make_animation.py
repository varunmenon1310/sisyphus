"""Animated GIF for the README: Sisyphus pushes a boulder up a terraced hill whose ledges are
lighter and lighter logical operators of the [[360,12,24]] bivariate bicycle code.

The codewords are real: a plain random information-set search (one fresh information set per
step) on the code records every improvement of the lightest X logical, and Sisyphus itself supplies
the last ledge, a logical of weight 24 = d. The panel below shows the current codeword on the
360 qubits (two 6 x 30 blocks, x^i y^j at row j, column i) and, in blue, a schematic of the
learned column weights: reinforced by every light codeword, partially forgotten when a climb
stalls (the boulder rolls back), as in the search itself.

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

# ----------------------------------------------------------------- the hill
LEDGE, SLOPE, RISE = 0.75, 0.95, 1.0
pts = [(-0.6, 0.0), (0.6, 0.0)]
x = 0.6
ledge_mid = [0.0]
for i in range(1, K):
    x += SLOPE
    pts.append((x, i * RISE))
    ledge_mid.append(x + LEDGE / 2)
    x += LEDGE
    pts.append((x, i * RISE))
pts.append((x + 0.5, (K - 1) * RISE))
P = np.array(pts)
seg = np.diff(P, axis=0)
seglen = np.hypot(seg[:, 0], seg[:, 1])
cum = np.concatenate([[0], np.cumsum(seglen)])


def s_of_x(xq):
    """Arc length at horizontal position xq."""
    j = np.searchsorted(P[:, 0], xq) - 1
    j = int(np.clip(j, 0, len(seg) - 1))
    t = (xq - P[j, 0]) / seg[j, 0]
    return cum[j] + t * seglen[j]


def point(s):
    s = float(np.clip(s, 0, cum[-1]))
    j = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
    t = (s - cum[j]) / seglen[j]
    p = P[j] + t * seg[j]
    ang = np.arctan2(seg[j, 1], seg[j, 0])
    return p, ang


S_LEDGE = [s_of_x(m) for m in ledge_mid]
R_BALL = 0.26

# ----------------------------------------------------------------- timeline
# each frame: (s_ball, ledge index shown, caption, theta-event)
frames = []
STALL_AT = max(2, K - 3)            # the climb from this ledge stalls once and rolls back
BACK_TO = max(1, STALL_AT - 2)


def hold(n_fr, s, lvl, cap, ev=None):
    for t in range(n_fr):
        frames.append((s, lvl, cap, ev if t == 0 else None))


def move(n_fr, s0, s1, lvl, cap, ease=True):
    for t in range(1, n_fr + 1):
        u = t / n_fr
        u = 0.5 - 0.5 * np.cos(np.pi * u) if ease else u
        frames.append((s0 + (s1 - s0) * u, lvl, cap, None))


hold(10, S_LEDGE[0], 0, f"start: a random information set gives a logical of weight {W[0]}", "learn")
lvl = 0
while lvl < K - 1:
    if lvl == STALL_AT and not any(f[3] == "kick" for f in frames):
        mid = 0.5 * (S_LEDGE[lvl] + S_LEDGE[lvl + 1])
        move(10, S_LEDGE[lvl], mid, lvl, "climbing ...")
        for t in range(8):          # pushing, no progress
            frames.append((mid + 0.03 * np.sin(t * 2.2), lvl, "stalled: no lighter logical for a while", None))
        move(14, mid, S_LEDGE[BACK_TO], lvl, "partial forgetting of the learned weights, restart",
             ease=False)
        frames[-14] = (frames[-14][0], lvl, frames[-14][2], "kick")
        hold(6, S_LEDGE[BACK_TO], lvl, "the lightest logical found so far is kept")
        move(12, S_LEDGE[BACK_TO], S_LEDGE[lvl], lvl, "climbing again, on a different path")
    move(12, S_LEDGE[lvl], S_LEDGE[lvl + 1], lvl, "climbing ...")
    lvl += 1
    cap = (f"lighter logical found: weight {W[lvl]}  (verified)" if lvl < K - 1 else
           f"weight {W[lvl]} = d of the [[360,12,24]] bivariate bicycle code")
    hold(9 if lvl < K - 1 else 28, S_LEDGE[lvl], lvl, cap, "learn")

# learned weights theta per frame
theta = np.zeros(n)
thetas = []
for s, lv, cap, ev in frames:
    if ev == "learn":
        theta = 0.6 * theta + trace[lv]
    elif ev == "kick":
        sup = np.nonzero(trace[lv])[0]
        forget = rng.choice(sup, size=max(1, int(0.3 * len(sup))), replace=False)
        theta[forget] *= 0.15
        theta *= 0.7
    thetas.append(theta / max(theta.max(), 1e-9))

# ----------------------------------------------------------------- drawing
fig = plt.figure(figsize=(8.0, 5.6), dpi=100)
ax = fig.add_axes([0.0, 0.33, 1.0, 0.67])
axg = fig.add_axes([0.06, 0.03, 0.88, 0.25])
xmin, xmax = P[0, 0], P[-1, 0] + 0.3
ymin, ymax = -0.8, (K - 1) * RISE + 1.6
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.axis("off")
ax.set_aspect("equal")
sky = np.linspace(0, 1, 256)[:, None]
ax.imshow(sky, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap="Blues", vmin=-1.5, vmax=2.2,
          aspect="auto", zorder=0)
ground = np.vstack([P, [P[-1, 0] + 1, ymin], [P[0, 0] - 1, ymin]])
ax.add_patch(Polygon(ground, closed=True, fc="#8d6e4f", ec="#4e3b2a", lw=2, zorder=1))
for i in range(K):
    xl = ledge_mid[i]
    ax.text(xl, i * RISE - 0.33, f"w={W[i]}", ha="center", va="top", fontsize=9, color="#f5ead8",
            fontweight="bold", zorder=2)
ax.text(xmin + 0.15, ymax - 0.25, "Sisyphus: a hill climb over information sets", fontsize=13,
        fontweight="bold", va="top", color="#1f2d3d")
cap_txt = ax.text(xmin + 0.15, ymax - 0.75, "", fontsize=10.5, va="top", color="#1f2d3d")
flag_pole, = ax.plot([], [], color="#333", lw=2, zorder=3)
flag = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, fc="#e4572e", zorder=3)
ax.add_patch(flag)
ball = Circle((0, 0), R_BALL, fc="#9aa0a6", ec="#3c4043", lw=1.5, zorder=5)
ax.add_patch(ball)
crack, = ax.plot([], [], color="#3c4043", lw=1.2, zorder=6)
body, = ax.plot([], [], color="#202124", lw=4, solid_capstyle="round", zorder=4)
legs, = ax.plot([], [], color="#202124", lw=3, solid_capstyle="round", zorder=4)
arms, = ax.plot([], [], color="#202124", lw=2.5, solid_capstyle="round", zorder=4)
head = Circle((0, 0), 0.10, fc="#202124", zorder=4)
ax.add_patch(head)

# grid panel: 6 rows x (30 + gap + 30) columns
axg.set_xlim(-0.5, 2 * L_ + 0.5); axg.set_ylim(-0.5, 5.5); axg.axis("off"); axg.set_aspect("equal")
cells, dots = [], []
for q in range(n):
    b, r = divmod(q, L_ * M_)
    i, j = divmod(r, M_)
    cx = i + (L_ + 1 if b else 0)
    cells.append(axg.add_patch(Rectangle((cx - 0.45, j - 0.45), 0.9, 0.9, fc="white", ec="#c9d3dd",
                                         lw=0.6)))
    dots.append(axg.add_patch(Rectangle((cx - 0.28, j - 0.28), 0.56, 0.56, fc="#e4572e", ec="none",
                                        visible=False)))
axg.text(L_, 5.95, f"current logical on the {n} qubits (orange); learned column weights (blue)",
         ha="center", va="bottom", fontsize=9, color="#1f2d3d")
cmap = plt.get_cmap("Blues")


def draw(k):
    s, lv, cap, _ = frames[k]
    p, ang = point(s)
    nrm = np.array([-np.sin(ang), np.cos(ang)])
    c = p + nrm * R_BALL
    ball.center = c
    roll = -s / R_BALL
    crack.set_data([c[0] - 0.8 * R_BALL * np.cos(roll), c[0] + 0.8 * R_BALL * np.cos(roll)],
                   [c[1] - 0.8 * R_BALL * np.sin(roll), c[1] + 0.8 * R_BALL * np.sin(roll)])
    # Sisyphus stands downhill of the boulder, leaning into it, hands on the boulder
    u = np.array([np.cos(ang), np.sin(ang)])                 # uphill tangent
    cp = c + R_BALL * (-0.80 * u + 0.60 * nrm)               # hands on the upper back of the boulder
    tdir = np.array([np.cos(ang + 0.55), np.sin(ang + 0.55)])  # torso leaning into the push
    sh = cp - 0.24 * u + 0.02 * nrm
    hip = sh - 0.40 * tdir
    hd = sh + 0.10 * tdir + 0.09 * nrm
    s_hip = s - float(np.dot(c - hip, u))
    stride = 0.10 * np.sin(k * 0.8) if frames[k][3] is None else 0.0
    foot1, _ = point(s_hip - 0.34 + stride)                   # back leg, pushing
    foot2, _ = point(s_hip + 0.02 - stride)                   # front leg
    knee2 = 0.5 * (hip + foot2) + 0.07 * u + 0.03 * nrm
    hand = cp
    body.set_data([hip[0], sh[0]], [hip[1], sh[1]])
    legs.set_data([foot1[0], hip[0], knee2[0], foot2[0]], [foot1[1], hip[1], knee2[1], foot2[1]])
    arms.set_data([sh[0], hand[0]], [sh[1], hand[1]])
    head.center = hd
    cap_txt.set_text(cap)
    if lv == K - 1 and abs(s - S_LEDGE[-1]) < 1e-6:
        fx = ledge_mid[-1] + 0.45
        fy = (K - 1) * RISE
        flag_pole.set_data([fx, fx], [fy, fy + 0.8])
        flag.set_xy([[fx, fy + 0.8], [fx + 0.4, fy + 0.66], [fx, fy + 0.52]])
    else:
        flag_pole.set_data([], [])
        flag.set_xy([[0, -9], [0, -9], [0, -9]])
    th = thetas[k]
    cw = trace[lv]
    for q in range(n):
        cells[q].set_facecolor(cmap(0.08 + 0.75 * th[q]))
        dots[q].set_visible(bool(cw[q]))
    return []


anim = FuncAnimation(fig, draw, frames=len(frames), interval=90, blit=False)
anim.save(OUT, writer=PillowWriter(fps=11))
print("wrote", OUT, len(frames), "frames", flush=True)
