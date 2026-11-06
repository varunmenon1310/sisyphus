"""Animated GIF for the README: Sisyphus pushes a boulder up a terraced hill whose ledges are
lighter and lighter undetectable fault sets of a circuit.

Illustration only. The panel below is a caricature of a detector error model: a space-time grid
of detectors (columns: position, rows: rounds) and, in orange, a set of faults that together
flip no detector and connect the two boundaries, i.e. flip a logical observable. Each ledge
shows a lighter such fault set; the last one, weight 10, is the shortest chain across the grid.

    python docs/animation/make_animation.py [out.gif]      (numpy, matplotlib, pillow)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[2]
OUT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "docs" / "sisyphus_climb.gif")
rng = np.random.default_rng(3)

# ----------------------------------------------------------------- schematic fault sets
NCOL, NROW = 9, 5                  # detectors: positions x rounds (shortest crossing: NCOL + 1 = 10)
W = [24, 19, 15, 12, 10]           # weights of the successive fault sets (10 = distance)
K = len(W)


def chain(weight, seed):
    """A path of `weight` faults across the grid from the left to the right boundary: horizontal
    steps (data errors), vertical steps (measurement errors) and diagonal steps (hook errors)."""
    r = np.random.default_rng(seed)
    x, y = -1, int(r.integers(1, NROW - 1))
    pts = [(x, y)]
    steps = weight
    while steps > 0:
        need = NCOL - x                              # horizontal steps still needed to exit
        slack = steps - need
        if slack > 0 and 0 <= x < NCOL and r.random() < 0.55:   # spend slack on a vertical step
            ny = y + (1 if (y == 0 or (y < NROW - 1 and r.random() < 0.5)) else -1)
            x, y = x, ny
        else:
            dy = 0
            if 0 < x + 1 < NCOL and r.random() < 0.35:
                dy = int(r.choice([-1, 1]))
                if not 0 <= y + dy < NROW:
                    dy = -dy
            x, y = x + 1, y + dy
        pts.append((x, y))
        steps -= 1
    return pts


PANEL = []
for i, w in enumerate(W):
    pts = chain(w, 10 + i)
    segs = [[pts[j], pts[j + 1]] for j in range(len(pts) - 1)]
    inner = np.array([p for p in pts if 0 <= p[0] < NCOL], float)
    PANEL.append((segs, inner))


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

# ----------------------------------------------------------------- figure
fig = plt.figure(figsize=(8.0, 5.0), dpi=110, facecolor=PAPER)
ax = fig.add_axes([0.03, 0.37, 0.94, 0.61], facecolor=PAPER)
axg = fig.add_axes([0.07, 0.09, 0.86, 0.25], facecolor=PAPER)
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
readout = ax.text(xmin + 0.05, (K - 1) * RISE + 0.78, "", ha="left", va="top", fontsize=11,
                  color=INK, family="DejaVu Sans Mono", zorder=6)
ball = Circle((0, 0), R_BALL, fc=INK, ec="none", zorder=5)
ax.add_patch(ball)
spot = Circle((0, 0), 0.032, fc=PAPER, ec="none", zorder=6)
ax.add_patch(spot)
stroke = dict(color=INK, solid_capstyle="round", solid_joinstyle="round", zorder=4)
torso = Polygon(np.zeros((6, 2)), closed=True, fc=INK, ec=INK, lw=1.0, joinstyle="round",
                zorder=4)                     # broad chest and shoulders, narrower waist
ax.add_patch(torso)
# limbs as tapered segments with rounded joints: thigh/calf of each leg, upper arm/forearm
SEGS = {name: ax.add_patch(Polygon(np.zeros((6, 2)), closed=True, fc=INK, ec=INK, lw=0.8,
                                   joinstyle="round", zorder=4))
        for name in ("thigh1", "calf1", "thigh2", "calf2", "upper", "fore", "neck")}
JOINTS = [ax.add_patch(Circle((0, 0), 0.01, fc=INK, ec="none", zorder=4)) for _ in range(7)]
head = Circle((0, 0), 0.074, fc=INK, ec="none", zorder=4)
ax.add_patch(head)


def limb(name, p, q, w_p, w_q, w_m=None, at=0.45):
    """Tapered segment from p (half-width w_p) to q (half-width w_q), bulging to w_m at
    fraction `at` of its length (a muscle belly)."""
    d = q - p
    d = d / max(np.hypot(*d), 1e-9)
    nv = np.array([-d[1], d[0]])
    m = p + at * (q - p)
    w_m = max(w_p, w_q) if w_m is None else w_m
    SEGS[name].set_xy([p + w_p * nv, m + w_m * nv, q + w_q * nv,
                       q - w_q * nv, m - w_m * nv, p - w_p * nv])


summit = Circle((X_LEDGE[-1] + 0.45, (K - 1) * RISE + 0.0), 0.05, fc=ACCENT, ec="none", zorder=3,
                visible=False)
ax.add_patch(summit)

# schematic space-time panel: detectors in grey, boundaries as thin lines, the fault set in orange
axg.set_xlim(-1.6, NCOL + 0.6)
axg.set_ylim(-0.7, NROW - 0.3)
axg.set_aspect("equal")
axg.axis("off")
gx, gy = np.meshgrid(np.arange(NCOL), np.arange(NROW))
axg.scatter(gx.ravel(), gy.ravel(), s=9, c=MUTED, lw=0, zorder=1)
for bx in (-1, NCOL):
    axg.plot([bx, bx], [-0.4, NROW - 0.6], color=MUTED, lw=1.0, zorder=1)
axg.annotate("", xy=(-1.45, NROW - 0.6), xytext=(-1.45, -0.4),
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
axg.text(-1.6, (NROW - 1) / 2, "time", fontsize=7, color=MUTED, ha="right", va="center", rotation=90)
FAULT_LINES = LineCollection([], colors=ACCENT, linewidths=2.4, capstyle="round", zorder=3)
axg.add_collection(FAULT_LINES)
FAULT_DOTS = axg.scatter([], [], s=22, c=ACCENT, lw=0, zorder=4)


# one live sentence under the space-time panel
legend = fig.text(0.5, 0.04, "", ha="center", va="center", color=INK, fontsize=9.5)


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
    lean = np.deg2rad(50) + 0.5 * ang            # torso angle above the horizontal
    tdir = np.array([np.cos(lean), np.sin(lean)])
    sh = hand + np.array([-0.12, 0.22])
    hip = sh - 0.27 * tdir
    hd = sh + 0.11 * tdir + np.array([0.0, 0.05])
    s_hip = float(np.interp(hip[0], XS, ARC))
    phase = 9.0 * s
    f1, _ = at_s(s_hip - 0.19 + 0.07 * np.sin(phase))
    f2, _ = at_s(s_hip + 0.06 - 0.07 * np.sin(phase))
    knee = 0.5 * (hip + f2) + 0.08 * u + 0.03 * nrm
    perp = np.array([-tdir[1], tdir[0]])          # normal to the torso (towards the back)
    chest = hip + 0.62 * (sh - hip)
    torso.set_xy([sh + 0.066 * perp, chest + 0.090 * perp, hip + 0.050 * perp,
                  hip - 0.042 * perp, chest - 0.058 * perp, sh - 0.060 * perp])
    knee1 = 0.5 * (hip + f1) + 0.015 * nrm         # back leg nearly straight: it pushes
    elbow = 0.5 * (sh + hand) - 0.05 * nrm + 0.01 * u    # elbows bent under the load
    neck = sh + 0.05 * tdir + 0.02 * np.array([0.0, 1.0])
    limb("thigh1", hip, knee1, 0.050, 0.032, 0.058, 0.35)
    limb("calf1", knee1, f1, 0.030, 0.017, 0.040, 0.30)
    limb("thigh2", hip, knee, 0.050, 0.032, 0.058, 0.35)
    limb("calf2", knee, f2, 0.030, 0.017, 0.040, 0.30)
    limb("upper", sh, elbow, 0.040, 0.026, 0.046, 0.45)     # biceps
    limb("fore", elbow, hand, 0.026, 0.018, 0.032, 0.30)
    limb("neck", sh, neck, 0.034, 0.030)
    for jc, (pt, r) in zip(JOINTS, [(hip, 0.050), (knee1, 0.031), (knee, 0.032), (sh, 0.060),
                                    (elbow, 0.027), (f1, 0.018), (f2, 0.018)]):
        jc.center, jc.radius = pt, r
    head.center = hd
    done = lv == K - 1 and abs(s - S_LEDGE[-1]) < 1e-9
    readout.set_text(f"w = {W[lv]}" + (" = d" if done else ""))
    readout.set_color(ACCENT if done else INK)
    legend.set_text(f"Weight-{W[lv]} undetectable fault set (schematic detector error model)")
    summit.set_visible(done)
    segs, dots = PANEL[lv]
    FAULT_LINES.set_segments(segs)
    FAULT_DOTS.set_offsets(dots if len(dots) else np.zeros((0, 2)))
    return []


anim = FuncAnimation(fig, draw, frames=len(frames), interval=40, blit=False)
anim.save(OUT, writer=PillowWriter(fps=25), savefig_kwargs={"facecolor": PAPER})
print("wrote", OUT, len(frames), "frames", flush=True)
