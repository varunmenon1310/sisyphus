"""Animated GIF for the README: Sisyphus pushes a boulder across a rugged landscape whose height
is the lightness of the lightest fault set visible from the current information set.

Illustration of the search, not a trace of a real run:
  * the walk moves between neighbouring information sets (one column swap at a time), guided by
    learned column priorities, so it climbs towards lighter fault sets (higher ground);
  * the landscape is rugged: most climbs end on a local optimum where every nearby swap makes the
    visible fault sets heavier;
  * when a walker stalls, a kick forgets part of its priorities and it rolls into a new basin,
    sometimes onto a peak that is no better than the best so far;
  * the best fault set so far is kept (flags, lower-left panel), and the run ends when its weight
    reaches the target (here the distance).
The lower-left panel is a caricature of a detector error model: a space-time grid of detectors
and, in orange, the lightest set of faults found so far that flips no detector and connects the
two boundaries. The lower-right panel plots the current weight (grey) and the best weight (black).

    python docs/animation/make_animation.py [out.gif]      (numpy, matplotlib, pillow)
    STILL=5,40 python docs/animation/make_animation.py     (PNG stills of those frames)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[2]
OUT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "docs" / "sisyphus_climb.gif")
FPS = 25

# ----------------------------------------------------------------- schematic fault sets
NCOL, NROW = 9, 5                  # detectors: positions x rounds (shortest crossing: NCOL + 1 = 10)
W = [24, 19, 15, 12, 10]           # weights of the successive best fault sets (10 = distance)


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


PANEL = {}
for i, w in enumerate(W):
    pts = chain(w, 10 + i)
    segs = [[pts[j], pts[j + 1]] for j in range(len(pts) - 1)]
    inner = np.array([p for p in pts if 0 <= p[0] < NCOL], float)
    PANEL[w] = (segs, inner)

# ----------------------------------------------------------------- style
INK, PAPER, HILL, MUTED, ACCENT = "#232323", "#fafaf7", "#ecebe6", "#9b9a95", "#e0603a"
FOG = "#c8c7c1"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.antialiased": True,
                     "lines.antialiased": True, "patch.antialiased": True})

# ----------------------------------------------------------------- the landscape
# Control points (kind, weight): height grows as the weight falls. Consecutive points are joined
# by smooth steps, so every control point is flat: peaks, valleys, and ledges on long slopes.
DW = 0.15                                          # height per unit of weight
W_TOP = 31


def hgt(w):
    return (W_TOP - w) * DW


CTRL = [("start", 30), ("ledge", 27), ("peak", 24), ("valley", 29), ("peak", 27), ("valley", 28),
        ("ledge", 23), ("peak", 19), ("valley", 25), ("peak", 21), ("valley", 29), ("ledge", 24),
        ("ledge", 18), ("peak", 15), ("valley", 20), ("ledge", 16), ("peak", 12), ("valley", 17),
        ("peak", 10), ("valley", 16), ("end", 16)]
CX = [0.0]
for (_, w0), (_, w1) in zip(CTRL[:-1], CTRL[1:]):
    CX.append(CX[-1] + 0.6 + 1.35 * abs(hgt(w1) - hgt(w0)))
CX = np.array(CX)
CX[-1] = CX[-2] + 1.6
CH = np.array([hgt(w) for _, w in CTRL])
X0, X1 = -1.2, CX[-1]


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


XS = np.linspace(X0, X1, 24000)
YS = np.full_like(XS, CH[0])
for i in range(len(CX) - 1):
    m = (XS >= CX[i]) & (XS <= CX[i + 1])
    u = (XS[m] - CX[i]) / (CX[i + 1] - CX[i])
    k = max(1, int(round((CX[i + 1] - CX[i]) / 0.42)))
    rough = 0.03 * np.sin(np.pi * u) * np.sin(2 * np.pi * k * u)    # small-scale roughness
    YS[m] = CH[i] + (CH[i + 1] - CH[i]) * _smooth(u) + rough
ARC = np.concatenate([[0], np.cumsum(np.hypot(np.diff(XS), np.diff(YS)))])


def height(x):
    return np.interp(x, XS, YS)


def at_s(s):
    """Ground point and tangent angle at arc length s."""
    x = float(np.interp(s, ARC, XS))
    dx = 2e-3
    ang = np.arctan2(height(x + dx) - height(x - dx), 2 * dx)
    return np.array([x, float(height(x))]), float(ang)


def s_of_x(x):
    return float(np.interp(x, XS, ARC))


R_BALL = 0.30
FIG = 0.60                      # scale of the figure (lengths below are at FIG = 1)
L_ARM, L_TORSO, L_THIGH, L_CALF = 0.25, 0.27, 0.19, 0.19
LEG = FIG * (L_THIGH + L_CALF)
REACH = 0.95 * FIG * L_ARM


def ball_center(s):
    p, ang = at_s(s)
    return p + R_BALL * np.array([-np.sin(ang), np.cos(ang)]), ang


def body(s_h, lean):
    g, a_g = at_s(s_h)                             # hips higher on a slope (uphill front foot)
    hip = g + np.array([0.0, (0.80 + 0.22 * np.sin(max(a_g, 0.0))) * LEG])
    return hip, hip + FIG * L_TORSO * np.array([np.cos(lean), np.sin(lean)])


def push_lean(ang):
    return np.deg2rad(68) + 0.30 * ang


def push_hip(s):
    """Arc position of the hips when the arms, pointing at the boulder's centre, just reach it."""
    c, ang = ball_center(s)
    lean = push_lean(ang)
    lo, step = s - 2.0, 0.02
    while step > 1e-4:
        while lo + step < s and np.hypot(*(body(lo + step, lean)[1] - c)) - R_BALL > REACH:
            lo += step
        step *= 0.5
    return lo


# ----------------------------------------------------------------- timeline
# frame = (ball arc, figure arc or None (pushing), best weight, status, camera zoom)
S_PT = [s_of_x(x) for x in CX]
frames = []
best = [None]
status = ["climb"]


def emit(sb, sf=None, zoom=0.0):
    frames.append((sb, sf, best[0], status[0], zoom))


def push(i0, i1, speed=0.095):
    """Push the boulder from control point i0 to i1 (pauses briefly on ledges in between)."""
    for i in range(i0, i1):
        a, b = S_PT[i], S_PT[i + 1]
        if i == i0 and i0 == 0:
            a = s_of_x(-0.35)
        n = max(8, int((b - a) / speed))
        for t in range(1, n + 1):
            u = t / n
            emit(a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * u)) if i in (i0, i1 - 1)
                 else a + (b - a) * u)
        if i + 1 < i1 and CTRL[i + 1][0] == "ledge":
            for _ in range(5):
                emit(b)


def stall(i, n=28, amp=0.10):
    """Strain on a local optimum: tries nearby moves, all of them downhill (heavier)."""
    s0 = S_PT[i]
    for t in range(n):
        emit(s0 + amp * np.sin(2 * np.pi * t / 17) * min(1.0, t / 6))
    emit(s0)


def kick(i_from, i_to):
    """The boulder rolls from a peak into the next valley; the figure follows on foot."""
    a, b = S_PT[i_from], S_PT[i_to]
    f0, f1 = push_hip(a), push_hip(b)
    n = max(16, int((b - a) / 0.11))
    m = n + 12
    for t in range(1, m + 1):
        u = min(1.0, t / n)
        sb = a + (b - a) * u * u * (3 - 2 * u)
        v = np.clip((t - 6) / (m - 6), 0.0, 1.0)
        emit(sb, f0 + (f1 - f0) * (0.5 - 0.5 * np.cos(np.pi * v)))


def record(i, hold_n=12):
    best[0] = CTRL[i][1]
    status[0] = "record"
    for _ in range(hold_n):
        emit(S_PT[i])


def idx(kind, w, start=0):
    return next(j for j in range(start, len(CTRL)) if CTRL[j] == (kind, w))


i24, i19, i21, i15, i12, i10 = (idx("peak", 24), idx("peak", 19), idx("peak", 21),
                                idx("peak", 15), idx("peak", 12), idx("peak", 10))
for _ in range(14):
    emit(s_of_x(-0.35))
push(0, i24)
record(i24)
status[0] = "stuck"; stall(i24)
status[0] = "kick"; kick(i24, i24 + 1)
status[0] = "climb"; push(i24 + 1, i19)
record(i19)
status[0] = "stuck"; stall(i19)
status[0] = "kick"; kick(i19, i19 + 1)
status[0] = "climb"; push(i19 + 1, i21)
status[0] = "worse"; stall(i21, n=34)
status[0] = "kick"; kick(i21, i21 + 1)
status[0] = "climb"; push(i21 + 1, i15)
record(i15)
status[0] = "stuck"; stall(i15, n=20)
status[0] = "kick"; kick(i15, i15 + 1)
status[0] = "climb"; push(i15 + 1, i12)
record(i12)
status[0] = "stuck"; stall(i12, n=18)
status[0] = "kick"; kick(i12, i12 + 1)
status[0] = "climb"; push(i12 + 1, i10)
best[0] = 10
status[0] = "done"
for _ in range(20):
    emit(S_PT[i10])
NZ = 45
for t in range(1, NZ + 1):                         # zoom out over the explored landscape
    emit(S_PT[i10], zoom=0.5 - 0.5 * np.cos(np.pi * t / NZ))
for _ in range(55):
    emit(S_PT[i10], zoom=1.0)
NF = len(frames)

STATUS = {
    "climb": "Climbing: learned column priorities choose the swaps",
    "record": "New lightest fault set: weight {b}",
    "stuck": "Local optimum: every nearby swap gives heavier fault sets",
    "kick": "Kick: forget part of the priorities and roll into a new basin",
    "worse": "Local optimum, heavier than the best so far ({b})",
    "done": "Weight {b} reached: the distance",
}

# camera: follows the boulder (smoothed), then zooms out to the whole explored landscape
bx = np.array([ball_center(f[0])[0][0] for f in frames])
by = np.array([ball_center(f[0])[0][1] for f in frames])
ker = np.exp(-0.5 * (np.arange(-30, 31) / 12.0) ** 2)
ker /= ker.sum()


def _sm(v):
    return np.convolve(np.pad(v, 30, mode="edge"), ker, mode="valid")


VW = 7.4                                            # view width at zoom 0 (data units)
CAMX, CAMY = _sm(bx + 1.0), _sm(by - 0.05)
w_cur = np.array([W_TOP - (by[k] - R_BALL) / DW for k in range(NF)])
explored = np.maximum.accumulate(bx + R_BALL + 0.2)

# ----------------------------------------------------------------- figure
fig = plt.figure(figsize=(8.0, 5.0), dpi=int(os.environ.get("DPI", 100)), facecolor=PAPER)
AX_RECT = [0.03, 0.40, 0.94, 0.58]
ax = fig.add_axes(AX_RECT, facecolor=PAPER)
ASPECT = (AX_RECT[3] * 5.0) / (AX_RECT[2] * 8.0)    # view height / view width
ax.set_aspect("equal")
ax.axis("off")
ax.fill_between(XS, YS, -6, color=HILL, lw=0, zorder=1)
ax.plot(XS, YS, color=FOG, lw=1.1, zorder=2)        # terrain the search has not seen yet
seen_line, = ax.plot([], [], color=INK, lw=1.4, solid_capstyle="round", zorder=2)
LABELS = []
for j, (kind, w) in enumerate(CTRL):
    if kind in ("peak", "valley"):
        t = ax.text(CX[j], CH[j] - 0.12, f"{w}", ha="center", va="top", fontsize=8.5,
                    color=MUTED, zorder=2, visible=False)
        LABELS.append((CX[j], t))
FLAGS = {}
for w in W:
    j = idx("peak", w)
    x, y = CX[j] + 0.42, CH[j]
    pole, = ax.plot([x, x], [y, y + 0.46], color=INK, lw=1.0, zorder=3, visible=False)
    cloth = ax.add_patch(Polygon([[x, y + 0.46], [x + 0.2, y + 0.39], [x, y + 0.32]],
                                 closed=True, fc=ACCENT, ec="none", zorder=3, visible=False))
    FLAGS[w] = (pole, cloth)
readout = ax.text(0.01, 0.97, "", transform=ax.transAxes, ha="left", va="top", fontsize=11,
                  color=INK, family="DejaVu Sans Mono", zorder=6)
readout2 = ax.text(0.01, 0.86, "", transform=ax.transAxes, ha="left", va="top", fontsize=9,
                   color=MUTED, family="DejaVu Sans Mono", zorder=6)
ball = Circle((0, 0), R_BALL, fc=INK, ec="none", zorder=5)
ax.add_patch(ball)
spot = Circle((0, 0), 0.026, fc=PAPER, ec="none", zorder=6)
ax.add_patch(spot)
torso = Polygon(np.zeros((6, 2)), closed=True, fc=INK, ec=INK, lw=1.0, joinstyle="round", zorder=4)
ax.add_patch(torso)
SEGS = {name: ax.add_patch(Polygon(np.zeros((6, 2)), closed=True, fc=INK, ec=INK, lw=0.8,
                                   joinstyle="round", zorder=4))
        for name in ("thigh1", "calf1", "thigh2", "calf2", "upper", "fore", "neck")}
JOINTS = [ax.add_patch(Circle((0, 0), 0.01, fc=INK, ec="none", zorder=4)) for _ in range(7)]
head = Circle((0, 0), FIG * 0.074, fc=INK, ec="none", zorder=4)
ax.add_patch(head)
status_txt = fig.text(0.5, 0.375, "", ha="center", va="center", color=INK, fontsize=10)


def ik_knee(hip, foot, a, b, fwd):
    """Knee of a two-segment leg (thigh a, calf b) from hip to foot, bent towards `fwd`."""
    d = foot - hip
    D = max(np.hypot(*d), 1e-9)
    e = d / D
    if D >= a + b:
        return hip + a * e
    x = (a * a - b * b + D * D) / (2 * D)
    h = np.sqrt(max(a * a - x * x, 0.0))
    nv = np.array([-e[1], e[0]])
    if np.dot(nv, fwd) < 0:
        nv = -nv
    return hip + x * e + h * nv


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


# lower left: schematic space-time panel (best fault set so far)
axg = fig.add_axes([0.04, 0.07, 0.52, 0.26], facecolor=PAPER)
axg.set_xlim(-1.6, NCOL + 0.6)
axg.set_ylim(-1.75, NROW - 0.3)
axg.set_aspect("equal")
axg.axis("off")
gx, gy = np.meshgrid(np.arange(NCOL), np.arange(NROW))
axg.scatter(gx.ravel(), gy.ravel(), s=8, c=MUTED, lw=0, zorder=1)
for bxx in (-1, NCOL):
    axg.plot([bxx, bxx], [-0.4, NROW - 0.6], color=MUTED, lw=1.0, zorder=1)
axg.annotate("", xy=(-1.45, NROW - 0.6), xytext=(-1.45, -0.4),
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
axg.text(-1.6, (NROW - 1) / 2, "time", fontsize=7, color=MUTED, ha="right", va="center", rotation=90)
axg.annotate("", xy=(NCOL - 0.6, -1.0), xytext=(-0.4, -1.0),
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
axg.text((NCOL - 1) / 2, -1.25, "space", fontsize=7, color=MUTED, ha="center", va="top")
FAULT_LINES = LineCollection([], colors=ACCENT, linewidths=2.2, capstyle="round", zorder=3)
axg.add_collection(FAULT_LINES)
FAULT_DOTS = axg.scatter([], [], s=18, c=ACCENT, lw=0, zorder=4)
panel_txt = fig.text(0.30, 0.035, "", ha="center", va="center", color=INK, fontsize=9)

# lower right: current weight (grey) and best weight (black) over time
axt = fig.add_axes([0.63, 0.10, 0.33, 0.20], facecolor=PAPER)
axt.set_xlim(0, NF)
axt.set_ylim(31, 8.5)                              # lighter is up, as in the landscape
for sp in ("top", "right"):
    axt.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    axt.spines[sp].set_color(MUTED)
    axt.spines[sp].set_linewidth(0.8)
axt.set_xticks([])
axt.set_yticks([10, 20, 30])
axt.tick_params(axis="y", colors=MUTED, labelsize=7, length=2)
axt.set_xlabel("time", fontsize=7, color=MUTED, labelpad=2)
axt.set_ylabel("weight", fontsize=7, color=MUTED, labelpad=2)
cur_line, = axt.plot([], [], color=FOG, lw=1.0)
best_line, = axt.plot([], [], color=INK, lw=1.4, drawstyle="steps-post")
axt.legend([cur_line, best_line], ["current", "best"], loc="lower right", bbox_to_anchor=(1.0, 1.0),
           ncol=2, frameon=False, fontsize=7, handlelength=1.4, borderaxespad=0.1, labelcolor=MUTED)
FR = np.arange(NF)
BEST = np.array([f[2] if f[2] is not None else np.nan for f in frames], float)


def draw(k):
    sb, sf, bst, st, zoom = frames[k]
    c, ang = ball_center(sb)
    ball.center = c
    roll = -sb / R_BALL
    spot.center = c + 0.6 * R_BALL * np.array([np.cos(roll), np.sin(roll)])

    # camera
    full_w = (explored[k] - X0) + 0.6
    vw = VW + (full_w - VW) * zoom
    cx = CAMX[k] + ((X0 + explored[k]) / 2 + 0.25 - CAMX[k]) * zoom
    vh = vw * ASPECT
    cy = CAMY[k] + (YS.max() + 1.25 - vh / 2 - CAMY[k]) * zoom
    cx = min(max(cx, X0 + vw / 2), X1 - vw / 2)
    ax.set_xlim(cx - vw / 2, cx + vw / 2)
    ax.set_ylim(cy - vh / 2, cy + vh / 2)

    # the figure: pushing (arms at the boulder's centre) or walking after it (arms down)
    if sf is None:
        s_h, lean = push_hip(sb), push_lean(ang)
    else:
        _g, a_h = at_s(sf)
        s_h = sf
        lean = np.deg2rad(80) + 0.2 * a_h
    tdir = np.array([np.cos(lean), np.sin(lean)])
    hip, sh = body(s_h, lean)
    to_c = c - sh
    dist = np.hypot(*to_c)
    a_dir = to_c / dist
    phase = 9.0 * s_h
    if dist - R_BALL <= REACH * 1.03:
        hand = sh + (dist - R_BALL) * a_dir            # on the boulder's surface
        elbow = 0.5 * (sh + hand) - FIG * 0.018 * np.array([-a_dir[1], a_dir[0]])
        hd = sh + FIG * 0.12 * tdir + FIG * 0.03 * a_dir
    else:
        sw = -1.30 + 0.25 * np.sin(phase)              # arm swinging by the side
        hand = sh + FIG * L_ARM * 0.95 * np.array([np.cos(sw), np.sin(sw)])
        elbow = 0.5 * (sh + hand) + FIG * 0.02 * np.array([1.0, 0.0])
        hd = sh + FIG * 0.12 * tdir + FIG * 0.02 * np.array([1.0, 0.0])
    r_head = FIG * 0.074
    gap = np.hypot(*(hd - c)) - R_BALL - r_head
    if gap < 0.004:                                    # keep the head off the boulder
        hd = hd - (0.004 - gap) * (hd - c) / np.hypot(*(hd - c))
    s_b = s_h - 0.05                                   # back foot: the straight pushing leg
    for _ in range(80):
        fb, _a = at_s(s_b)
        if np.hypot(*(fb - hip)) >= (0.975 + 0.02 * np.sin(phase)) * LEG:
            break
        s_b -= 0.01
    f1 = fb
    _g, a_h = at_s(s_h)
    f2, _ = at_s(s_h + FIG * (0.13 - 0.05 * np.sin(phase)) * np.cos(a_h) ** 2)   # front foot
    u = np.array([np.cos(ang), np.sin(ang)])
    nrm = np.array([-np.sin(ang), np.cos(ang)])
    knee1 = ik_knee(hip, f1, FIG * L_THIGH, FIG * L_CALF, -u + nrm)
    knee = ik_knee(hip, f2, FIG * L_THIGH, FIG * L_CALF, u + 0.3 * nrm)
    perp = np.array([-tdir[1], tdir[0]])
    chest = hip + 0.62 * (sh - hip)
    torso.set_xy([sh + FIG * 0.064 * perp, chest + FIG * 0.088 * perp, hip + FIG * 0.050 * perp,
                  hip - FIG * 0.042 * perp, chest - FIG * 0.056 * perp, sh - FIG * 0.058 * perp])
    neck = sh + FIG * 0.05 * tdir
    limb("thigh1", hip, knee1, FIG * 0.050, FIG * 0.032, FIG * 0.058, 0.35)
    limb("calf1", knee1, f1, FIG * 0.030, FIG * 0.017, FIG * 0.040, 0.30)
    limb("thigh2", hip, knee, FIG * 0.050, FIG * 0.032, FIG * 0.058, 0.35)
    limb("calf2", knee, f2, FIG * 0.030, FIG * 0.017, FIG * 0.040, 0.30)
    limb("upper", sh, elbow, FIG * 0.040, FIG * 0.028, FIG * 0.046, 0.45)
    limb("fore", elbow, hand, FIG * 0.028, FIG * 0.020, FIG * 0.033, 0.35)
    limb("neck", sh, neck, FIG * 0.034, FIG * 0.030)
    for jc, (pt, r) in zip(JOINTS, [(hip, 0.050), (knee1, 0.031), (knee, 0.032), (sh, 0.058),
                                    (elbow, 0.028), (f1, 0.018), (f2, 0.018)]):
        jc.center, jc.radius = pt, FIG * r
    head.center = hd

    # explored terrain, weights of the extremes seen so far, flags on the records
    n_seen = int(np.searchsorted(XS, explored[k]))
    seen_line.set_data(XS[:n_seen], YS[:n_seen])
    for x, t in LABELS:
        t.set_visible(x <= explored[k] - 0.1)
    for w, (pole, cloth) in FLAGS.items():
        on = bst is not None and w >= bst and any(f[2] == w for f in frames[:k + 1])
        pole.set_visible(on)
        cloth.set_visible(on)
        cloth.set_facecolor(ACCENT if w == bst else MUTED)
    done = st == "done"
    readout.set_text("best w = " + (f"{bst}" if bst is not None else "-") + (" = d" if done else ""))
    readout.set_color(ACCENT if done else INK)
    readout2.set_text(f"current w = {int(round(w_cur[k]))}")
    status_txt.set_text(STATUS[st].format(b=bst))

    # lower panels
    if bst is not None:
        segs, dots = PANEL[bst]
        FAULT_LINES.set_segments(segs)
        FAULT_DOTS.set_offsets(dots if len(dots) else np.zeros((0, 2)))
        panel_txt.set_text(f"Lightest fault set so far: weight {bst}")
    else:
        FAULT_LINES.set_segments([])
        FAULT_DOTS.set_offsets(np.zeros((0, 2)))
        panel_txt.set_text("Lightest fault set so far: none yet")
    cur_line.set_data(FR[:k + 1], w_cur[:k + 1])
    best_line.set_data(FR[:k + 1], BEST[:k + 1])
    return []


if os.environ.get("STILL"):                       # STILL=k1,k2,...: PNG stills instead of the GIF
    for k in [int(v) for v in os.environ["STILL"].split(",")]:
        draw(min(k, NF - 1))
        fig.savefig(OUT.replace(".gif", "") + f"_f{k}.png", facecolor=PAPER)
    print("frames", NF)
    sys.exit(0)
# one shared palette for all frames (the drawing uses a handful of colours plus their
# anti-aliasing shades), no dithering, and every frame after the first stores only the pixels
# that changed (the rest transparent): a few times smaller than plain per-frame palettes
from PIL import Image
STEP = float(os.environ.get("STEP", 1.25))        # timeline frames per GIF frame (20 fps)
imgs = []
for k in np.unique(np.minimum(np.round(np.arange(0, NF, STEP)), NF - 1).astype(int)):
    draw(k)
    fig.canvas.draw()
    imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
sample = Image.new("RGB", (imgs[0].width, imgs[0].height * 4))
for j, k in enumerate(np.linspace(0, len(imgs) - 1, 4).astype(int)):
    sample.paste(imgs[k], (0, j * imgs[0].height))
NCOL_PAL = int(os.environ.get("COLORS", 40))
pal = sample.quantize(colors=NCOL_PAL, method=Image.Quantize.MEDIANCUT)
TRANSP = 255
pal_rgb = (pal.getpalette()[:3 * NCOL_PAL] + [0, 0, 0] * 256)[:768]
pal_rgb[3 * TRANSP:3 * TRANSP + 3] = [255, 0, 255]
idx_frames = [np.asarray(im.quantize(palette=pal, dither=Image.Dither.NONE)) for im in imgs]
out = []
prev = None
for a in idx_frames:
    b = a.copy()
    if prev is not None:
        b[a == prev] = TRANSP
    prev = a
    im = Image.fromarray(b, mode="P")
    im.putpalette(pal_rgb)
    im.info["transparency"] = TRANSP
    out.append(im)
out[0].save(OUT, save_all=True, append_images=out[1:], duration=int(round(1000 * STEP / FPS)),
            loop=0, optimize=False, disposal=1, transparency=TRANSP)
print("wrote", OUT, NF, "frames", flush=True)
