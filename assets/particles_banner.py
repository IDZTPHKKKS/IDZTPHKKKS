"""Header animation: noise -> flow -> spiral -> noise (seamless loop).
Particles start scattered, get carried by a vortex-street flow, gather into a
gather into a spinning spiral distribution, then dissolve back to where they started."""
import sys, os
import numpy as np
from PIL import Image, ImageDraw

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
W, H = 1100, 312
SS = 2                                  # supersampling for splats
N = 3200
rng = np.random.default_rng(7)

# ---------------- timeline (frames)
F_NOISE, F_FLOW, F_MORPH, F_HOLD, F_DISS = 18, 58, 36, 40, 30
NF = F_NOISE + F_FLOW + F_MORPH + F_HOLD + F_DISS
DUR = 50  # ms per frame

# ---------------- target: a 3-arm spiral distribution (toy target for generative samplers)
CXs, CYs = W * 0.5, H * 0.5
arm = rng.integers(0, 3, N)
u = rng.uniform(0, 1, N) ** 0.8
rad = 8 + u * 132
ang = arm * 2 * np.pi / 3 + u * 3.4 + rng.normal(0, 0.10 + 0.18 * u, N)
target = np.stack([CXs + rad * np.cos(ang) * 1.9, CYs + rad * np.sin(ang) * 0.92], 1)
core = rng.random(N) < 0.10                                    # dense bright core
target[core] = np.stack([CXs + rng.normal(0, 12, core.sum()) * 1.9, CYs + rng.normal(0, 10, core.sum())], 1)
near = 1 - np.clip(np.hypot((target[:, 0] - CXs) / 1.9, target[:, 1] - CYs) / 150, 0, 1) * 0.75
perm = rng.permutation(N); target = target[perm]; near = near[perm]

# ---------------- start positions (noise)
P0 = np.stack([rng.uniform(0, W, N), rng.uniform(0, H, N)], 1)

# ---------------- vortex street velocity field
U = 2.2
vort = []
sp = 150.0
for k in range(-2, 10):
    vort.append((k * sp, H * 0.5 - 38, +1))
    vort.append((k * sp + sp / 2, H * 0.5 + 38, -1))
vort = np.array(vort, float)
GAM, CORE = 1300.0, 26.0
def velocity(p, t):
    x, y = p[:, 0:1], p[:, 1:2]
    vx0 = (vort[:, 0] + 1.3 * t) % (12 * sp) - 2 * sp
    dx = x - vx0[None, :]; dy = y - vort[None, :, 1]
    r2 = dx * dx + dy * dy + CORE * CORE
    s = vort[None, :, 2] * GAM / (2 * np.pi) / r2
    u = U + (-dy * s).sum(1); v = (dx * s).sum(1)
    return np.stack([u, v], 1)

def ease(s): return s * s * (3 - 2 * s)

def spin(pts, th):
    x = (pts[:, 0] - CXs) / 1.9; y = pts[:, 1] - CYs
    c, s_ = np.cos(th), np.sin(th)
    return np.stack([CXs + (c * x - s_ * y) * 1.9, CYs + s_ * x + c * y], 1)

# ---------------- simulate positions per frame
frames_pos, glow = [], []
p = P0.copy()
walk = np.zeros_like(P0)
for f in range(F_NOISE):                         # noise: brownian shimmer
    s = f / F_NOISE
    walk += rng.normal(0, 0.9, P0.shape)
    frames_pos.append(P0 + walk * np.sin(np.pi * 0.5 * s)); glow.append(0.0)
p = frames_pos[-1].copy()
t = 0.0
for f in range(F_FLOW):                          # flow: advection ramps up
    w_ = ease(min(1.0, f / 14))
    for _ in range(3):
        p = p + w_ * velocity(p, t) * 1.0
        t += 1.0
    p[:, 0] %= W; p[:, 1] = np.clip(p[:, 1], 0, H - 1)
    frames_pos.append(p.copy()); glow.append(0.0)
p_flow = p.copy()
swirl_dir = rng.normal(0, 1, (N, 2))
for f in range(F_MORPH):                         # gather into the spiral, still drifting with the flow
    s = ease((f + 1) / F_MORPH)
    for _ in range(3):
        p_flow = p_flow + (1 - s) * velocity(p_flow, t); t += 1.0
    p_flow[:, 0] %= W
    pos = (1 - s) * p_flow + s * target + np.sin(np.pi * s) * 14 * swirl_dir
    frames_pos.append(pos); glow.append(s)
SPIN = 0.035
for f in range(F_HOLD):                          # hold: the spiral spins
    frames_pos.append(spin(target, SPIN * (f + 1))); glow.append(1.0)
target_end = spin(target, SPIN * F_HOLD)
for f in range(F_DISS):                          # dissolve back to the start (forward diffusion)
    s = ease((f + 1) / F_DISS)
    pos = (1 - s) * spin(target_end, SPIN * (f + 1) * (1 - s)) + s * P0 + np.sin(np.pi * s) * 22 * rng.normal(0, 1, P0.shape)
    frames_pos.append(pos); glow.append(1 - s)
assert len(frames_pos) == NF

# ---------------- colours
def hx(h): h = h.lstrip('#'); return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], float)
def grad(stops, t):
    c = np.array([hx(s) for s in stops]); t = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2); a = (t - i)[:, None]
    return c[i] * (1 - a) + c[i + 1] * a
key = (np.arctan2(target[:, 1] - CYs, (target[:, 0] - CXs) / 1.9) / (2 * np.pi)) % 1.0
key = 1 - np.abs(2 * key - 1)   # wrap smoothly

THEMES = {
    "dark":  dict(bg=hx("#0d1117"), stops=["#38bdf8", "#818cf8", "#f472b6", "#fb923c"], add=True),
    "light": dict(bg=hx("#ffffff"), stops=["#0369a1", "#4f46e5", "#c026d3", "#ea580c"], add=False),
}

# gaussian splat kernel
R = 3
ky, kx = np.mgrid[-R:R+1, -R:R+1]
KER = np.exp(-(kx**2 + ky**2) / (2 * 1.45**2))

def render(pos, cols, theme, g, prev=None):
    if prev is not None:
        d = pos - prev
        okk = (np.abs(d[:, 0]) < 60) & (np.abs(d[:, 1]) < 60)
        pts = [pos] + [pos - d * okk[:, None] * (k / 4.0) * 1.6 for k in range(1, 5)]
        pos = np.concatenate(pts); cols = np.concatenate([cols] * 5)
    Hs, Ws = H * SS, W * SS
    acc = np.zeros((Hs + 2*R + 2, Ws + 2*R + 2, 3)); wsum = np.zeros((Hs + 2*R + 2, Ws + 2*R + 2))
    xs = np.rint(pos[:, 0] * SS).astype(int) + R; ys = np.rint(pos[:, 1] * SS).astype(int) + R
    ok = (xs >= R) & (xs < Ws + R) & (ys >= R) & (ys < Hs + R)
    xs, ys, cc = xs[ok], ys[ok], cols[ok]
    inten = 0.9 + 0.3 * g
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            k = KER[dy + R, dx + R] * inten
            np.add.at(wsum, (ys + dy, xs + dx), k)
            for ch in range(3):
                np.add.at(acc[..., ch], (ys + dy, xs + dx), k * cc[:, ch])
    acc = acc[R:R+Hs, R:R+Ws]; wsum = wsum[R:R+Hs, R:R+Ws]
    col = acc / np.maximum(wsum, 1e-6)[..., None]
    a = 1 - np.exp(-wsum * 1.6)
    if theme["add"]:
        col = col * (1 + 0.5 * np.clip(wsum - 1.2, 0, 2)[..., None] * 0.35)   # brighter where dense
    img = theme["bg"] * (1 - a[..., None]) + np.clip(col, 0, 255) * a[..., None]
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGB")
    return im.resize((W, H), Image.LANCZOS)

for name, th in THEMES.items():
    cols = grad(th["stops"], key)
    shade = (0.45 + 0.55 * near)[:, None]
    cols = cols * shade + th["bg"] * (1 - shade)
    ims = [render(frames_pos[i], cols, th, glow[i],
                  frames_pos[i - 1] if F_NOISE <= i < F_NOISE + F_FLOW + F_MORPH + F_HOLD else None) for i in range(NF)]
    # one shared palette from a few sample frames
    path = f"{OUT}/hero-{name}.webp"
    small = [im.resize((880, int(round(880 * H / W))), Image.LANCZOS) for im in ims]
    small[0].save(path, save_all=True, append_images=small[1:], duration=DUR, loop=0, quality=58, method=6)
    for tag, i in [("noise", 3), ("flow", F_NOISE + F_FLOW - 1), ("spiral", F_NOISE + F_FLOW + F_MORPH + 12)]:
        ims[i].save(f"{OUT}/still-{name}-{tag}.png")
    print(name, NF, "frames", f"{os.path.getsize(path)/1e6:.2f} MB", flush=True)
