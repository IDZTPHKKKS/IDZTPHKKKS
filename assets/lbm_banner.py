"""Lattice-Boltzmann (D2Q9, BGK) flow past a slotted elliptical cylinder.
Renders a seamlessly looping vorticity GIF for a GitHub profile header,
in a dark-mode and a light-mode palette."""
import sys, time
import numpy as np
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
NX, NY = 640, 160
U0, RE = 0.08, 160
# obstacle: ellipse, semi-axes ax (streamwise) and by (cross-stream), slot along x
CX, CY, AX, BY, SLOT = NX * 0.22, NY / 2, 11, 19, 3.0
D = 2 * BY
NU = U0 * D / RE
TAU = 3 * NU + 0.5
print(f"tau={TAU:.4f} nu={NU:.4f}", flush=True)

c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
w = np.array([4/9] + [1/9]*4 + [1/36]*4, dtype=np.float32)
opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]

Y, X = np.mgrid[0:NY, 0:NX].astype(np.float32)
def obstacle_mask(Xg, Yg):
    ell = ((Xg - CX) / AX) ** 2 + ((Yg - CY) / BY) ** 2 <= 1.0
    slot = np.abs(Yg - CY) <= SLOT / 2
    return ell & ~slot
solid = obstacle_mask(X + 0.5, Y + 0.5)

def feq(rho, ux, uy):
    out = np.empty((9, NY, NX), dtype=np.float32)
    usq = 1.5 * (ux * ux + uy * uy)
    for i in range(9):
        cu = 3 * (c[i, 0] * ux + c[i, 1] * uy)
        out[i] = rho * w[i] * (1 + cu + 0.5 * cu * cu - usq)
    return out

rho = np.ones((NY, NX), np.float32)
ux = np.full((NY, NX), U0, np.float32)
uy = (0.01 * U0 * np.sin(2 * np.pi * Y / NY)).astype(np.float32)  # symmetry breaking
ux[solid] = 0; uy[solid] = 0
f = feq(rho, ux, uy)
f_in = feq(np.ones((NY, 1), np.float32), np.full((NY, 1), U0, np.float32), np.zeros((NY, 1), np.float32))[:, :, 0]

def step(f):
    rho = f.sum(0)
    ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho
    uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho
    fe = feq(rho, ux, uy)
    fpost = f - (f - fe) / TAU
    # bounce-back on solid nodes (uses pre-collision populations)
    for i in range(9):
        fpost[i][solid] = f[opp[i]][solid]
    for i in range(9):  # streaming, periodic in y
        fpost[i] = np.roll(np.roll(fpost[i], c[i, 0], axis=1), c[i, 1], axis=0)
    fpost[:, :, 0] = f_in               # inlet: equilibrium at U0
    fpost[:, :, -1] = fpost[:, :, -2]   # outlet: zero gradient
    return fpost, ux, uy

def vorticity(ux, uy):
    w_ = (np.roll(uy, -1, 1) - np.roll(uy, 1, 1)) / 2 - (np.roll(ux, -1, 0) - np.roll(ux, 1, 0)) / 2
    w_[solid] = 0
    w_[:, :2] = 0; w_[:, -3:] = 0
    return w_

# ---- warm-up until shedding is periodic
t0 = time.time()
WARM = 26000
probe = []
for n in range(WARM):
    f, ux, uy = step(f)
    if n >= WARM - 8000:
        probe.append(float(uy[int(CY), int(CX + 4 * AX + 40)]))
    if n % 2000 == 0:
        print(f"step {n} ({time.time()-t0:.0f}s)", flush=True)
p = np.array(probe) - np.mean(probe)
zc = np.where((p[:-1] < 0) & (p[1:] >= 0))[0]
T = float(np.mean(np.diff(zc))) if len(zc) > 2 else 1500.0
print(f"shedding period ~ {T:.1f} steps, amplitude {p.std():.4f}", flush=True)

# ---- capture ~3 periods at a fixed spacing, then cut at the best-matching frame
DT = max(1, int(round(T / 30)))
NCAP = 100
frames = []
for k in range(NCAP):
    for _ in range(DT):
        f, ux, uy = step(f)
    frames.append(vorticity(ux, uy).copy())
fr = np.stack(frames)
diffs = [np.abs(fr[k] - fr[0]).mean() for k in range(40, NCAP)]
kbest = 40 + int(np.argmin(diffs))
adj = np.abs(fr[1] - fr[0]).mean()
print(f"DT={DT} best loop length {kbest} frames, seam/adjacent = {min(diffs)/adj:.2f}", flush=True)
np.save(f"{OUT}/frames.npy", fr[:kbest])
print(f"saved ({time.time()-t0:.0f}s)", flush=True)
