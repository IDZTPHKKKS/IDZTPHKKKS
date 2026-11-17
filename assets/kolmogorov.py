"""Forced 2D Kolmogorov flow (pseudo-spectral, vorticity form) -> looping vorticity GIF.
w_t + u.grad(w) = nu lap(w) - alpha w + n cos(n y)"""
import sys, numpy as np
from PIL import Image

out = sys.argv[1]
NY, NX = 128, 256
L = 2 * np.pi
nu, alpha, nf = 1 / 120, 0.1, 4
dt = 0.01
kx = np.fft.fftfreq(NX, 1 / NX) / 2   # Lx = 4*pi
ky = np.fft.fftfreq(NY, 1 / NY)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2; K2i = np.where(K2 == 0, 1, K2)
dealias = (np.abs(KX) < NX / 6) & (np.abs(KY) < NY / 3)
X, Y = np.meshgrid(np.arange(NX) * 2 * L / NX, np.arange(NY) * L / NY)
F = np.fft.fft2(nf * np.cos(nf * Y))
rng = np.random.default_rng(3)
wh = np.fft.fft2(rng.normal(0, 1, (NY, NX))) * dealias * np.exp(-K2 / 50)

def rhs(wh):
    psih = wh / K2i; psih[0, 0] = 0
    u = np.real(np.fft.ifft2(1j * KY * psih)); v = np.real(np.fft.ifft2(-1j * KX * psih))
    wx = np.real(np.fft.ifft2(1j * KX * wh)); wy = np.real(np.fft.ifft2(1j * KY * wh))
    nl = np.fft.fft2(u * wx + v * wy) * dealias
    return -nl - nu * K2 * wh - alpha * wh + F

def step(wh):
    a = rhs(wh); b = rhs(wh + dt / 2 * a); c = rhs(wh + dt / 2 * b); d = rhs(wh + dt * c)
    return wh + dt / 6 * (a + 2 * b + 2 * c + d)

for i in range(4000):
    wh = step(wh)
frames = []
for i in range(48):
    for _ in range(12):
        wh = step(wh)
    frames.append(np.real(np.fft.ifft2(wh)))
fr = np.stack(frames)
s = np.percentile(np.abs(fr), 99)

def hx(h): h = h.lstrip('#'); return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], float)
def ramp(stops, n):
    c = np.array([hx(t) for t in stops]); t = np.linspace(0, 1, n) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2); a = (t - i)[:, None]
    return c[i] * (1 - a) + c[i + 1] * a
half = 110
lut = np.vstack([ramp(["#0d1117", "#123a63", "#1f78c8", "#5ec8f2", "#e3f6ff"], half)[::-1],
                 ramp(["#0d1117", "#5a1d0e", "#c2410c", "#fb923c", "#ffe7b3"], half)[1:]])
pal = lut.clip(0, 255).astype(np.uint8)
palimg = Image.new("P", (1, 1)); palimg.putpalette(pal.flatten().tolist() + [0] * (768 - pal.size))
ims = []
for w in fr:
    v = np.clip(w / s, -1, 1); v = np.sign(v) * np.abs(v) ** 0.8
    big = np.asarray(Image.fromarray(v.astype(np.float32), "F").resize((520, 260), Image.BICUBIC))
    idx = np.rint((np.clip(big, -1, 1) + 1) / 2 * (len(lut) - 1)).astype(int)
    im = Image.fromarray(lut[idx].clip(0, 255).astype(np.uint8), "RGB")
    ims.append(im)
ims[0].save(out, save_all=True, append_images=ims[1:], duration=80, loop=0, quality=70, method=6)
import os; print(f"{os.path.getsize(out)/1e6:.2f} MB, max|w|={np.abs(fr).max():.1f}")
