"""Turn saved vorticity frames into looping GIF banners (dark + light)."""
import sys
import numpy as np
from PIL import Image

src, outdir = sys.argv[1], sys.argv[2]
W_OUT = int(sys.argv[3]) if len(sys.argv) > 3 else 1100
LEVELS = int(sys.argv[4]) if len(sys.argv) > 4 else 112
frames = np.load(src)                     # (NF, NY, NX)
NF, NY, NX = frames.shape
CX, CY, AX, BY, SLOT = NX * 0.22, NY / 2, 11, 19, 3.0
X0, X1 = 70, NX - 6                       # crop
crop = frames[:, :, X0:X1]
h, w = crop.shape[1:]
H_OUT = int(round(W_OUT * h / w))

scale = np.percentile(np.abs(crop), 99.3)
v = np.clip(crop / scale, -1, 1)
v = np.sign(v) * np.abs(v) ** 0.65        # lift weak structures

# soft fade at left/right edges
xs = np.linspace(0, 1, w)
fade = np.clip(xs / 0.04, 0, 1) * np.clip((1 - xs) / 0.10, 0, 1)
v = v * fade[None, None, :]

def hex2rgb(hx):
    hx = hx.lstrip('#'); return np.array([int(hx[i:i+2], 16) for i in (0, 2, 4)], float)

def ramp(stops, n):
    """stops: list of hex, evenly spaced from 0..1 -> n colors"""
    cols = np.array([hex2rgb(s) for s in stops])
    t = np.linspace(0, 1, n) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    a = (t - i)[:, None]
    return cols[i] * (1 - a) + cols[i + 1] * a

THEMES = {
    "dark": dict(bg="#0d1117", obst="#8b949e",
                 neg=["#0d1117", "#123a63", "#1f78c8", "#5ec8f2", "#e3f6ff"],
                 pos=["#0d1117", "#5a1d0e", "#c2410c", "#fb923c", "#ffe7b3"]),
    "light": dict(bg="#ffffff", obst="#57606a",
                  neg=["#ffffff", "#cfe9fb", "#5aaee3", "#0b6aa8", "#0b3a5e"],
                  pos=["#ffffff", "#fde0c5", "#f59e5b", "#c2410c", "#6b1f0b"]),
}

# high-res obstacle alpha (3x supersampled), in output pixel coords
ss = 3
yy, xx = np.mgrid[0:H_OUT * ss, 0:W_OUT * ss].astype(float)
gx = X0 + (xx + 0.5) / (W_OUT * ss) * w
gy = (yy + 0.5) / (H_OUT * ss) * h
ell = ((gx - CX) / AX) ** 2 + ((gy - CY) / BY) ** 2 <= 1.0
slot = np.abs(gy - CY) <= SLOT / 2
alpha = (ell & ~slot).astype(float).reshape(H_OUT, ss, W_OUT, ss).mean((1, 3))[..., None]

for name, th in THEMES.items():
    half = LEVELS // 2
    lut = np.vstack([ramp(th["neg"], half)[::-1], ramp(th["pos"], half)[1:]])   # -1..+1
    n_lut = len(lut)
    obst_blend = ramp([th["bg"], th["obst"]], 256 - n_lut) if 256 - n_lut > 1 else np.array([hex2rgb(th["obst"])])
    pal_cols = np.vstack([lut, obst_blend]).clip(0, 255).astype(np.uint8)
    palimg = Image.new("P", (1, 1))
    palimg.putpalette(pal_cols.flatten().tolist() + [0] * (768 - pal_cols.size))
    obst_rgb = hex2rgb(th["obst"])

    out = []
    for k in range(NF):
        fld = Image.fromarray(v[k].astype(np.float32), mode="F").resize((W_OUT, H_OUT), Image.BICUBIC)
        a = np.clip(np.asarray(fld), -1, 1)
        idx = np.rint((a + 1) / 2 * (n_lut - 1)).astype(int)
        rgb = lut[idx]
        rgb = rgb * (1 - alpha) + obst_rgb * alpha
        im = Image.fromarray(rgb.clip(0, 255).astype(np.uint8), "RGB")
        out.append(im.quantize(palette=palimg, dither=Image.Dither.NONE))
    path = f"{outdir}/flow-{name}.gif"
    out[0].save(path, save_all=True, append_images=out[1:], duration=55, loop=0,
                optimize=False, disposal=1)
    out[NF // 3].convert("RGB").save(f"{outdir}/flow-{name}-still.png")
    import os
    print(name, f"{W_OUT}x{H_OUT}", f"{os.path.getsize(path)/1e6:.2f} MB", flush=True)
