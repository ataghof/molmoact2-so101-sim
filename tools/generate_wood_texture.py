"""Regenerate the bundled wood-table texture from scratch (clean-room, deterministic).

The table texture used by `molmoact2_so101_sim.realism` is original procedural work: this
script generates it from numpy noise alone, with no third-party source image, and writes a
bare PNG with no text/metadata chunks. Output is distributable under the repository's
Apache-2.0 license. Run it to reproduce or tweak the asset:

    python tools/generate_wood_texture.py src/molmoact2_so101_sim/assets/wood_table.png

Requires numpy (a core dependency) and pillow (in the `sim` extra).
"""
import sys

import numpy as np
from PIL import Image

N = 1024
PLANKS = 4
rng = np.random.default_rng(20260715)


def tileable_noise(cells, octaves=5, persist=0.5):
    """Smooth, seamlessly tileable value noise in [0,1]: wrapped low-res lattices upsampled
    bicubically and summed over octaves."""
    acc = np.zeros((N, N), np.float64)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = cells * (2 ** o)
        lo = rng.standard_normal((c + 1, c + 1))
        lo[-1, :] = lo[0, :]              # wrap edges so the upsample tiles
        lo[:, -1] = lo[:, 0]
        up = np.asarray(Image.fromarray(lo).resize((N, N), Image.BICUBIC), np.float64)
        acc += amp * up
        tot += amp
        amp *= persist
    acc /= tot
    return (acc - acc.min()) / (np.ptp(acc) + 1e-9)


def main(out):
    y, x = np.mgrid[0:N, 0:N].astype(np.float64)
    u, v = x / N, y / N

    plank_id = np.clip((v * PLANKS).astype(int), 0, PLANKS - 1)
    plank_hue = rng.uniform(-0.05, 0.05, PLANKS)[plank_id]
    plank_phase = rng.uniform(0, 2 * np.pi, PLANKS)[plank_id]
    plank_stretch = rng.uniform(0.85, 1.2, PLANKS)[plank_id]

    warp = tileable_noise(6) - 0.5
    grain = (0.5 + 0.5 * np.sin(2 * np.pi * (u * plank_stretch * 34.0 + plank_phase + warp * 4.0))) ** 1.5
    fig = tileable_noise(3)
    mottle = tileable_noise(12)

    base_light = np.array([0.54, 0.37, 0.21])
    base_dark = np.array([0.28, 0.165, 0.085])
    t = np.clip(0.58 * grain + 0.34 * fig + 0.12 * mottle - 0.02, 0, 1)[..., None]
    rgb = base_dark + (base_light - base_dark) * t
    rgb = rgb * (1.0 + plank_hue[..., None] * np.array([1.0, 0.6, 0.25]))

    band_v = (v * PLANKS) % 1.0
    seam = np.clip(1.0 - np.minimum(band_v, 1.0 - band_v) / 0.012, 0, 1) ** 2
    rgb *= (1.0 - 0.5 * seam[..., None])
    rgb += rng.standard_normal((N, N, 1)) * 0.008
    rgb = np.clip(rgb, 0, 1)

    img = Image.fromarray((rgb * 255 + 0.5).astype(np.uint8), mode="RGB")
    img.save(out, format="PNG", optimize=True)          # no pnginfo -> no metadata chunks
    print("wrote", out, img.size)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "src/molmoact2_so101_sim/assets/wood_table.png")
