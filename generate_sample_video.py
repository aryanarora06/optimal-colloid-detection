"""
generate_sample_video.py
------------------------
Generates a synthetic phase-contrast microscopy video of drifting / Brownian
colloid particles.  No labels, no annotations — raw grayscale frames, just
like real microscopy footage, ready to feed into analyze_video.py.

Output: sample_microscopy.mp4  (in the same directory)
Usage:  python generate_sample_video.py
"""

import cv2
import numpy as np
import math
import os

# ── Config ────────────────────────────────────────────────────────
WIDTH, HEIGHT = 512, 512
FPS           = 15
N_FRAMES      = 150          # 10 seconds
N_PARTICLES   = 35
MIN_R, MAX_R  = 7, 14        # particle radii in pixels
BROWNIAN_STD  = 1.2          # pixels per frame (random walk step)
DRIFT_VEL     = (0.3, 0.15)  # gentle global flow (dx, dy) px/frame
OUTPUT        = "sample_microscopy.mp4"


# ── Phase-contrast renderer (matches notebook style) ──────────────
def render_phase_contrast(image, cx, cy, r, intensity, defocus):
    pad  = int(r * 4.0)
    x0   = max(0, cx - pad);  x1 = min(WIDTH,  cx + pad + 1)
    y0   = max(0, cy - pad);  y1 = min(HEIGHT, cy + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return

    xs = np.arange(x0, x1, dtype=np.float32) - cx
    ys = np.arange(y0, y1, dtype=np.float32) - cy
    xx, yy = np.meshgrid(xs, ys)
    rr  = np.sqrt(xx**2 + yy**2)
    rn  = rr / r

    dark_thickness = 0.12 + 0.15 * defocus
    dark_ring      = -0.8 * intensity * np.exp(-((rn - 0.95)**2) /
                                                (2 * dark_thickness**2))
    bright_center  = 0.6  * intensity * (1.0 - defocus * 0.3) * \
                     np.exp(-(rn**2) / (2 * 0.25**2))
    fringe_freq    = 3.0 + 1.5 * defocus
    fringe_decay   = 0.3 + 0.2 * defocus
    fringes        = (0.2 * intensity
                      * np.cos((rn - 1.0) * fringe_freq * math.pi)
                      * np.exp(-(rn - 1.0) / fringe_decay))
    fringes        = np.where(rn > 1.0, fringes, 0.0)

    transmission = np.clip(1.0 + dark_ring + bright_center + fringes, 0.05, 2.5)
    edge_blend   = np.clip((4.0 - rn) / 0.8, 0.0, 1.0)
    transmission = 1.0 + (transmission - 1.0) * edge_blend
    roi          = rn < 4.0

    patch = image[y0:y1, x0:x1]
    patch[roi] = np.clip(patch[roi] * transmission[roi], 0.0, 1.0)


# ── Initialise particles ──────────────────────────────────────────
rng = np.random.default_rng(42)

particles = []
for _ in range(N_PARTICLES):
    r         = rng.integers(MIN_R, MAX_R + 1)
    cx        = rng.uniform(r * 4, WIDTH  - r * 4)
    cy        = rng.uniform(r * 4, HEIGHT - r * 4)
    intensity = rng.uniform(0.55, 0.95)
    defocus   = rng.uniform(0.10, 0.55)
    # per-particle random-walk state
    particles.append(dict(cx=cx, cy=cy, r=int(r),
                          intensity=intensity, defocus=defocus))


# ── Video writer ──────────────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT, fourcc, FPS, (WIDTH, HEIGHT))

print(f"Generating {N_FRAMES} frames  ({WIDTH}x{HEIGHT} @ {FPS} fps)…")

for frame_idx in range(N_FRAMES):
    # Background: uniform grey + subtle Gaussian noise
    bg    = np.full((HEIGHT, WIDTH), 0.82, dtype=np.float32)
    noise = rng.normal(0, 0.015, bg.shape).astype(np.float32)
    img   = np.clip(bg + noise, 0.0, 1.0)

    # Render each particle
    for p in particles:
        render_phase_contrast(img, int(p['cx']), int(p['cy']),
                              p['r'], p['intensity'], p['defocus'])

        # Move: Brownian + global drift, wrap at borders
        p['cx'] = (p['cx'] + DRIFT_VEL[0]
                   + rng.normal(0, BROWNIAN_STD)) % WIDTH
        p['cy'] = (p['cy'] + DRIFT_VEL[1]
                   + rng.normal(0, BROWNIAN_STD)) % HEIGHT

    # Convert to 8-bit BGR for VideoWriter
    frame_8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_8, cv2.COLOR_GRAY2BGR)
    out.write(frame_bgr)

    if (frame_idx + 1) % 30 == 0:
        print(f"  frame {frame_idx + 1}/{N_FRAMES}")

out.release()
print(f"\nDone!  Saved → {os.path.abspath(OUTPUT)}")
print(f"  Particles : {N_PARTICLES}")
print(f"  Duration  : {N_FRAMES / FPS:.1f} s  ({N_FRAMES} frames @ {FPS} fps)")
print(f"  Resolution: {WIDTH}x{HEIGHT}")
