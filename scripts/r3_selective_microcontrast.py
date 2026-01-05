#!/usr/bin/env python3
import os
import cv2
import numpy as np

def env_float(name, default):
    v = os.environ.get(name, str(default)).strip()
    try:
        return float(v)
    except:
        return float(default)

def env_int(name, default):
    v = os.environ.get(name, str(default)).strip()
    try:
        return int(v)
    except:
        return int(default)

img_in = os.environ["IMG_IN"]
out = os.environ["OUT"]

# Strength knobs
MC = env_float("MC", 0.06)              # microcontrast amount (start 0.06–0.08)
EDGE_PROTECT = env_float("EDGE_PROTECT", 0.75)  # 0..1 (higher = protect edges more)
SKY_PROTECT = env_float("SKY_PROTECT", 0.85)    # 0..1 (higher = protect sky more)
HIGHLIGHT_PROTECT = env_float("HIGHLIGHT_PROTECT", 0.60)  # protect bright reflections

# Edge mask settings
CANNY1 = env_int("CANNY1", 70)
CANNY2 = env_int("CANNY2", 140)
EDGE_DILATE = env_int("EDGE_DILATE", 2)

img = cv2.imread(img_in, cv2.IMREAD_COLOR)
if img is None:
    raise RuntimeError(f"Failed to read image: {img_in}")

# --- Build SKY mask (HSV: blue-ish + bright-ish) ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

# Heuristic sky: hue in blue range + moderate saturation + high value
sky = ((H >= 85) & (H <= 135) & (S >= 20) & (V >= 140)).astype(np.uint8) * 255
sky = cv2.GaussianBlur(sky, (0, 0), 3)

# --- Build EDGE mask (protect long straight edges + grid lines) ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, CANNY1, CANNY2)
if EDGE_DILATE > 0:
    k = np.ones((EDGE_DILATE*2+1, EDGE_DILATE*2+1), np.uint8)
    edges = cv2.dilate(edges, k, iterations=1)
edges = cv2.GaussianBlur(edges, (0, 0), 1.5)

# --- Highlight mask (protect bright reflections / speculars) ---
# Bright areas in value channel
high = (V >= 220).astype(np.uint8) * 255
high = cv2.GaussianBlur(high, (0, 0), 2.0)

# Combine protection masks into one "do-not-sharpen" map
protect = np.zeros_like(gray, dtype=np.float32)

protect += (edges.astype(np.float32) / 255.0) * EDGE_PROTECT
protect += (sky.astype(np.float32) / 255.0) * SKY_PROTECT
protect += (high.astype(np.float32) / 255.0) * HIGHLIGHT_PROTECT

protect = np.clip(protect, 0.0, 1.0)

# "apply" mask is inverse of protect
apply_mask = 1.0 - protect

# --- Luminance-only sharpening in LAB ---
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# Unsharp on L only
# Classic unsharp: L + amount*(L - blur(L))
blur = cv2.GaussianBlur(L, (0, 0), 2.0)
detail = cv2.subtract(L, blur)

sharpened_L = L.astype(np.float32) + (MC * 255.0) * (detail.astype(np.float32) / 255.0)

# Apply selectively (don’t sharpen protected areas)
sharpened_L = (L.astype(np.float32) * (1.0 - apply_mask) + sharpened_L * apply_mask)

sharpened_L = np.clip(sharpened_L, 0, 255).astype(np.uint8)

lab2 = cv2.merge([sharpened_L, A, B])
out_img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
ok = cv2.imwrite(out, out_img)
if not ok:
    raise RuntimeError(f"Failed to write output: {out}")

print(f"Saved: {out}")
