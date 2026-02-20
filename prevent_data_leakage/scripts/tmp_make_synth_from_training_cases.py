#!/usr/bin/env python3
import argparse, random
from pathlib import Path
import cv2
import numpy as np

def rand_affine(img, max_rot=12, max_scale=0.1, max_shift=0.05):
    h,w = img.shape[:2]
    ang = random.uniform(-max_rot, max_rot)
    sc  = 1.0 + random.uniform(-max_scale, max_scale)
    tx  = random.uniform(-max_shift, max_shift) * w
    ty  = random.uniform(-max_shift, max_shift) * h
    M = cv2.getRotationMatrix2D((w/2,h/2), ang, sc)
    M[:,2] += [tx, ty]
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def jitter(img):
    # brightness/contrast/noise/jpeg-like blur
    alpha = 1.0 + random.uniform(-0.15, 0.15)   # contrast
    beta  = random.uniform(-20, 20)             # brightness
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if random.random()<0.5:
        out = cv2.GaussianBlur(out,(3,3),0)
    if random.random()<0.4:
        noise = np.random.normal(0, 8, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16)+noise,0,255).astype(np.uint8)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="case3/gan_train_input")   # cases 4–6 live here
    ap.add_argument("--out", default="case3/synth/images")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    src = Path(args.src); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pool = [p for p in src.glob("*.*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    assert pool, f"No images in {src}"
    rng = random.Random(20250901)
    for i in range(args.n):
        p = rng.choice(pool)
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None: continue
        im = rand_affine(im)
        im = jitter(im)
        cv2.imwrite(str(out / f"synth_{i:05d}.jpg"), im, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(80,95)])
    print(f"[OK] Wrote {args.n} augmented images to {out}")

if __name__ == "__main__":
    main()
