#!/usr/bin/env python3
import argparse, glob
from pathlib import Path
import cv2
import numpy as np

def detect(gray, dp=1.2, minDist=12, param1=100, param2=25, minRadius=3, maxRadius=60):
    cs = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                          param1=param1, param2=param2,
                          minRadius=minRadius, maxRadius=maxRadius)
    return [] if cs is None else np.round(cs[0]).astype(int)  # (x,y,r)

def process(img_path, out_dir):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] cannot read {img_path}")
        return []
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # try a few thresholds to get circles
    circles = []
    for p2 in (25, 22, 20, 18):
        circles = detect(gray, dp=1.2, minDist=12, param1=100, param2=p2, minRadius=3, maxRadius=60)
        if len(circles) > 0:
            break

    if len(circles) == 0:
        print(f"[INFO] no circles found in {img_path}")
        return []

    # save an overlay for sanity check
    vis = img.copy()
    for (x,y,r) in circles[:200]:
        cv2.circle(vis, (x,y), r, (0,255,0), 2)
        cv2.circle(vis, (x,y), 2, (0,0,255), 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (Path(img_path).stem + "_overlay.jpg")
    cv2.imwrite(str(out_file), vis)
    print(f"[OK] overlay -> {out_file}")

    return [int(r) for (_,_,r) in circles]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="actual_images/case3_*_actual*.jpg")
    ap.add_argument("--out_dir", default="case3/eval_real/overlays")
    args = ap.parse_args()

    r_all = []
    for img_path in glob.glob(args.pattern):
        r_all += process(img_path, Path(args.out_dir))

    if not r_all:
        print("[FAIL] no radii measured. Try a different image or relax thresholds.")
        raise SystemExit(1)

    r_all = np.array(r_all)
    r_med = float(np.median(r_all))
    r_mean = float(np.mean(r_all))
    r_min, r_max = int(np.min(r_all)), int(np.max(r_all))

    # recommend params
    min_r = max(3, int(round(0.80 * r_med)))
    max_r = int(round(1.30 * r_med))
    min_dist = max(8, int(round(1.20 * r_med)))

    print("\n=== Radius summary (pixels) ===")
    print(f"count={len(r_all)}  median={r_med:.2f}  mean={r_mean:.2f}  min={r_min}  max={r_max}")
    print("\n=== Recommended auto-label params ===")
    print(f"--min_r {min_r} --max_r {max_r} --min_dist {min_dist} --param1 100 --param2 26 --blur 5")
    print("(Too many circles? raise --param2 to 28/30. Too few? lower to 24/22.)")

if __name__ == "__main__":
    main()
