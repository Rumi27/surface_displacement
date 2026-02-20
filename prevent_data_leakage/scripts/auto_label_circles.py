#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np

def detect_circles(gray, dp=1.2, minDist=12, param1=80, param2=18, minRadius=3, maxRadius=20):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    if circles is None: 
        return []
    return np.round(circles[0, :]).astype("int")  # (x,y,r)

def yolo_line(xc, yc, w, h, W, H):
    return f"0 {xc/W:.6f} {yc/H:.6f} {w/W:.6f} {h/H:.6f}\n"

def process_image(img_path, out_txt, args):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None: 
        return 0
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.blur > 0:
        gray = cv2.GaussianBlur(gray, (args.blur, args.blur), 0)

    cs = detect_circles(
        gray,
        dp=args.dp,
        minDist=args.min_dist,
        param1=args.param1,
        param2=args.param2,
        minRadius=args.min_r,
        maxRadius=args.max_r
    )
    if len(cs) == 0:
        return 0

    lines = []
    for (x, y, r) in cs:
        w = h = max(2*r, 4)  # square box around circle, min 4 px
        lines.append(yolo_line(x, y, w, h, W, H))

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.writelines(lines)
    return len(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--exts", default=".jpg,.jpeg,.png")
    ap.add_argument("--blur", type=int, default=3)
    ap.add_argument("--dp", type=float, default=1.2)
    ap.add_argument("--min_dist", type=int, default=12)
    ap.add_argument("--param1", type=int, default=80)
    ap.add_argument("--param2", type=int, default=18)
    ap.add_argument("--min_r", type=int, default=3)
    ap.add_argument("--max_r", type=int, default=20)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    exts = {e.strip().lower() for e in args.exts.split(",")}
    imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])

    total_imgs = 0
    total_dets = 0
    for img in imgs:
        txt = labels_dir / (img.stem + ".txt")
        n = process_image(img, txt, args)
        total_imgs += 1
        total_dets += n
    print(f"[OK] Labeled {total_imgs} images, total {total_dets} markers.")

if __name__ == "__main__":
    main()
