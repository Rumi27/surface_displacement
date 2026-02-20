#!/usr/bin/env python3
import argparse, math, csv
from pathlib import Path
from PIL import Image, ImageDraw  # pillow

def load_image_size(stem, actual_dir):
    for ext in (".jpg",".jpeg",".png",".JPG",".PNG"):
        p = Path(actual_dir) / f"{stem}{ext}"
        if p.exists():
            with Image.open(p) as im:
                return p, im.size
    # fallback: try any file starting with stem
    cands = sorted(Path(actual_dir).glob(stem + "*"))
    if cands:
        with Image.open(cands[0]) as im:
            return cands[0], im.size
    raise FileNotFoundError(f"No image for stem {stem}")

def parse_yolo_txt(txt_path):
    rows = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:  # class xc yc w h [conf]
            continue
        cls, xc, yc, w, h = parts[:5]
        conf = float(parts[5]) if len(parts) >= 6 else 0.5
        rows.append((float(xc), float(yc), conf))
    return rows

def greedy_select(points_xyc, K, min_dist_px, W, H):
    # points_xyc: list of (xc_norm, yc_norm, conf)
    pts = sorted(points_xyc, key=lambda t: t[2], reverse=True)
    selected = []
    for xc, yc, conf in pts:
        x = xc * W; y = yc * H
        ok = True
        for xs, ys, _ in selected:
            if math.hypot(x - xs, y - ys) < min_dist_px:
                ok = False; break
        if ok:
            selected.append((x, y, conf))
            if len(selected) >= K:
                break
    # if we didn’t reach K, just return what we have (don’t force noisy picks)
    return selected

def save_csv(out_csv, sel):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_px","y_px","conf"])
        for x, y, c in sel:
            w.writerow([f"{x:.2f}", f"{y:.2f}", f"{c:.4f}"])

def make_overlay(img_path, sel, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(img_path).convert("RGB") as im:
        dr = ImageDraw.Draw(im)
        r = 8
        for x, y, _ in sel:
            dr.ellipse((x-r, y-r, x+r, y+r), outline=(0,255,0), width=2)
        im.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=["case3","case4","case5","case6"])
    ap.add_argument("--run_name", required=True, help="predict folder name under fold/eval_real (e.g., raw_c008)")
    ap.add_argument("--K", type=int, default=49)
    ap.add_argument("--min_dist_px", type=float, default=28.0)
    ap.add_argument("--actual_dir", default="actual_images")
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()

    fold = Path(args.fold)
    labels_dir = fold / "eval_real" / args.run_name / "labels"
    if not labels_dir.exists():
        raise SystemExit(f"Missing labels dir: {labels_dir}")

    out_dir = fold / "eval_real" / "centroids_select" / args.run_name
    ov_dir  = fold / "eval_real" / "overlays_select" / args.run_name

    for txt in sorted(labels_dir.glob("*.txt")):
        stem = txt.stem  # e.g. case3_before_actual(1)
        img_path, (W,H) = load_image_size(stem, args.actual_dir)
        preds = parse_yolo_txt(txt)
        sel = greedy_select(preds, args.K, args.min_dist_px, W, H)
        save_csv(out_dir / f"{stem}.csv", sel)
        if args.overlay:
            make_overlay(img_path, sel, ov_dir / f"{stem}_overlay.jpg")
        print(f"[{stem}] kept {len(sel)} (K={args.K}, min_dist={args.min_dist_px}px)")

if __name__ == "__main__":
    main()
