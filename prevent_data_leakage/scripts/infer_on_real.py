#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
from PIL import Image  # pip install pillow

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=["case3","case4","case5","case6"])
    ap.add_argument("--actual_dir", default="actual_images")
    ap.add_argument("--run_name", default="raw")  # which predict folder under eval_real
    args = ap.parse_args()

    fold = Path(args.fold)
    labels_dir = fold / "eval_real" / args.run_name / "labels"
    out_dir    = fold / "eval_real" / "centroids"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not labels_dir.exists():
        raise SystemExit(f"Missing predictions at {labels_dir}. Run yolo predict with save_txt=True first.")

    for txt in sorted(labels_dir.glob("*.txt")):
        stem = txt.stem  # e.g., case3_before_actual(1)
        img_path = None
        for ext in (".jpg",".jpeg",".png"):
            p = Path(args.actual_dir) / f"{stem}{ext}"
            if p.exists():
                img_path = p; break
        if img_path is None:
            cands = list(Path(args.actual_dir).glob(f"{stem}*"))
            if cands: img_path = cands[0]
            else:
                print(f"[WARN] No matching image for {txt.name}; skipping.")
                continue

        W, H = Image.open(img_path).size

        rows = []
        for line in txt.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5: 
                continue
            _, xc, yc, w, h = parts[:5]  # class, xc, yc, w, h (normalized)
            xc, yc = float(xc), float(yc)
            rows.append((xc * W, yc * H))

        out_csv = out_dir / (stem + ".csv")
        with out_csv.open("w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["x_px","y_px"])
            cw.writerows(rows)
    print(f"[OK] Wrote centroid CSVs to {out_dir}")

if __name__ == "__main__":
    main()
