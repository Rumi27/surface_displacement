#!/usr/bin/env python3
import random, shutil, sys
from pathlib import Path

SEED = 20250901
TRAIN_FRAC = 0.85

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"case3","case4","case5","case6"}:
        print("Usage: python scripts/gen_split_from_synth.py case3|case4|case5|case6")
        raise SystemExit(1)
    fold = Path(sys.argv[1]).resolve()
    src_img_dir = fold / "synth" / "images"
    src_lbl_dir = fold / "labels"
    dst_tr_img = fold / "images" / "train"
    dst_vl_img = fold / "images" / "val"
    dst_tr_lbl = fold / "labels" / "train"
    dst_vl_lbl = fold / "labels" / "val"

    for d in [dst_tr_img, dst_vl_img, dst_tr_lbl, dst_vl_lbl]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True)

    imgs = sorted([p for p in src_img_dir.glob("*.*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    if not imgs:
        raise SystemExit(f"No synthetic images found in {src_img_dir}")
    random.Random(SEED).shuffle(imgs)
    n_train = int(len(imgs)*TRAIN_FRAC)
    train_imgs = imgs[:n_train]
    val_imgs   = imgs[n_train:]

    def copy_pair(img, dst_img_dir, dst_lbl_dir):
        stem = img.stem
        lbl = src_lbl_dir / f"{stem}.txt"
        if not lbl.exists():
            raise FileNotFoundError(f"Missing label for {img.name} at {lbl}")
        shutil.copy2(img, dst_img_dir / img.name)
        shutil.copy2(lbl, dst_lbl_dir / lbl.name)

    for im in train_imgs: copy_pair(im, dst_tr_img, dst_tr_lbl)
    for im in val_imgs:   copy_pair(im, dst_vl_img, dst_vl_lbl)

    print(f"[OK] Train: {len(train_imgs)}, Val: {len(val_imgs)}")

if __name__ == "__main__":
    main()
