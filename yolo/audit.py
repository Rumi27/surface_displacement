#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO data separation audit for marker detection & displacement pipeline.
Run from: ~/Desktop/surface_displacement/yolo

Outputs:
- Console summary
- yolo/leakage_audit_report.csv  (key counts + overlap lists)
"""

import os
import re
import csv
from pathlib import Path
from typing import List, Set, Tuple

HERE = Path.cwd()

# ---- Config: where to look ----
ACTUAL_DIRS = [
    HERE / "actual_images",
    HERE / "actual_measurment",  # if you also kept originals here
]

# Common places your training split may be defined
POTENTIAL_LIST_FILES = [
    HERE / "train_list.txt",
    HERE / "val_list.txt",
    HERE.parent / "train_list.txt",
    HERE.parent / "val_list.txt",
]

DATA_YAML = HERE / "data.yaml"

# Where we might find images used for inference / displacement calc
INFERENCE_CANDIDATE_DIRS = [
    HERE / "results",
    HERE / "results_hungarian",
    HERE / "results_mapped",
    HERE / "real_8",
    HERE / "new_frames",
    HERE / "runs" / "detect",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def read_lines(path: Path) -> List[str]:
    try:
        return [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    except Exception:
        return []


def collect_images_recursively(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and is_img(p)]


def parse_data_yaml(yaml_path: Path) -> Tuple[List[Path], List[Path]]:
    """
    Very light YAML parser (no external deps). Looks for lines starting with 'train:' and 'val:'.
    If they point to .txt -> read paths from the file. If they point to a folder -> glob images.
    """
    train_images, val_images = [], []
    if not yaml_path.exists():
        return train_images, val_images

    lines = read_lines(yaml_path)
    kv = {}
    pat = re.compile(r"^\s*(train|val)\s*:\s*(.+?)\s*$")
    for ln in lines:
        m = pat.match(ln)
        if m:
            key, val = m.group(1), m.group(2).strip().strip("'\"")
            kv[key] = val

    for key in ("train", "val"):
        if key not in kv:
            continue
        path = Path(kv[key]).expanduser()
        if not path.is_absolute():
            # Interpret relative to data.yaml location
            path = yaml_path.parent / path

        paths: List[Path] = []
        if path.suffix.lower() == ".txt" and path.exists():
            # Each line is an image path (or directory)
            for ln in read_lines(path):
                p = Path(ln).expanduser()
                if not p.is_absolute():
                    p = path.parent / p
                if p.is_dir():
                    paths.extend(collect_images_recursively(p))
                elif p.is_file() and is_img(p):
                    paths.append(p)
        elif path.is_dir():
            paths = collect_images_recursively(path)
        elif path.is_file() and is_img(path):
            paths = [path]

        if key == "train":
            train_images = paths
        else:
            val_images = paths

    return train_images, val_images


def parse_txt_splits() -> Tuple[List[Path], List[Path]]:
    """
    Fallback parser for train/val split from known txt files.
    """
    train_list, val_list = [], []
    train_file = None
    val_file = None
    for f in POTENTIAL_LIST_FILES:
        name = f.name.lower()
        if not f.exists():
            continue
        if "train" in name and f.suffix == ".txt":
            train_file = f
        if "val" in name and f.suffix == ".txt":
            val_file = f

    def expand_from_txt(txt: Path) -> List[Path]:
        out: List[Path] = []
        if not txt: 
            return out
        for ln in read_lines(txt):
            p = Path(ln).expanduser()
            if not p.is_absolute():
                p = txt.parent / p
            if p.is_dir():
                out.extend(collect_images_recursively(p))
            elif p.is_file() and is_img(p):
                out.append(p)
        return out

    return expand_from_txt(train_file), expand_from_txt(val_file)


def basenames(paths: List[Path]) -> Set[str]:
    return {p.name for p in paths}


def detect_inference_images() -> List[Path]:
    """
    Try to infer which images were used for detection/displacement by scanning
    candidate dirs for image files. Also dive into runs/detect/* subdirs.
    """
    found: List[Path] = []
    for root in INFERENCE_CANDIDATE_DIRS:
        if not root.exists():
            continue
        if root.name == "detect":
            # YOLO 'runs/detect/exp*' structure
            for exp in root.glob("exp*"):
                found += [p for p in exp.rglob("*") if p.is_file() and is_img(p)]
        else:
            found += collect_images_recursively(root)
    # Deduplicate while keeping order
    seen = set()
    uniq = []
    for p in found:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def main():
    print(f"Working dir: {HERE}")

    # 1) Collect actual images
    actual_imgs: List[Path] = []
    for d in ACTUAL_DIRS:
        imgs = collect_images_recursively(d)
        actual_imgs += imgs
        print(f"[Actual] {d}: {len(imgs)} images")

    # 2) Parse YOLO split
    train_imgs_yaml, val_imgs_yaml = parse_data_yaml(DATA_YAML)
    # If data.yaml did not yield anything, try txt fallback
    if not train_imgs_yaml and not val_imgs_yaml:
        print("[Info] data.yaml did not yield a split. Trying txt-based lists…")
        train_imgs_yaml, val_imgs_yaml = parse_txt_splits()

    print(f"[YOLO] Train images detected: {len(train_imgs_yaml)}")
    print(f"[YOLO] Val images detected:   {len(val_imgs_yaml)}")

    # 3) Compute overlaps
    names_actual = basenames(actual_imgs)
    names_train = basenames(train_imgs_yaml)
    names_val   = basenames(val_imgs_yaml)

    leak_train = sorted(names_actual & names_train)
    leak_val   = sorted(names_actual & names_val)

    # 4) Detect which images were used for inference / displacement
    infer_imgs = detect_inference_images()
    names_infer = basenames(infer_imgs)
    actual_used_for_infer = sorted(names_actual & names_infer)

    # 5) Report
    print("\n=== AUDIT SUMMARY ===")
    print(f"Actual images total: {len(names_actual)}")
    print(f"YOLO train total:    {len(names_train)}")
    print(f"YOLO val total:      {len(names_val)}")
    print(f"Overlap (Actual ∩ Train): {len(leak_train)}")
    print(f"Overlap (Actual ∩ Val):   {len(leak_val)}")
    print(f"Images used for inference found: {len(names_infer)}")
    print(f"Actual images used for inference: {len(actual_used_for_infer)}")

    if leak_train or leak_val:
        print("\n[WARNING] Potential leakage detected!")
        if leak_train:
            print(f" - {len(leak_train)} actual images present in TRAIN: {leak_train[:20]}{' …' if len(leak_train)>20 else ''}")
        if leak_val:
            print(f" - {len(leak_val)} actual images present in VAL:   {leak_val[:20]}{' …' if len(leak_val)>20 else ''}")
    else:
        print("\n✅ No leakage from actual_images into YOLO train/val detected (by filename match).")

    # 6) Save CSV report
    report_path = HERE / "leakage_audit_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["actual_count", len(names_actual)])
        w.writerow(["train_count", len(names_train)])
        w.writerow(["val_count", len(names_val)])
        w.writerow(["overlap_actual_train", len(leak_train)])
        w.writerow(["overlap_actual_val", len(leak_val)])
        w.writerow(["inference_images_found", len(names_infer)])
        w.writerow(["actual_used_for_inference", len(actual_used_for_infer)])

        # optional append detailed lists (one-per-row)
        if leak_train:
            w.writerow([])
            w.writerow(["leaked_train_filenames"])
            for n in leak_train:
                w.writerow([n])
        if leak_val:
            w.writerow([])
            w.writerow(["leaked_val_filenames"])
            for n in leak_val:
                w.writerow([n])
        if actual_used_for_infer:
            w.writerow([])
            w.writerow(["actual_images_used_for_inference"])
            for n in actual_used_for_infer:
                w.writerow([n])

    print(f"\nSaved: {report_path}")
    print("\nTip: if your splits are stored elsewhere, point DATA_YAML or POTENTIAL_LIST_FILES to the correct locations.")


if __name__ == "__main__":
    main()
