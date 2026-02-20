# calc_displacement_hungarian.py
# Matching marker centroids using Hungarian algorithm instead of KD-tree, with ground-truth marker labels

from pathlib import Path
import re, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch, numpy as np

# ---------- Settings ----------------------------------
weights     = "runs/train/markers_m_img1920/weights/best.pt"
images_dir  = Path("actual_images")
gt_dir      = Path("actual_measurment")
mm_real_width = 1500.0
out_dir     = Path("results_hungarian")
out_dir.mkdir(exist_ok=True)

# ---------- KD label offsets per case -----------------
OFFSETS = {
    "Case3": (2, 4, 11, 13),
    "Case4": (2, 4, 10, 12),
    "Case5": (2, 4, 11, 13),
    "Case6": (2, 4, 10, 12),
}

def load_gt_labels(case: str) -> list:
    """Load marker labels from GT file to match YOLO outputs by index."""
    xb_i, yb_i, xa_i, ya_i = OFFSETS[case]
    gt_file = next(gt_dir.glob(f"*{case.split('Case')[-1]}*.csv"))
    raw, labels = pd.read_csv(gt_file, sep=";", header=None, engine="python"), []
    for toks in raw.values:
        toks = [str(t).strip() for t in toks if pd.notna(t) and str(t).strip()]
        if not (toks and re.fullmatch(r"[A-G][1-7]", toks[0], re.I)):
            continue
        labels.append(toks[0].upper())
    return labels

# ---------- Load YOLOv5 model --------------------------
model = torch.hub.load("ultralytics/yolov5", "custom", path=weights)
model.conf = 0.25  # confidence threshold

# ---------- Pair images --------------------------------
pairs = {}
for p in images_dir.iterdir():
    m = re.search(r"case[-_]?(\d+)", p.name, re.I)
    if not m: continue
    cid = f"Case{m.group(1)}"
    low = p.name.lower()
    if "before" in low:
        pairs.setdefault(cid, [None, None])[0] = p
    elif "after" in low:
        pairs.setdefault(cid, [None, None])[1] = p

# ---------- Match using Hungarian Algorithm ----------
def to_centroids(det):
    boxes = det.xywh[0].cpu().numpy()
    return {i: (b[0], b[1]) for i, b in enumerate(boxes)}

for cid, (before_path, after_path) in pairs.items():
    if before_path is None or after_path is None:
        print(f"[WARN] missing before/after for {cid} – skipping")
        continue

    print(f"[INFO] processing {cid} using Hungarian matcher")

    gt_labels = load_gt_labels(cid)  # get ground truth labels for matching
    det_bef = model(before_path)
    det_aft = model(after_path)

    pts_bef = to_centroids(det_bef)
    pts_aft = to_centroids(det_aft)

    coords_bef = np.array(list(pts_bef.values()))
    coords_aft = np.array(list(pts_aft.values()))
    mm_per_px = mm_real_width / det_bef.ims[0].shape[1]

    if len(coords_bef) == 0 or len(coords_aft) == 0:
        print(f"[WARN] no detections in {cid} – skipping")
        continue

    # Cost matrix: Euclidean distances
    cost_matrix = cdist(coords_bef, coords_aft)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    disp = []
    for i, j in zip(row_ind, col_ind):
        x0, y0 = coords_bef[i]
        x1, y1 = coords_aft[j]
        dx_mm = (x1 - x0) * mm_per_px
        dy_mm = (y1 - y0) * mm_per_px
        label = gt_labels[i] if i < len(gt_labels) else f"ID{i}"
        disp.append({
            "marker_label": label,
            "dx_mm": dx_mm,
            "dy_mm": dy_mm,
            "total_mm": (dx_mm**2 + dy_mm**2)**0.5
        })

    pd.DataFrame(disp).to_csv(out_dir / f"{cid}_displacement_mm.csv", index=False)
    print(f"💾  saved {cid} Hungarian displacement")

print("✓ Hungarian matching complete")

