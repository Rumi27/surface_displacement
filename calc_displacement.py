# -----------------------------------------------------------
#  calc_displacement.py
#  pairs every CaseX_before / CaseX_after image in actual_images/
#  runs YOLO detection, matches nearest neighbours, and writes
#  results/CaseX_displacement_mm.csv

#Load the trained YOLOv5 model (torch.hub.load + .conf).

#Locate each before/after pair in actual_images/.

#Detect the black dots in both images.

#Match every before centroid to its nearest after centroid with a KD-tree.

#Convert pixel shifts to millimetres using the known image width (1 px ≈ 1500 / img_w mm).

#Save per-marker Δx, Δy and magnitude to results/CaseX_displacement_mm.csv.
# -----------------------------------------------------------

from pathlib import Path
import re, pandas as pd
from scipy.spatial import cKDTree
import torch, numpy as np

# ---------- user settings ----------------------------------
weights   = "runs/train/markers_m_img1920/weights/best.pt"
images_dir = Path("actual_images")
mm_real_width = 1500.0        # real‐world width corresponding to image width
out_dir  = Path("results")
out_dir.mkdir(exist_ok=True)
# -----------------------------------------------------------

# 1.  load model
model = torch.hub.load("ultralytics/yolov5", "custom",
                       path="runs/train/markers_m_img1920/weights/best.pt")
model.conf = 0.25  # confidence threshold

# 2.  discover before/after pairs
pairs = {}                    # {Case3: (before_path, after_path)}
for p in images_dir.iterdir():
    m = re.search(r"case[-_]?(\d+)", p.name, re.I)
    if not m:                                    # skip files w/o “case”
        continue
    cid = f"Case{m.group(1)}"
    low = p.name.lower()
    if "before" in low:
        pairs.setdefault(cid, [None, None])[0] = p
    elif "after" in low:
        pairs.setdefault(cid, [None, None])[1] = p

# 3.  process each complete pair
def to_centroids(det):
    # det.xywh[0] shape = (N, 6)  [x, y, w, h, conf, cls]
    boxes = det.xywh[0].cpu().numpy()
    return {i: (b[0], b[1]) for i, b in enumerate(boxes)}

for cid, (before_path, after_path) in pairs.items():
    if before_path is None or after_path is None:
        print(f"[WARN] missing before/after for {cid} – skipping")
        continue
    print(f"[INFO] processing {cid}")

    det_bef = model(before_path)
    det_aft = model(after_path)

    pts_bef = to_centroids(det_bef)
    pts_aft = to_centroids(det_aft)

    # nearest-neighbour match
    tree = cKDTree(list(pts_aft.values()))
    _, idx = tree.query(list(pts_bef.values()))
    matched = {k: list(pts_aft.values())[j] for k, j in zip(pts_bef, idx)}

    # pixel → mm scale  (replace the old line)
    img_w = det_bef.ims[0].shape[1]      # ims[0] is an (H,W,C) numpy array
    mm_per_px = mm_real_width / img_w

    disp = []
    for mid in pts_bef:
        x0, y0 = pts_bef[mid];  x1, y1 = matched[mid]
        dx_mm = (x1 - x0) * mm_per_px
        dy_mm = (y1 - y0) * mm_per_px
        disp.append({
            "marker_id": mid,
            "dx_mm": dx_mm,
            "dy_mm": dy_mm,
            "total_mm": (dx_mm**2 + dy_mm**2) ** 0.5
        })

    pd.DataFrame(disp).to_csv(out_dir / f"{cid}_displacement_mm.csv", index=False)
    print(f"💾  saved {cid} displacement")

print("✓ done")

