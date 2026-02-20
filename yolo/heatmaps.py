#!/usr/bin/env python
# -----------------------------------------------------------
#  heatmaps.py – YOLO vs. ground-truth displacement (7×7 grid)
#                • absolute-error panel
#                • per-case CSV of |error|
#                • per-case orientation correction
#                • computes: Pearson R, MAE, SSIM
# -----------------------------------------------------------
import numpy as np, pandas as pd, matplotlib.pyplot as plt, re
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'

# ------------------------------------------------------------------
# folders and cases
# ------------------------------------------------------------------
gt_dir = Path("actual_measurment")  # if this is still correct in root dir
yolo_dir = Path("yolo/results_mapped")
out_tab  = Path("tables");  out_tab.mkdir(exist_ok=True)
cases    = ["Case3", "Case4", "Case5", "Case6"]

# ------------------------------------------------------------------
# board orientation  (rotate = clockwise quarter-turns, flip_lr = mirror)
# ------------------------------------------------------------------
ROTATE   = dict(Case3=3,   Case4=0, Case5=-3, Case6=0)   # -1 == 3 ccw turns
FLIP_LR  = dict(Case3=False, Case4=False, Case5=True, Case6=False)

def orient(mat: np.ndarray, case: str) -> np.ndarray:
    """Rotate/flip matrix so North is up, West is left."""
    k = ROTATE.get(case, 0)
    mat = np.rot90(mat, -k)  # numpy rot90 is CCW by default
    if FLIP_LR.get(case, False):
        mat = np.fliplr(mat)
    return mat

# ------------------------------------------------------------------
# helpers: label→index and file loaders
# ------------------------------------------------------------------
def label_to_index(lbl: str):
    m = re.fullmatch(r"([A-Ga-g])\s*([1-7])", lbl.strip())
    if not m: raise ValueError(lbl)
    return ord(m.group(1).upper())-65, int(m.group(2))-1  # row, col (0-based)

def load_gt_matrix(case: str) -> np.ndarray:
    f = next(gt_dir.glob(f"*{case.split('Case')[-1]}*.csv"))
    raw = pd.read_csv(f, sep=";", header=None, engine="python")
    mat = np.full((7,7), np.nan)

    xb, yb, xa, ya = (2,4,11,13) if case in ("Case3","Case5") else (2,4,10,12)
    for row in raw.values:
        toks = [str(t).strip() for t in row if pd.notna(t) and str(t).strip()]
        if toks and re.fullmatch(r"[A-G][1-7]", toks[0], re.I):
            r,c = label_to_index(toks[0])
            try:
                x_b,y_b = float(toks[xb]), float(toks[yb])
                x_a,y_a = float(toks[xa]), float(toks[ya])
                mat[r,c] = np.hypot(x_a-x_b, y_a-y_b)
            except Exception: pass
    return orient(mat, case)

def load_yolo_matrix(case: str) -> np.ndarray:
    df = pd.read_csv(yolo_dir/f"{case}_displacement_mm.csv")
    df["marker_label"] = df["marker_label"].astype(str).str.strip().str.upper()
    mat = np.full((7,7), np.nan)
    for _, row in df.iterrows():
        try:
            r, c = label_to_index(row.marker_label)
            mat[r,c] = np.hypot(row.dx_mm, row.dy_mm)
        except ValueError: pass
    return orient(mat, case)

# ------------------------------------------------------------------
# plotting
# ------------------------------------------------------------------
fig, axs = plt.subplots(len(cases), 2, figsize=(8, 10), constrained_layout=True)
cmap = plt.cm.viridis.copy(); cmap.set_bad(color="lightgrey")

for i, case in enumerate(cases):
    m_yolo = load_yolo_matrix(case)
    m_gt   = load_gt_matrix(case)

    panels = [(m_yolo, "YOLO"),
              (m_gt  , "Ground truth")]

    for j, (mat, col_label) in enumerate(panels):
        ax = axs[i, j]
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=np.nanmax(mat) or 1)
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(range(1,8)); ax.set_yticklabels(list("ABCDEFG"))

        # Add case title above each row
        ax.set_title(f"{case}", fontsize=16, pad=4)

        # Only show Y-axis label on leftmost
        if j == 0:
            ax.set_ylabel("YOLO", fontsize=16, labelpad=20)
        elif j == 1:
            ax.set_ylabel("Ground truth", fontsize=16, labelpad=20)

        for spine in ax.spines.values(): spine.set_visible(False)

    # Colorbar
    cbar = fig.colorbar(im, ax=axs[i, -1], fraction=0.046, pad=0.02)
    cbar.ax.set_ylabel("mm", rotation=270, labelpad=16)

fig.savefig("fig2_displacement_heatmaps_clean.png", dpi=600, bbox_inches="tight")
print("✓ Saved: fig2_displacement_heatmaps_clean.png")

# ------------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------------
print("\n📊 Quantitative Evaluation (YOLO vs Ground Truth)")
summary = []

for case in cases:
    m_yolo = load_yolo_matrix(case)
    m_gt   = load_gt_matrix(case)

    mask = ~np.isnan(m_gt) & ~np.isnan(m_yolo)
    yolo_vals = m_yolo[mask]
    gt_vals   = m_gt[mask]

    if len(gt_vals) > 0:
        mae = mean_absolute_error(gt_vals, yolo_vals)
        r, _ = pearsonr(gt_vals, yolo_vals)
        try:
            ssim_val = ssim(m_gt, m_yolo, data_range=np.nanmax(m_gt) - np.nanmin(m_gt))
        except:
            ssim_val = np.nan
        summary.append((case, r, mae, ssim_val))
        print(f"{case}: R = {r:.3f}, MAE = {mae:.2f} mm, SSIM = {ssim_val:.3f}")
    else:
        print(f"{case}: ⚠️ No valid comparison points.")

# Save summary table
df_summary = pd.DataFrame(summary, columns=["Case", "Pearson_R", "MAE_mm", "SSIM"])
df_summary.to_csv(out_tab / "displacement_metrics_summary.csv", index=False)
print("📁 Saved: tables/displacement_metrics_summary.csv")



