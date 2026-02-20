import numpy as np, pandas as pd, matplotlib.pyplot as plt, re
from pathlib import Path

# Font
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.titlesize'] = 18       # axis title
matplotlib.rcParams['axes.labelsize'] = 16       # axis labels
matplotlib.rcParams['xtick.labelsize'] = 14      # x-axis tick labels
matplotlib.rcParams['ytick.labelsize'] = 14      # y-axis tick labels
matplotlib.rcParams['font.size'] = 14            # base font size

# Setup
gt_dir = Path("actual_measurment")
ROTATE = dict(Case3=3, Case4=0, Case5=-3, Case6=0)
FLIP_LR = dict(Case3=False, Case4=False, Case5=True, Case6=False)
cases = ["Case3", "Case5", "Case4", "Case6"]

def orient(mat, case):
    k = ROTATE.get(case, 0)
    mat = np.rot90(mat, -k)
    if FLIP_LR.get(case, False): mat = np.fliplr(mat)
    return mat

def label_to_index(lbl):
    m = re.fullmatch(r"([A-Ga-g])\s*([1-7])", lbl.strip())
    if not m: raise ValueError(lbl)
    return ord(m.group(1).upper())-65, int(m.group(2))-1

yolo_dir = Path("yolo/results_mapped")

def load_yolo_matrix(case):
    df = pd.read_csv(yolo_dir/f"{case}_displacement_mm.csv")
    df["marker_label"] = df["marker_label"].astype(str).str.strip().str.upper()
    mat = np.full((7,7), np.nan)
    for _,row in df.iterrows():
        try:
            r,c = label_to_index(row.marker_label)
            mat[r,c] = np.hypot(row.dx_mm, row.dy_mm)
        except ValueError: pass
    return orient(mat, case)

# Plot
fig, axs = plt.subplots(2, 2, figsize=(9.5, 8), constrained_layout=True)
cmap = plt.cm.viridis.copy(); cmap.set_bad(color="lightgrey")

for ax, case in zip(axs.flat, cases):
    mat = load_yolo_matrix(case)
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=np.nanmax(mat) or 1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(range(1,8), fontsize=14)
    ax.set_yticklabels(list("ABCDEFG"), fontsize=14)
    ax.set_title(f"{case} – YOLO", fontsize=18)
    for spine in ax.spines.values(): spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel("mm", rotation=270, labelpad=12, fontsize=16)

fig.savefig("fig7_yolo_grid.png", dpi=600, bbox_inches="tight")
plt.close()
print("✓ Saved: fig7_yolo_grid.png")

