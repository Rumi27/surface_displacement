#!/usr/bin/env python
# -----------------------------------------------------------
#  actual.py – compare mapped YOLO CSVs vs. ground-truth
#             with RMSE and R² for each case
# -----------------------------------------------------------

from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score
import re

# ---------- global font sizes ------------------------------------
mpl.rc('font',  size=18)
mpl.rc('axes',  titlesize=20, labelsize=18)
mpl.rc('xtick', labelsize=18)
mpl.rc('ytick', labelsize=18)

from pathlib import Path
HERE = Path(__file__).parent.resolve()

gt_dir   = HERE / "actual_measurment"
yolo_dir = HERE / "results_mapped"
cases    = ["Case3", "Case4", "Case5", "Case6"]

# token indices for each case  (xb, yb, xa, ya)
OFFSETS = {
    "Case3": (2, 4, 11, 13),
    "Case4": (2, 4, 10, 12),
    "Case5": (2, 4, 11, 13),
    "Case6": (2, 4, 10, 12),
}

# ------------------------------------------------------------------
def load_gt(path: Path, case: str) -> pd.DataFrame:
    xb_i, yb_i, xa_i, ya_i = OFFSETS[case]
    raw, rows = pd.read_csv(path, sep=";", header=None, engine="python"), []
    for toks in raw.values:
        toks = [str(t).strip() for t in toks if pd.notna(t) and str(t).strip()]
        if not (toks and re.fullmatch(r"[A-G][1-7]", toks[0], re.I)):
            continue
        lab = toks[0].upper()
        try:
            x_b, y_b = float(toks[xb_i]), float(toks[yb_i])
            x_a, y_a = float(toks[xa_i]), float(toks[ya_i])
        except (ValueError, IndexError):
            continue
        rows.append([lab, x_a - x_b, y_a - y_b])
    return pd.DataFrame(rows, columns=["marker_label", "dx_true", "dy_true"])

# ------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
axs = axs.ravel()
rmses, r2s = [], []

for ax, case in zip(axs, cases):
    gt_file = next(gt_dir.glob(f"*{case.split('Case')[-1]}*.csv"))
    gt = load_gt(gt_file, case)

    yolo = pd.read_csv(yolo_dir / f"{case}_displacement_mm.csv")
    yolo.columns = [c.lower().strip() for c in yolo.columns]
    yolo["marker_label"] = yolo["marker_label"].astype(str).str.strip().str.upper()

    df = pd.merge(gt, yolo, on="marker_label", how="inner")

    if df.empty:
        print(f"[WARN] no ID overlap for {case}")
        ax.axis("off")
        rmses.append(np.nan)
        r2s.append(np.nan)
        continue

    df["d_gt"]   = np.hypot(df.dx_true, df.dy_true)
    df["d_yolo"] = np.hypot(df.dx_mm,   df.dy_mm)

    ax.scatter(df.d_gt, df.d_yolo, s=22, alpha=.75)
    lim = df[["d_gt", "d_yolo"]].to_numpy().max()
    ax.plot([0, lim], [0, lim], "k--")
    ax.set_title(case, fontsize=28)
    ax.set_xlabel("True disp. [mm]", fontsize=24)
    ax.set_ylabel("YOLO disp. [mm]", fontsize=24)

    rmse = np.sqrt(np.mean((df.d_gt - df.d_yolo) ** 2))
    r2 = r2_score(df.d_gt, df.d_yolo)
    rmses.append(rmse)
    r2s.append(r2)

    
    METRIC_FONTSIZE = 22   # <- tweak this
    METRIC_WEIGHT   = "normal"

    ax.text(0.04, 0.76,
            f"RMSE = {rmse:.2f} mm\nR² = {r2:.2f}",
            transform=ax.transAxes,
            fontsize=METRIC_FONTSIZE,
            fontweight=METRIC_WEIGHT,
            linespacing=1.3,
            bbox=dict(fc="white", ec="0.4", lw=1.2, boxstyle="round,pad=0.35"))

plt.suptitle("YOLO-derived vs. ground-truth marker displacements", y=1.02)
plt.tight_layout()
plt.savefig("fig9_yolo_vs_gt_r2.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Print RMSE and R² summary to terminal
print("\n📊 RMSE and R² summary:")
for c, rm, r2v in zip(cases, rmses, r2s):
    print(f"{c}: RMSE = {rm:.2f} mm, R² = {r2v:.3f}")
if any(not np.isnan(v) for v in rmses):
    print("Overall RMSE:", f"{np.nanmean(rmses):.2f} mm")






