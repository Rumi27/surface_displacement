import pandas as pd, numpy as np
from pathlib import Path
import re

# Tolerance for considering a match a true positive (in mm)
MATCH_TOLERANCE_MM = 10.0

gt_dir   = Path("actual_measurment")
yolo_dir = Path("results_mapped")
cases    = ["Case3", "Case4", "Case5", "Case6"]

OFFSETS = {
    "Case3": (2, 4, 11, 13),
    "Case4": (2, 4, 10, 12),
    "Case5": (2, 4, 11, 13),
    "Case6": (2, 4, 10, 12),
}

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

# Evaluation results
results = []

for case in cases:
    gt_file = next(gt_dir.glob(f"*{case.split('Case')[-1]}*.csv"))
    gt = load_gt(gt_file, case)
    yolo = pd.read_csv(yolo_dir / f"{case}_displacement_mm.csv")
    yolo.columns = [c.lower().strip() for c in yolo.columns]
    yolo["marker_label"] = yolo["marker_label"].astype(str).str.strip().str.upper()

    df = pd.merge(gt, yolo, on="marker_label", how="outer", suffixes=("_gt", "_yolo"))
    df = df.dropna(subset=["dx_true", "dy_true", "dx_mm", "dy_mm"], how="any")

    df["d_gt"]   = np.hypot(df.dx_true, df.dy_true)
    df["d_yolo"] = np.hypot(df.dx_mm, df.dy_mm)
    df["error"]  = np.abs(df.d_gt - df.d_yolo)

    tp = (df["error"] <= MATCH_TOLERANCE_MM).sum()
    fn = len(gt) - tp
    fp = len(yolo) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mae       = df["error"].mean()
    std_dev   = df["error"].std()
    rmse      = np.sqrt(((df["d_gt"] - df["d_yolo"]) ** 2).mean())

    results.append({
        "Case": case,
        "RMSE": rmse,
        "MAE": mae,
        "STD_DEV": std_dev,
        "Precision": precision,
        "Recall": recall,
        "F1": f1_score,
        "TP": tp,
        "FP": fp,
        "FN": fn
    })

# Display summary
print(f"{'Case':<7} {'RMSE':>8} {'MAE':>8} {'σ':>8} {'Prec.':>8} {'Recall':>8} {'F1':>8}")
for r in results:
    print(f"{r['Case']:<7} {r['RMSE']:8.2f} {r['MAE']:8.2f} {r['STD_DEV']:8.2f} "
          f"{r['Precision']:8.2f} {r['Recall']:8.2f} {r['F1']:8.2f}")
