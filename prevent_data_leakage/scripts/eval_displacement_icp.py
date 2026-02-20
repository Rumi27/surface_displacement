#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import cv2

def load_points_csv(p):
    pts = []
    with open(p, 'r') as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            # allow comma or whitespace; skip non-numeric headers
            parts = [x for x in s.replace('\t', ' ').replace(',', ' ').split(' ') if x]
            try:
                x = float(parts[0]); y = float(parts[1])
                pts.append([x, y])
            except Exception:
                continue
    return np.asarray(pts, dtype=np.float32)

def umeyama_similarity(src, dst, allow_reflection=False):
    # src, dst: Nx2
    assert src.shape == dst.shape and src.shape[1] == 2
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = (src_c.T @ dst_c) / n  # 2x2
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    var_src = (src_c ** 2).sum() / n
    s = 1.0 if var_src < 1e-12 else float(S.sum() / var_src)
    t = mu_d - s * (R @ mu_s)
    return s, R.astype(np.float32), t.astype(np.float32)

def apply_transform(x, s, R, t):
    return (s * (x @ R.T)) + t

def greedy_match(A, B, gate):
    # A, B: Nx2; returns list of (iA, iB, dist)
    if len(A) == 0 or len(B) == 0:
        return []
    d2 = np.sum((A[:, None, :] - B[None, :, :])**2, axis=2)
    D = np.sqrt(d2)
    cand = np.argwhere(D <= gate)
    if cand.size == 0:
        return []
    order = np.argsort(D[cand[:, 0], cand[:, 1]])
    usedA, usedB = set(), set()
    out = []
    for k in order:
        ia, ib = int(cand[k, 0]), int(cand[k, 1])
        if ia in usedA or ib in usedB:
            continue
        out.append((ia, ib, float(D[ia, ib])))
        usedA.add(ia); usedB.add(ib)
    return out

def compose_delta(s, R, t, sd, Rd, td):
    # new = delta ∘ current
    s_new = sd * s
    R_new = Rd @ R
    t_new = sd * (Rd @ t) + td
    return s_new, R_new, t_new

def guess_image_path(stem):
    base = Path("actual_images")
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        p = base / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--pred_before_csv", required=True)
    ap.add_argument("--pred_after_csv", required=True)
    ap.add_argument("--px_to_mm", type=float, required=True)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--init_gate", type=float, default=160.0)
    ap.add_argument("--final_gate", type=float, default=50.0)
    ap.add_argument("--final_match_gate", type=float, default=35.0)
    ap.add_argument("--make_overlay", action="store_true")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    A = load_points_csv(args.pred_before_csv)  # BEFORE (reference)
    B = load_points_csv(args.pred_after_csv)   # AFTER (moving)
    N_a, N_b = len(A), len(B)

    # Similarity transform B -> A
    s, R, t = 1.0, np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
    best_res = None
    for it in range(max(1, args.iters)):
        gate = args.final_gate if args.iters == 1 else (1 - it/(args.iters-1)) * args.init_gate + (it/(args.iters-1)) * args.final_gate
        B_t = apply_transform(B, s, R, t)
        pairs = greedy_match(A, B_t, gate)
        if len(pairs) < 6:
            continue
        ia = np.array([p[0] for p in pairs], dtype=int)
        ib = np.array([p[1] for p in pairs], dtype=int)
        src = B_t[ib]
        dst = A[ia]
        sd, Rd, td = umeyama_similarity(src, dst, allow_reflection=False)
        s, R, t = compose_delta(s, R, t, sd, Rd, td)

        # track residual
        B_chk = apply_transform(B, s, R, t)
        pairs_chk = greedy_match(A, B_chk, gate)
        if pairs_chk:
            resid = float(np.mean([p[2] for p in pairs_chk]))
            best_res = resid if best_res is None else min(best_res, resid)

    # Final match
    B_final = apply_transform(B, s, R, t)
    pairs_final = greedy_match(A, B_final, args.final_match_gate)

    matched = len(pairs_final)
    dists_px, rows = [], []
    for ia, ib, _ in pairs_final:
        ax, ay = A[ia].tolist()
        bx, by = B[ib].tolist()
        bxa, bya = B_final[ib].tolist()
        dx, dy = (bxa - ax), (bya - ay)
        dist_px = math.hypot(dx, dy)
        dists_px.append(dist_px)
        rows.append([ax, ay, bx, by, bxa, bya, dx, dy, dist_px])

    mean_px = float(np.mean(dists_px)) if dists_px else 0.0
    rmse_px = float(np.sqrt(np.mean(np.square(dists_px)))) if dists_px else 0.0
    px_to_mm = float(args.px_to_mm)
    mean_mm = mean_px * px_to_mm
    rmse_mm = rmse_px * px_to_mm

    theta = math.degrees(math.atan2(R[1,0], R[0,0]))
    transform_info = {"scale": float(s), "rotation_deg": float(theta), "translation": [float(t[0]), float(t[1])]}

    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "before_count": int(N_a),
        "after_count": int(N_b),
        "matched": int(matched),
        "gate_px": float(args.final_match_gate),
        "mean_disp_px": mean_px,
        "rmse_disp_px": rmse_px,
        "mean_disp_mm": mean_mm,
        "rmse_disp_mm": rmse_mm,
        "px_to_mm": px_to_mm,
        "transform": transform_info
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    pairs_dir = Path(args.fold) / "eval_real" / "metrics"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = pairs_dir / "pairs_icp.csv"
    with open(pairs_csv, "w") as f:
        f.write("ax,ay,bx,by,bx_aligned,by_aligned,dx,dy,dist_px,dist_mm\n")
        for r in rows:
            f.write(",".join(f"{v:.6f}" for v in (r + [r[-1]*px_to_mm])) + "\n")

    if args.make_overlay:
        before_stem = Path(args.pred_before_csv).stem
        img_path = guess_image_path(before_stem)
        vec_dir = Path(args.fold) / "eval_real" / "vectors"
        vec_dir.mkdir(parents=True, exist_ok=True)
        out_img_path = vec_dir / f"{before_stem}_vectors_icp.jpg"
        if img_path and Path(img_path).exists():
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                for ia, ib, _ in pairs_final:
                    ax, ay = map(int, np.round(A[ia]))
                    bxa, bya = map(int, np.round(B_final[ib]))
                    cv2.circle(img, (ax, ay), 4, (0, 255, 0), -1)
                    cv2.arrowedLine(img, (ax, ay), (bxa, bya), (0, 255, 255), 2, tipLength=0.25)
                txt = f"ICP matches: {matched}/{N_a}  mean={mean_px:.1f}px ({mean_mm:.2f}mm)  rmse={rmse_px:.1f}px"
                cv2.rectangle(img, (10, 10), (10 + 9*len(txt), 40), (0,0,0), -1)
                cv2.putText(img, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                cv2.imwrite(str(out_img_path), img)

    print("=== Displacement summary (ICP) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"[OK] pairs CSV -> {pairs_csv}")
    if args.make_overlay:
        print(f"[OK] overlay -> {out_img_path}")

if __name__ == "__main__":
    main()
