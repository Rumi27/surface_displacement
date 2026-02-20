#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np

def make_ring_template(r, w_frac=0.3):
    r = int(round(r))
    w = max(1, int(round(w_frac * r)))  # ring half-width
    S = 2 * (r + w) + 1
    t = np.zeros((S, S), np.float32)
    cy = cx = r + w
    yy, xx = np.ogrid[:S, :S]
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    inner = r - w
    outer = r + w
    t[(rr >= inner) & (rr <= outer)] = 1.0
    # smooth + normalize for TM_CCOEFF_NORMED
    t = cv2.GaussianBlur(t, (3, 3), 0)
    t = t - t.mean()
    n = np.linalg.norm(t.ravel())
    if n > 1e-9:
        t /= n
    return t.astype(np.float32)

def greedy_topk_with_suppression(score, K, min_dist):
    H, W = score.shape
    flat_idx = np.argsort(score.ravel())[::-1]  # descending by score
    sel = []
    min_d2 = (min_dist ** 2)
    coords = []
    # do NOT early-break on low score; we target exact-K
    for idx in flat_idx:
        if len(sel) >= K:
            break
        y = int(idx // W)
        x = int(idx % W)
        ok = True
        for (px, py) in coords:
            dx = x - px
            dy = y - py
            if dx*dx + dy*dy < min_d2:
                ok = False
                break
        if ok:
            sel.append((x, y))
            coords.append((x, y))
    return sel

def label_single_image(img_path, out_txt, ring_r, min_dist, save_overlay_dir=None):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return 0
    H, W = img.shape[:2]
    gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # stabilize contrast a bit
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_u8 = clahe.apply(gray_u8)
    gray_u8 = cv2.GaussianBlur(gray_u8, (3,3), 0)

    # convert to float32 for matchTemplate (must match template dtype)
    gray = gray_u8.astype(np.float32)

    # build ring template and ensure it fits in the image; if not, shrink
    T = make_ring_template(ring_r, w_frac=0.3)  # float32
    Th, Tw = T.shape
    if Th > H or Tw > W:
        # shrink ring_r until template fits
        scale = min(H / max(Th,1), W / max(Tw,1))
        # keep it simple: reduce radius by ~20% and rebuild until fits
        rr = max(3, int(round(ring_r * max(0.3, min(0.9, scale*0.9)))))
        T = make_ring_template(rr, w_frac=0.3)
        Th, Tw = T.shape

    # try both polarities, but keep everything float32
    res_dark   = cv2.matchTemplate(255.0 - gray, T, cv2.TM_CCOEFF_NORMED)
    res_bright = cv2.matchTemplate(gray,          T, cv2.TM_CCOEFF_NORMED)
    if res_dark.max() >= res_bright.max():
        res = res_dark
    else:
        res = res_bright

    # offsets to convert template top-left to center
    offx = Tw // 2
    offy = Th // 2

    # Greedy pick exactly K peaks with min-dist suppression
    K = 49
    picks = greedy_topk_with_suppression(res, K, min_dist)

    # If fewer than K, relax min_dist progressively
    relax = 0
    while len(picks) < K and relax < 4:
        picks = greedy_topk_with_suppression(res, K, max(1, min_dist - 4*relax))
        relax += 1

    # Convert to centers in full image coordinates
    centers = []
    for (x, y) in picks[:K]:
        cx = int(x + offx)
        cy = int(y + offy)
        centers.append((cx, cy))

    # If still <K, pad with next-best ignoring suppression (rare)
    if len(centers) < K:
        flat_idx = np.argsort(res.ravel())[::-1]
        Wres = res.shape[1]
        for idx in flat_idx:
            if len(centers) >= K: break
            y = int(idx // Wres); x = int(idx % Wres)
            cx = int(x + offx); cy = int(y + offy)
            centers.append((cx, cy))

    # Write YOLO boxes: square 2*ring_r (clip if near borders)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    w = h = 2 * ring_r
    with open(out_txt, "w") as f:
        for (cx, cy) in centers[:K]:
            # normalized YOLO
            f.write(f"0 {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}\n")

    # Optional overlay
    if save_overlay_dir is not None:
        vis = img.copy()
        for (cx, cy) in centers[:K]:
            cv2.circle(vis, (cx, cy), ring_r, (0,255,0), 2)
            cv2.circle(vis, (cx, cy), 2, (0,0,255), 2)
        save_overlay_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_overlay_dir / (img_path.stem + "_lab.jpg")), vis)

    return len(centers[:K])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--radius_px", type=int, required=True, help="marker radius in pixels (e.g., 26)")
    ap.add_argument("--min_dist", type=int, default=31, help="minimum spacing between picks in pixels")
    ap.add_argument("--overlay_dir", default="", help="optional dir to save visual overlays")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None

    imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    total = 0
    ok49 = 0
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(imgs):
        txt = labels_dir / (img.stem + ".txt")
        n = label_single_image(img, txt, args.radius_px, args.min_dist, save_overlay_dir=overlay_dir)
        total += 1
        if n == 49: ok49 += 1
        if (i+1) % 50 == 0:
            print(f"[{i+1}/{len(imgs)}] {img.name} -> {n} labels")

    print(f"[DONE] Images: {total}, exactly-49: {ok49} ({100.0*ok49/total:.1f}%)")

if __name__ == "__main__":
    main()
