#!/usr/bin/env python3
import argparse, csv, json, math
from pathlib import Path
from PIL import Image, ImageDraw

def read_csv_points(p):
    rows=[]
    with open(p, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append((float(row["x_px"]), float(row["y_px"])))
    return rows

def mutual_nn_match(A, B, gate_px):
    # A, B: lists of (x,y). Return list of (ia, ib, dist)
    def nn(src, dst):
        idx=[]
        for i,(x,y) in enumerate(src):
            best=-1; bd=1e18
            for j,(u,v) in enumerate(dst):
                d=(x-u)**2+(y-v)**2
                if d<bd: bd=d; best=j
            idx.append((best, math.sqrt(bd)))
        return idx
    a2b = nn(A,B)
    b2a = nn(B,A)
    pairs=[]
    used_b=set()
    for ia,(jb,da) in enumerate(a2b):
        if jb<0: continue
        ib,jd = jb, b2a[jb]
        if jd[0]==ia and da<=gate_px:
            if ib not in used_b:
                pairs.append((ia, ib, da))
                used_b.add(ib)
    return pairs

def draw_vectors(before_img, pairs, A, B, out_path):
    with Image.open(before_img).convert("RGB") as im:
        dr=ImageDraw.Draw(im)
        r=6
        for ia, ib, _ in pairs:
            x,y=A[ia]; u,v=B[ib]
            dr.ellipse((x-r,y-r,x+r,y+r), outline=(0,255,0), width=2)
            dr.line((x,y,u,v), fill=(255,0,0), width=2)
            dr.ellipse((u-r,v-r,u+r,v+r), outline=(0,0,255), width=2)
        im.save(out_path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=["case3","case4","case5","case6"])
    ap.add_argument("--pred_before_csv", required=True)
    ap.add_argument("--pred_after_csv",  required=True)
    ap.add_argument("--px_to_mm", type=float, default=0.1923)
    ap.add_argument("--gate_px",   type=float, default=52.0)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--make_overlay", action="store_true")
    ap.add_argument("--actual_dir", default="actual_images")
    args=ap.parse_args()

    A = read_csv_points(args.pred_before_csv)
    B = read_csv_points(args.pred_after_csv)

    pairs = mutual_nn_match(A,B,args.gate_px)
    nA,nB,nM = len(A), len(B), len(pairs)

    dxdy = []
    for ia, ib, d in pairs:
        x,y=A[ia]; u,v=B[ib]
        dx=u-x; dy=v-y
        dxdy.append((dx,dy, math.hypot(dx,dy)))

    if dxdy:
        mean_px = sum(d for _,_,d in dxdy)/len(dxdy)
        rmse_px = (sum(d*d for _,_,d in dxdy)/len(dxdy))**0.5
    else:
        mean_px = rmse_px = 0.0

    mm = args.px_to_mm
    summary = {
        "before_count": nA,
        "after_count":  nB,
        "matched":      nM,
        "gate_px":      args.gate_px,
        "mean_disp_px": round(mean_px,3),
        "rmse_disp_px": round(rmse_px,3),
        "mean_disp_mm": round(mean_px*mm,4),
        "rmse_disp_mm": round(rmse_px*mm,4),
        "px_to_mm":     mm
    }

    out_dir = Path(args.fold) / "eval_real" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = out_dir / "pairs.csv"
    with pairs_csv.open("w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["x_before","y_before","x_after","y_after","dx_px","dy_px","d_px","d_mm"])
        for ia,ib,_ in pairs:
            x,y=A[ia]; u,v=B[ib]
            dx=u-x; dy=v-y; d=math.hypot(dx,dy)
            w.writerow([f"{x:.2f}",f"{y:.2f}",f"{u:.2f}",f"{v:.2f}",f"{dx:.2f}",f"{dy:.2f}",f"{d:.2f}",f"{d*mm:.4f}"])

    if args.out_json:
        with open(args.out_json,"w") as f:
            json.dump(summary,f,indent=2)

    print("=== Displacement summary ===")
    for k,v in summary.items():
        print(f"{k}: {v}")

    if args.make_overlay:
        # find before image path
        stem = Path(args.pred_before_csv).stem
        img_path = None
        for ext in (".jpg",".jpeg",".png"):
            p = Path(args.actual_dir)/f"{stem}{ext}"
            if p.exists(): img_path=p; break
        if img_path:
            ov_dir = Path(args.fold)/"eval_real"/"vectors"
            ov_dir.mkdir(parents=True, exist_ok=True)
            draw_vectors(img_path, pairs, A, B, ov_dir/(stem+"_vectors.jpg"))

if __name__=="__main__":
    main()
