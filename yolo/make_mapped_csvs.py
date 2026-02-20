# make_mapped_csvs.py  – run once
import pandas as pd
from pathlib import Path

# numeric → label dictionary -------------------------------------------------
id2name = {
     0:"A1",  1:"A2",  2:"A3",  3:"A4",  4:"A5",  5:"A6",  6:"A7",
     7:"B1",  8:"B2",  9:"B3", 10:"B4", 11:"B5", 12:"B6", 13:"B7",
    14:"C1", 15:"C2", 16:"C3", 17:"C4", 18:"C5", 19:"C6", 20:"C7",
    21:"D1", 22:"D2", 23:"D3", 24:"D4", 25:"D5", 26:"D6", 27:"D7",
    28:"E1", 29:"E2", 30:"E3", 31:"E4", 32:"E5", 33:"E6", 34:"E7",
    35:"F1", 36:"F2", 37:"F3", 38:"F4", 39:"F5", 40:"F6", 41:"F7",
    42:"G1", 43:"G2", 44:"G3", 45:"G4", 46:"G5", 47:"G6", 48:"G7"
}

src_dir  = Path("results")
dst_dir  = Path("results_mapped")
dst_dir.mkdir(exist_ok=True)

for csv in src_dir.glob("Case*_displacement_mm.csv"):
    df = pd.read_csv(csv)
    df["marker_label"] = (
        df["marker_id"]
          .astype(float).astype(int)      # 0.0 → 0
          .map(id2name)                   # 0 → 'A1'
    )
    # keep numeric id too, but sorted columns look nicer
    cols = ["marker_id","marker_label","dx_mm","dy_mm","total_mm"]
    df[cols].to_csv(dst_dir / csv.name, index=False)
    print("✅ wrote", dst_dir / csv.name)

print("\nAll mapped files saved to", dst_dir)
