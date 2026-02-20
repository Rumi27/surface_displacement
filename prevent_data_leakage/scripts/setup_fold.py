# scripts/setup_fold.py
#!/usr/bin/env python3
import shutil, re, sys
from pathlib import Path

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"case3","case4","case5","case6"}:
        print("Usage: python scripts/setup_fold.py case3|case4|case5|case6")
        sys.exit(1)
    holdout = sys.argv[1]  # e.g., "case3"

    root = Path(".").resolve()
    ai = root / "actual_images"
    fold = root / holdout
    gti = fold / "gan_train_input"
    if not ai.exists():
        raise FileNotFoundError(f"Missing {ai}")

    # clear gan_train_input then refill
    if gti.exists():
        shutil.rmtree(gti)
    gti.mkdir(parents=True)

    pat = re.compile(r"^(case[3-6])_(before|after)_actual.*\.(jpg|png|jpeg)$", re.IGNORECASE)

    for p in ai.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        case_id = m.group(1).lower()
        if case_id == holdout:
            continue  # skip held-out case
        shutil.copy2(p, gti / p.name)

    print(f"[OK] Prepared {holdout}/gan_train_input with all cases except {holdout}.")

if __name__ == "__main__":
    main()
