#!/usr/bin/env python3
import sys
from pathlib import Path

TEMPLATE = """# Auto-generated
path: {abs_path}
train: images/train
val: images/val
names: [marker]
"""

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"case3","case4","case5","case6"}:
        print("Usage: python scripts/yolo_make_yaml.py case3|case4|case5|case6")
        raise SystemExit(1)
    fold = Path(sys.argv[1]).resolve()
    yaml = TEMPLATE.format(abs_path=str(fold))
    (fold / "yolo_ds.yaml").write_text(yaml)
    print(f"[OK] Wrote {(fold/'yolo_ds.yaml')} with path={fold}")

if __name__ == "__main__":
    main()


