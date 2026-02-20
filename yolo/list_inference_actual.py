# save as list_inference_actual.py and run: python list_inference_actual.py
from pathlib import Path
from audit import detect_inference_images, ACTUAL_DIRS, basenames  # reuse functions/imports from your audit.py
HERE = Path.cwd()
actual = set()
for d in ACTUAL_DIRS:
    if d.exists():
        actual |= {p.name for p in d.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}}
infer = basenames(detect_inference_images())
used = sorted(actual & infer)
print("\nActual images used for inference ({}):".format(len(used)))
for n in used: print(n)
