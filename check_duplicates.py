import os

train_dir = "yolo/marker_yolo/images/train"
val_dir = "yolo/marker_yolo/images/val"

train_images = set(os.listdir(train_dir))
val_images = set(os.listdir(val_dir))

duplicates = train_images.intersection(val_images)

if duplicates:
    print(f"❌ Found {len(duplicates)} overlapping images:")
    for name in sorted(duplicates):
        print(name)
else:
    print("✅ No overlapping images found between training and validation sets.")
