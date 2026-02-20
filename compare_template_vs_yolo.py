import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# === Configuration ===
template_image_path = "actual images/case3_after_actual.jpg"
template_crop_box = (1249, 150, 38, 30)  # (x, y, width, height)
val_images_dir = "yolo/marker_yolo/images/val"
val_labels_dir = "yolo/marker_yolo/labels/val"
method = cv2.TM_CCOEFF_NORMED
threshold = 0.5  # Confidence threshold for match

# === Load and crop template ===
template_img = cv2.imread(template_image_path, 0)
x, y, w, h = template_crop_box
template = template_img[y:y+h, x:x+w]
print(f"✅ Loaded template shape: {template.shape}")

# === Utility: Euclidean distance
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# === Init results ===
all_distances = []
missed = 0
images_checked = 0
visualize_max = 3

# === Iterate over validation images
for image_path in tqdm(sorted(glob(os.path.join(val_images_dir, "*.jpg")))):
    base = os.path.basename(image_path).replace(".jpg", "")
    label_path = os.path.join(val_labels_dir, f"{base}.txt")

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"⚠️ Failed to load {image_path}")
        continue
    h_img, w_img = img.shape

    # Load YOLO labels
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, bw, bh = map(float, parts)
                px = cx * w_img
                py = cy * h_img
                labels.append((px, py))

    # Apply template matching
    res = cv2.matchTemplate(img, template, method)
    loc = np.where(res >= threshold)
    matches = list(zip(*loc[::-1]))  # (x, y) of top-left corners
    pred_centers = [(x + w // 2, y + h // 2) for (x, y) in matches]
    true_centers = labels.copy()

    matched_gt = set()
    for pred in pred_centers:
        distances = [euclidean(pred, gt) for gt in true_centers]
        if distances:
            min_dist = min(distances)
            all_distances.append(min_dist)
            matched_gt.add(distances.index(min_dist))

    missed += len(true_centers) - len(matched_gt)
    images_checked += 1

    # Optional Visualization
    if images_checked <= visualize_max:
        img_color = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(img_color)

        for i, pt in enumerate(pred_centers):
            label = 'Predicted' if i == 0 else ""
            plt.plot(pt[0], pt[1], 'go', label=label)

        for i, gt in enumerate(true_centers):
            label = 'Ground Truth' if i == 0 else ""
            plt.plot(gt[0], gt[1], 'ro', label=label)

        plt.title(f"Template Matching: {base}")
        plt.legend()
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# === Results Summary ===
if all_distances:
    print("\n📊 === Template Matching Evaluation ===")
    print(f"🧪 Tested images         : {images_checked}")
    print(f"📍 Average error (px)    : {np.mean(all_distances):.2f}")
    print(f"❌ Missed detections     : {missed}")
    print(f"📈 Avg detections/image  : {len(all_distances) / images_checked:.2f}")

    # Plot histogram
    plt.hist(all_distances, bins=20, edgecolor='black')
    plt.title("Localization Error Histogram")
    plt.xlabel("Euclidean Distance (pixels)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("template_vs_yolo_localization_error.png")
    print("📁 Saved: template_vs_yolo_localization_error.png")
else:
    print("⚠️ No detections above threshold.")

