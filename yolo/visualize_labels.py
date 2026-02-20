import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend


# === Config ===
images_dir = "marker_yolo/images/train"
labels_dir = "marker_yolo/labels/train"

# === Read all images ===
image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]

# === Visualize a few images with bounding boxes ===
for idx, img_file in enumerate(sorted(image_files)):
    if idx >= 10:  # visualize only first 10 for speed
        break

    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Could not read image: {img_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, bw, bh = map(float, parts)
                # Convert YOLO to pixel coordinates
                x_pixel = (x - bw / 2) * w
                y_pixel = (y - bh / 2) * h
                bw_pixel = bw * w
                bh_pixel = bh * h

                rect = patches.Rectangle((x_pixel, y_pixel), bw_pixel, bh_pixel, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x_pixel, y_pixel - 5, f"Class {int(cls)}", color='yellow', fontsize=8, weight='bold')

    else:
        print(f"⚠️ No label file for: {img_file}")

    plt.title(f"{img_file}")
    plt.axis("off")
    plt.tight_layout()
    
    # Save instead of showing
    output_path = f"label_vis_{idx}.png"
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Saved: {output_path}")

