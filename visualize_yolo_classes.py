import os
import glob
import cv2
import matplotlib.pyplot as plt

folder_path = '/content/drive/MyDrive/pavement_pci/pavement_crack_images'
label_files = glob.glob(os.path.join(folder_path, '*.txt'))

# The mystery classes we want to identify
target_classes = {0, 5, 6, 7, 12}
found_examples = {}

print("Searching for examples of each class...")

for label_file in label_files:
    # Skip the classes.txt file if it somehow exists
    if 'classes' in label_file: 
        continue
    
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                
                # If we found one of our target classes and haven't saved an example for it yet
                if class_id in target_classes and class_id not in found_examples:
                    
                    # Figure out the matching image name (handling that .xml.txt anomaly we found earlier)
                    if label_file.endswith('.xml.txt'):
                        img_path = label_file.replace('.xml.txt', '.jpg')
                    else:
                        img_path = label_file.replace('.txt', '.jpg')
                    
                    # If the image actually exists, save the bounding box info to draw later
                    if os.path.exists(img_path):
                        found_examples[class_id] = (img_path, parts)

print(f"Found examples for classes: {list(found_examples.keys())}\n")

# Now, let's draw the boxes and display the images!
for cls_id, (img_path, box_data) in sorted(found_examples.items()):
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert for matplotlib
    h, w, _ = img.shape
    
    # Extract YOLO coordinates (normalized)
    _, x_center, y_center, box_w, box_h = map(float, box_data)
    
    # Convert normalized YOLO coordinates to actual image pixels
    x_center_pix = int(x_center * w)
    y_center_pix = int(y_center * h)
    box_w_pix = int(box_w * w)
    box_h_pix = int(box_h * h)
    
    # Calculate box corners
    xmin = int(x_center_pix - (box_w_pix / 2))
    ymin = int(y_center_pix - (box_h_pix / 2))
    xmax = int(x_center_pix + (box_w_pix / 2))
    ymax = int(y_center_pix + (box_h_pix / 2))
    
    # Draw a thick red box and put the class ID text on it
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4)
    
    # Add a background rectangle for the text so it's easy to read
    cv2.rectangle(img, (xmin, ymin - 35), (xmin + 150, ymin), (255, 0, 0), -1)
    cv2.putText(img, f'Class {cls_id}', (xmin + 5, ymin - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Display it in Colab
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"Visually Identifying Class {cls_id}")
    plt.axis('off')
    plt.show()
