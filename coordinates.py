import cv2
import matplotlib.pyplot as plt

img_path = "actual images/case3_after_actual.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Click two corners of a marker (top-left and bottom-right)")
coords = plt.ginput(2)
plt.show()

print("You clicked:", coords)

