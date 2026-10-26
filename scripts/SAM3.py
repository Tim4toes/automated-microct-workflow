import cv2
import numpy as np
from ultralytics import SAM
import os

# 1. Define Windows input and output directories
input_dir = r"D:\automated-microct-workflow\3_workflow\sam3\base_image"
output_dir = r"D:\automated-microct-workflow\3_workflow\sam3\saved_mask"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# 2. Specify the target image and construct paths
image_name = "115814LEM__rec_Tra00001774"
image_path = os.path.join(input_dir, image_name)

# Initialize model and load the specific micro CT image
model = SAM("sam3.pt")
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")

# Variables to store interactive clicks and the current mask
points = []
labels = []
current_mask = np.zeros(img.shape[:2], dtype=np.uint8)

def update_segmentation():
    """Runs the SAM model with the current points and updates the mask."""
    global current_mask
    if not points:
        current_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        return

    results = model(source=img, points=points, labels=labels, device=0, verbose=False)
    
    if results[0].masks is not None:
        mask_tensor = results[0].masks.data[0]
        mask_np = mask_tensor.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        current_mask = (mask_resized * 255).astype(np.uint8)

def mouse_click(event, x, y, flags, param):
    """Captures mouse clicks to add positive (left) or negative (right) points."""
    global points, labels
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        labels.append(1)
        update_segmentation()
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x, y])
        labels.append(0)
        update_segmentation()

# Set up the interactive window
cv2.namedWindow("SAM Interactive Editor")
cv2.setMouseCallback("SAM Interactive Editor", mouse_click)

print("Controls:")
print("Left Click: Add positive point (include region)")
print("Right Click: Add negative point (exclude region)")
print("'s': Save the current mask as a CTAn bitmap")
print("'c': Clear all points")
print("'q': Quit the interface")

while True:
    display_img = img.copy()
    
    # Overlay the current red segmentation mask
    if np.any(current_mask):
        colored_mask = np.zeros_like(img)
        colored_mask[current_mask > 0] = [0, 0, 255] 
        display_img = cv2.addWeighted(display_img, 0.7, colored_mask, 0.3, 0)
    
    # Draw the point markers (Green = Positive, Red = Negative)
    for pt, lbl in zip(points, labels):
        color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
        cv2.circle(display_img, (pt[0], pt[1]), 5, color, -1)

    cv2.imshow("SAM Interactive Editor", display_img)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        # Extract base name and append .bmp for correct OpenCV encoding
        base_name = os.path.splitext(image_name)[0]
        output_filename = f"{base_name}.bmp"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, current_mask)
        print(f"Bitmap mask saved to: {output_path}")
    elif key == ord('c'):
        points.clear()
        labels.clear()
        current_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        print("Points cleared.")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()