import os
import glob
import cv2
import numpy as np
from ultralytics import SAM

# 1. Define Windows input and output directories
input_dir = r"F:\automated-microct-workflow\3_workflow\annotation_testing"
output_dir = r"F:\automated-microct-workflow\3_workflow\annotation_testing\sam3_masks"
os.makedirs(output_dir, exist_ok=True)

# 2. Discover images in input_dir (prioritizing image_52a6b6.jpg if present)
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
image_files = sorted(
    [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
)

if not image_files:
    raise FileNotFoundError(f"No valid images found in {input_dir}")

target_default = "115814LEM__rec_Tra00001805/bmp"
current_idx = image_files.index(target_default) if target_default in image_files else 0

# 3. Initialize SAM model (uses sam3.pt if downloaded, otherwise auto-downloads sam2.1_b.pt)
model_path = "sam3.pt" if os.path.exists("sam3.pt") else "sam2.1_b.pt"
print(f"Loading model: {model_path} on GPU (device=0)...")
model = SAM(r"F:\automated-microct-workflow\checkpoints\foundation_models\sam3.pt")

# State variables
img = None
image_name = ""
# Each item in `objects` is a dict: {"points": [[x, y], ...], "labels": [1, 0, ...], "mask": np.ndarray (bool)}
objects = []
active_obj_idx = -1
pure_mask_view = False  # Toggle between overlay view and pure binary CTAn mask view
PIN_CLICK_RADIUS = 12   # Pixel radius for clicking an existing point marker to delete it


def get_combined_mask():
    """Combines all individual object boolean masks into a single 8-bit mask (0 or 255)."""
    if not objects:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    combined = np.zeros(img.shape[:2], dtype=bool)
    for obj in objects:
        if obj["mask"] is not None:
            combined |= obj["mask"]
    return (combined * 255).astype(np.uint8)


def run_object_inference(obj_idx):
    """Runs SAM inference for a single specific object using nested 3D point prompts."""
    if obj_idx < 0 or obj_idx >= len(objects):
        return

    obj = objects[obj_idx]
    # If an object has no positive points left, remove it
    if not obj["points"] or 1 not in obj["labels"]:
        objects.pop(obj_idx)
        return

    # Nest points and labels as [[[x1, y1], [x2, y2]]] and [[1, 0]] so SAM groups them onto ONE object
    results = model.predict(
        source=img,
        points=[obj["points"]],
        labels=[obj["labels"]],
        device=0,
        verbose=False
    )

    if results and results[0].masks is not None and len(results[0].masks.data) > 0:
        mask_tensor = results[0].masks.data[0]
        mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
        if mask_np.shape[:2] != img.shape[:2]:
            mask_np = cv2.resize(
                mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        obj["mask"] = mask_np > 0
    else:
        obj["mask"] = np.zeros(img.shape[:2], dtype=bool)


def find_clicked_pin(x, y):
    """Checks if the user clicked on/near an existing point marker."""
    for obj_i, obj in enumerate(objects):
        for pt_i, (px, py) in enumerate(obj["points"]):
            if np.hypot(x - px, y - py) <= PIN_CLICK_RADIUS:
                return obj_i, pt_i
    return None, None


def find_object_covering_pixel(x, y):
    """Finds the most recently created object whose mask covers coordinate (x, y)."""
    for obj_i in range(len(objects) - 1, -1, -1):
        mask = objects[obj_i]["mask"]
        if mask is not None and mask[y, x]:
            return obj_i
    return None


def mouse_click(event, x, y, flags, param):
    """Handles interactive object selection, refinement, and unselection."""
    global active_obj_idx

    # 1. MIDDLE CLICK or CTRL + RIGHT CLICK: Immediately unselect/delete the entire object under cursor
    if event == cv2.EVENT_MBUTTONDOWN or (
        event == cv2.EVENT_RBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY)
    ):
        target_idx = find_object_covering_pixel(x, y)
        if target_idx is not None:
            objects.pop(target_idx)
            active_obj_idx = len(objects) - 1
            print(f"Unselected object #{target_idx + 1}.")
        return

    # 2. LEFT CLICK: Select a new object (or Shift + Left-Click to expand the active object)
    if event == cv2.EVENT_LBUTTONDOWN:
        # If Shift is held, add a positive point to the currently active object
        if (flags & cv2.EVENT_FLAG_SHIFTKEY) and active_obj_idx >= 0 and active_obj_idx < len(objects):
            objects[active_obj_idx]["points"].append([x, y])
            objects[active_obj_idx]["labels"].append(1)
            run_object_inference(active_obj_idx)
            print(f"Added positive refinement point to object #{active_obj_idx + 1}.")
        else:
            # Create and segment a new independent object
            new_obj = {"points": [[x, y]], "labels": [1], "mask": None}
            objects.append(new_obj)
            active_obj_idx = len(objects) - 1
            run_object_inference(active_obj_idx)
            print(f"Selected new object #{active_obj_idx + 1} at ({x}, {y}).")

    # 3. RIGHT CLICK: Unselect a pin/object OR add a negative exclusion point to trim a mask
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Check if user right-clicked directly on an existing green/red pin
        pin_obj_idx, pin_pt_idx = find_clicked_pin(x, y)
        if pin_obj_idx is not None:
            objects[pin_obj_idx]["points"].pop(pin_pt_idx)
            objects[pin_obj_idx]["labels"].pop(pin_pt_idx)
            if not objects[pin_obj_idx]["points"] or 1 not in objects[pin_obj_idx]["labels"]:
                objects.pop(pin_obj_idx)
                print(f"Unselected object #{pin_obj_idx + 1} (removed its point marker).")
            else:
                run_object_inference(pin_obj_idx)
                print(f"Removed point marker from object #{pin_obj_idx + 1}.")
            active_obj_idx = len(objects) - 1
            return

        # Otherwise, check if the user right-clicked inside an existing highlighted mask to trim/exclude that area
        target_idx = find_object_covering_pixel(x, y)
        if target_idx is None and active_obj_idx >= 0 and active_obj_idx < len(objects):
            target_idx = active_obj_idx

        if target_idx is not None:
            objects[target_idx]["points"].append([x, y])
            objects[target_idx]["labels"].append(0)  # Negative point
            active_obj_idx = target_idx
            run_object_inference(target_idx)
            print(f"Added negative exclusion point to object #{target_idx + 1} at ({x}, {y}).")


def load_current_image():
    """Loads the image at `current_idx` and resets object selections."""
    global img, image_name, objects, active_obj_idx
    image_name = image_files[current_idx]
    image_path = os.path.join(input_dir, image_name)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    objects.clear()
    active_obj_idx = -1
    print(f"\nLoaded [{current_idx + 1}/{len(image_files)}]: {image_name} ({img.shape[1]}x{img.shape[0]})")


# Load initial image and configure resizable window
load_current_image()
window_name = "SAM 3 Interactive Micro-CT Editor"

# Add the KEEPRATIO flag to prevent OpenCV from stretching the image
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

# Calculate a uniform scale to fit the screen while maintaining exact dimensions
scale = min(1400 / img.shape[1], 950 / img.shape[0])
cv2.resizeWindow(window_name, int(img.shape[1] * scale), int(img.shape[0] * scale))

cv2.setMouseCallback(window_name, mouse_click)

print("\n=== CONTROLS ===")
print("Left Click              : Select a discrete object (creates a new object mask)")
print("Shift + Left Click      : Add positive point to expand the currently active object")
print("Right Click on Mask     : Add negative point (red) to unselect/trim unwanted mask bleed")
print("Right Click on Pin      : Remove that pin (if it's the only positive pin, unselects the object)")
print("Middle Click (or Ctrl+R): Immediately unselect/delete the entire highlighted object under cursor")
print("'u' or 'z'              : Undo last object selection")
print("'m'                     : Toggle between Overlay View and pure Black/White Mask View")
print("'c'                     : Clear all selected objects")
print("'s'                     : Save CTAn .bmp mask to saved_mask folder")
print("'n' / 'p'               : Next / Previous image in base_image folder")
print("'q' or ESC              : Quit")

while True:
    combined_mask = get_combined_mask()

    if pure_mask_view:
        # Pure binary mask view (exact preview of the saved CTAn bitmap)
        display_img = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    else:
        display_img = img.copy()

        # Highlight selected objects with a white fill and a crisp solid white outline
        if np.any(combined_mask):
            white_layer = np.full_like(display_img, 255)
            mask_bool = combined_mask > 0
            # Blend 65% white fill over the selected objects so underlying bone texture remains slightly visible
            display_img[mask_bool] = cv2.addWeighted(
                display_img[mask_bool], 0.35, white_layer[mask_bool], 0.65, 0
            )

            # Draw solid white outline (2px thickness) around every selected object
            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(display_img, contours, -1, (255, 255, 255), 2)

        # Draw interactive point markers (Green = Positive, Red = Negative)
        for obj_i, obj in enumerate(objects):
            for pt, lbl in zip(obj["points"], obj["labels"]):
                fill_color = (0, 220, 0) if lbl == 1 else (0, 0, 255)
                # Highlight the active object's pins with a yellow outer ring, others with black
                ring_color = (0, 255, 255) if obj_i == active_obj_idx else (0, 0, 0)
                cv2.circle(display_img, (pt[0], pt[1]), 6, ring_color, -1)
                cv2.circle(display_img, (pt[0], pt[1]), 4, fill_color, -1)

    # Status bar overlay
    status_text = (
        f"{image_name} | Objects: {len(objects)} | View: {'BINARY MASK' if pure_mask_view else 'WHITE OVERLAY'} "
        f"| [L-Click] Select [R-Click] Trim/Remove Pin [M-Click/Ctrl+R] Delete Obj [M] View [S] Save"
    )
    cv2.putText(
        display_img, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        display_img, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    cv2.imshow(window_name, display_img)
    key = cv2.waitKey(15) & 0xFF

    if key == ord("s"):
        base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}.bmp")
        cv2.imwrite(output_path, combined_mask)
        print(f"Saved binary CTAn mask (white fill on black background) to: {output_path}")

    elif key == ord("m"):
        pure_mask_view = not pure_mask_view

    elif key in (ord("u"), ord("z")):
        if objects:
            objects.pop()
            active_obj_idx = len(objects) - 1
            print("Undid last object selection.")

    elif key == ord("c"):
        objects.clear()
        active_obj_idx = -1
        print("Cleared all objects.")

    elif key == ord("n"):
        current_idx = (current_idx + 1) % len(image_files)
        load_current_image()

    elif key == ord("p"):
        current_idx = (current_idx - 1) % len(image_files)
        load_current_image()

    elif key in (ord("q"), 27):  # 'q' or ESC
        break

cv2.destroyAllWindows()