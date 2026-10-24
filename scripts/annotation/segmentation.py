import os
import cv2
import numpy as np

# 1. Define input and output directories
input_dir = r"F:\automated-microct-workflow\3_workflow\annotation_testing"
output_dir = r"F:\automated-microct-workflow\3_workflow\annotation_testing\cc_masks"
os.makedirs(output_dir, exist_ok=True)

# 2. Discover images
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])

if not image_files:
    raise FileNotFoundError(f"No images found in {input_dir}")

current_idx = 0
img = None
image_name = ""

# State variables for segmentation
labels = None
num_labels = 0
selected_labels = set()
user_cuts = None
pure_mask_view = False
drawing_cut = False
last_cut_pt = None

# Interactive parameters
otsu_multiplier = 0.95
live_dilation_iters = 0     # Affects the live thresholding and component bridging
export_dilation_iters = 1   # Applied ONLY to the final saved mask
params_locked = False

# Rendering optimization
needs_redraw = True
display_img = None
combined_mask = None

# Camera viewport variables (Zoom & Pan)
VIEW_W, VIEW_H = 1400, 950
zoom_level = 1.0
view_x, view_y = 0.0, 0.0
is_panning = False
last_pan_pt = None

def reset_view():
    global zoom_level, view_x, view_y
    zoom_level = min(VIEW_W / img.shape[1], VIEW_H / img.shape[0])
    view_x = (img.shape[1] - (VIEW_W / zoom_level)) / 2
    view_y = (img.shape[0] - (VIEW_H / zoom_level)) / 2

def screen_to_img(x, y):
    ix = int((x / zoom_level) + view_x)
    iy = int((y / zoom_level) + view_y)
    return max(0, min(ix, img.shape[1] - 1)), max(0, min(iy, img.shape[0] - 1))

def recompute_components():
    global labels, num_labels, selected_labels, needs_redraw
    
    otsu_val, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(img, otsu_val * otsu_multiplier, 255, cv2.THRESH_BINARY)
    
    # Apply LIVE dilation (affects what is clickable/merged on screen)
    if live_dilation_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=live_dilation_iters)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(thresh)
    cv2.drawContours(filled, contours, -1, 255, -1)
    
    filled[user_cuts > 0] = 0
    num_labels, labels = cv2.connectedComponents(filled, connectivity=4)
    selected_labels = {l for l in selected_labels if l < num_labels}
    needs_redraw = True

def mouse_click(event, x, y, flags, param):
    global drawing_cut, last_cut_pt, is_panning, last_pan_pt
    global zoom_level, view_x, view_y, needs_redraw

    ix, iy = screen_to_img(x, y)

    # 1. ZOOM: Scroll Wheel
    if event == cv2.EVENT_MOUSEWHEEL:
        delta = flags >> 16
        zoom_factor = 1.15 if delta > 0 else (1 / 1.15)
        
        ix_before = (x / zoom_level) + view_x
        iy_before = (y / zoom_level) + view_y
        
        zoom_level = max(0.05, min(zoom_level * zoom_factor, 50.0))
        
        view_x = ix_before - (x / zoom_level)
        view_y = iy_before - (y / zoom_level)
        return

    # 2. PANNING: Middle Click & Drag
    if event == cv2.EVENT_MBUTTONDOWN:
        is_panning = True
        last_pan_pt = (x, y)
    elif event == cv2.EVENT_MBUTTONUP:
        is_panning = False
    elif event == cv2.EVENT_MOUSEMOVE and is_panning:
        dx = x - last_pan_pt[0]
        dy = y - last_pan_pt[1]
        view_x -= dx / zoom_level
        view_y -= dy / zoom_level
        last_pan_pt = (x, y)
        return

    # 3. SELECTION: Left Click
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_label = labels[iy, ix]
        if clicked_label > 0: 
            if clicked_label in selected_labels:
                selected_labels.remove(clicked_label)
            else:
                selected_labels.add(clicked_label)
            needs_redraw = True

    # 4. SCALPEL: Right Click & Drag
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing_cut = True
        last_cut_pt = (ix, iy)
        user_cuts[iy, ix] = 255  
        needs_redraw = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing_cut:
        if last_cut_pt is not None:
            cv2.line(user_cuts, last_cut_pt, (ix, iy), 255, thickness=1)
            last_cut_pt = (ix, iy)
            needs_redraw = True
    elif event == cv2.EVENT_RBUTTONUP:
        drawing_cut = False
        last_cut_pt = None
        recompute_components()

def load_current_image():
    global img, image_name, user_cuts, selected_labels
    image_name = image_files[current_idx]
    image_path = os.path.join(input_dir, image_name)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    user_cuts = np.zeros_like(img)
    selected_labels.clear()
    reset_view()
    recompute_components()
    print(f"\nLoaded: {image_name}")

# Initialize UI
load_current_image()
window_name = "CC & Watershed Micro-CT Editor"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, VIEW_W, VIEW_H)
cv2.setMouseCallback(window_name, mouse_click)

print("\n=== CONTROLS ===")
print("Left Click          : Select / Unselect a bone plate")
print("Right Click & Drag  : Draw 'scalpel' line to cut a bridging suture")
print("Scroll Wheel        : Zoom in/out at cursor location")
print("Middle Click & Drag : Pan around the zoomed image")
print("',' / '.' Keys      : Decrease/Increase Otsu Threshold by 1%")
print("'[' / ']' Keys      : Decrease/Increase LIVE Dilation by 1 pixel")
print("'{' / '}' Keys      : Decrease/Increase EXPORT Dilation by 1 pixel (Shift + [ / ])")
print("'l'                 : Lock/Unlock parameter tuning")
print("'r'                 : Reset view to fit screen")
print("'m'                 : Toggle Binary Mask View")
print("'c'                 : Clear all selections and cuts")
print("'s'                 : Save CTAn .bmp mask")
print("'n' / 'p'           : Next / Previous image")
print("'q'                 : Quit")

while True:
    if needs_redraw:
        active_mask = np.isin(labels, list(selected_labels))
        combined_mask = (active_mask * 255).astype(np.uint8)

        if pure_mask_view:
            display_img = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        else:
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Apply 20% translucent red overlay to selected objects
            if selected_labels:
                red_overlay = display_img.copy()
                red_overlay[combined_mask > 0] = [0, 0, 255]
                blended_red = cv2.addWeighted(display_img, 0.8, red_overlay, 0.2, 0)
                display_img[combined_mask > 0] = blended_red[combined_mask > 0]

            # Apply 20% opaque white overlay to scalpel cuts
            if np.any(user_cuts > 0):
                white_scalpel = np.full_like(display_img, 255)
                blended_scalpel = cv2.addWeighted(display_img, 0.8, white_scalpel, 0.2, 0)
                display_img[user_cuts > 0] = blended_scalpel[user_cuts > 0]
            
        needs_redraw = False

    M = np.array([
        [zoom_level, 0, -view_x * zoom_level],
        [0, zoom_level, -view_y * zoom_level]
    ], dtype=np.float32)
    
    view = cv2.warpAffine(display_img, M, (VIEW_W, VIEW_H), flags=cv2.INTER_NEAREST)

    status_text_1 = f"{image_name} | Zoom: {zoom_level:.1f}x | Mode: {'MASK' if pure_mask_view else 'OVERLAY'}"
    status_text_2 = f"Otsu: {otsu_multiplier*100:.0f}% | Live Dilate: {live_dilation_iters}px | Export Dilate: {export_dilation_iters}px | Locked: {params_locked}"
    
    cv2.putText(view, status_text_1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(view, status_text_2, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if params_locked else (255, 255, 255), 1)
    cv2.imshow(window_name, view)
    key = cv2.waitKey(15) & 0xFF

    if key == ord("s"):
        final_mask = combined_mask.copy()
        
        # Apply EXPORT dilation ONLY when saving the final bitmap
        if export_dilation_iters > 0:
            kernel = np.ones((3, 3), np.uint8)
            final_mask = cv2.dilate(final_mask, kernel, iterations=export_dilation_iters)
            
        base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}.bmp")
        cv2.imwrite(output_path, final_mask)
        print(f"Saved binary CTAn mask (Live Dilate: {live_dilation_iters}px, Export Dilate: {export_dilation_iters}px) to: {output_path}")
        
    elif key == ord("m"):
        pure_mask_view = not pure_mask_view
        needs_redraw = True
    elif key == ord("c"):
        selected_labels.clear()
        user_cuts.fill(0)
        recompute_components()
    elif key == ord("r"):
        reset_view()
    elif key == ord("l"):
        params_locked = not params_locked
        needs_redraw = True
    elif key == ord(".") or key == ord(">"):
        if not params_locked:
            otsu_multiplier = min(1.50, otsu_multiplier + 0.01)
            recompute_components()
    elif key == ord(",") or key == ord("<"):
        if not params_locked:
            otsu_multiplier = max(0.01, otsu_multiplier - 0.01)
            recompute_components()
    elif key == ord("]"):
        if not params_locked:
            live_dilation_iters += 1
            recompute_components()
    elif key == ord("["):
        if not params_locked:
            live_dilation_iters = max(0, live_dilation_iters - 1)
            recompute_components()
    elif key == ord("}"):
        if not params_locked:
            export_dilation_iters += 1
            needs_redraw = True
    elif key == ord("{"):
        if not params_locked:
            export_dilation_iters = max(0, export_dilation_iters - 1)
            needs_redraw = True
    elif key == ord("n"):
        current_idx = (current_idx + 1) % len(image_files)
        load_current_image()
    elif key == ord("p"):
        current_idx = (current_idx - 1) % len(image_files)
        load_current_image()
    elif key in (ord('+'), ord('='), ord('-')):
        cx, cy = VIEW_W / 2, VIEW_H / 2
        ix_before = (cx / zoom_level) + view_x
        iy_before = (cy / zoom_level) + view_y
        
        zoom_level = min(zoom_level * 1.15, 50.0) if key != ord('-') else max(zoom_level / 1.15, 0.05)
        
        view_x = ix_before - (cx / zoom_level)
        view_y = iy_before - (cy / zoom_level)
    elif key in (ord("q"), 27):
        break

cv2.destroyAllWindows()