import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# Hide the main tkinter window
tk_root = tk.Tk()
tk_root.withdraw()

# 1. Define initial input directory
input_dir = r"D:\automated-microct-workflow\3_workflow\annotation_testing\input_data\dataset1_sost"
dataset_name = os.path.basename(os.path.normpath(input_dir))
output_dir = os.path.join(r"D:\automated-microct-workflow\3_workflow\annotation_testing\cc_masks", dataset_name)

# 2. Discover images
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])

if not image_files:
    raise FileNotFoundError(f"No images found in {input_dir}")

sample_img = cv2.imread(os.path.join(input_dir, image_files[0]), cv2.IMREAD_GRAYSCALE)
IMAGE_SHAPE = sample_img.shape

# Volumetric & Memory State Variables
current_idx = 0
img = None
image_name = ""
z_min = 0
z_max = len(image_files) - 1

slice_cuts = {}            
slice_adds = {}
slice_vents = {}             # NEW: Tracks venting scalpels (applied before fill)
slice_approved_masks = {}  
keyframes = set()            # Tracks slices where user explicitly clicked/edited/interpolated
interpolated_slices = set()  # Protects rigid SDF tunnel masks from being overwritten by Otsu

# State variables for active slice segmentation
labels = None
num_labels = 0
selected_labels = set()
user_cuts = None
user_adds = None
user_vents = None            # NEW: Active slice venting scalpels
pure_mask_view = False

# Drawing state
drawing_cut = False
last_cut_pt = None
drawing_add = False
last_add_pt = None
drawing_vent = False         # NEW: Vent drawing state
last_vent_pt = None

# Interactive parameters
live_dilation_iters = 0     
export_dilation_iters = 1   
params_locked = True
persistent_edits = False    # Toggles automatic carry-over of cuts/bridges
is_reversed = False         # Tracks if dataset Z-axis is flipped

# Rendering variables
needs_redraw = True
display_img = None
combined_mask = None
VIEW_W, VIEW_H = 1400, 950
zoom_level = 1.0
view_x, view_y = 0.0, 0.0
is_panning = False
last_pan_pt = None

# --- Workspace Management ---

def save_workspace():
    os.makedirs(output_dir, exist_ok=True)
    session_file = os.path.join(output_dir, f"{dataset_name}_workspace.npz")
    # Save direction state alongside arrays
    save_dict = {"is_reversed": np.array(is_reversed)}
    
    # Cast arrays to booleans (True/False) to cut RAM and file size by 87.5%
    for k, v in slice_cuts.items(): save_dict[f"cut_{k}"] = v.astype(np.bool_)
    for k, v in slice_adds.items(): save_dict[f"add_{k}"] = v.astype(np.bool_)
    for k, v in slice_vents.items(): save_dict[f"vent_{k}"] = v.astype(np.bool_)
    for k, v in slice_approved_masks.items(): save_dict[f"mask_{k}"] = v.astype(np.bool_)
    
    # Save metadata for keyframes, interpolated regions, and bounds
    save_dict["meta_keyframes"] = np.array(list(keyframes), dtype=np.int32)
    save_dict["meta_interpolated"] = np.array(list(interpolated_slices), dtype=np.int32)
    save_dict["meta_bounds"] = np.array([z_min, z_max], dtype=np.int32)
    
    np.savez_compressed(session_file, **save_dict)
    print(f"\n--- Workspace successfully saved to {session_file} ---")

def load_workspace():
    global slice_cuts, slice_adds, slice_vents, slice_approved_masks, keyframes, interpolated_slices, z_min, z_max, is_reversed
    slice_cuts.clear()
    slice_adds.clear()
    slice_vents.clear()
    slice_approved_masks.clear()
    keyframes.clear()
    interpolated_slices.clear()
    
    session_file = os.path.join(output_dir, f"{dataset_name}_workspace.npz")
    
    if os.path.exists(session_file):
        data = np.load(session_file)

        # Align dataset physical direction with the saved workspace state
        if "is_reversed" in data.files:
            saved_rev = bool(data["is_reversed"])
            if saved_rev != is_reversed:
                is_reversed = saved_rev
                image_files.reverse()

        for key in data.files:
            # FIX 1: Skip "is_reversed" so key.split('_')[1] does not crash with ValueError
            if key == "is_reversed":
                continue
            elif key == "meta_keyframes":
                keyframes.update(data[key].tolist())
                continue
            elif key == "meta_interpolated":
                interpolated_slices.update(data[key].tolist())
                continue
            elif key == "meta_bounds":
                bounds = data[key].tolist()
                z_min, z_max = bounds[0], min(bounds[1], len(image_files) - 1)
                continue
                
            idx = int(key.split('_')[1])
            # Cast back to 8-bit and multiply by 255 to restore the 0/255 pixel values
            restored_array = data[key].astype(np.uint8) * 255
            
            if key.startswith("cut_"): slice_cuts[idx] = restored_array
            elif key.startswith("add_"): slice_adds[idx] = restored_array
            elif key.startswith("vent_"): slice_vents[idx] = restored_array
            elif key.startswith("mask_"): slice_approved_masks[idx] = restored_array
            
        # Backward compatibility: if loading an older workspace without keyframe metadata
        if not keyframes:
            keyframes.update(slice_cuts.keys())
            keyframes.update(slice_adds.keys())
            keyframes.update(slice_vents.keys())
            if 0 in slice_approved_masks:
                keyframes.add(0)
                
        print(f"\n--- Loaded existing workspace for '{dataset_name}' ---")
    else:
        print(f"\n--- No saved workspace found at {session_file} ---")

# --- Core Segmentation Logic ---

def get_overlapping_labels(target_labels_array, reference_mask):
    if reference_mask is None:
        return set()
    overlapping_pixels = target_labels_array[reference_mask > 0]
    unique_labels = set(np.unique(overlapping_pixels))
    unique_labels.discard(0) 
    return unique_labels

def generate_raw_components(image_array, cuts_array, adds_array, vents_array):
    _, thresh = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 1. Apply user bridges FIRST, closing outer gaps before contour detection
    thresh[adds_array > 0] = 255
    
    # 2. Apply venting scalpels BEFORE filling to pop massive cavities
    thresh[vents_array > 0] = 0
    
    if live_dilation_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=live_dilation_iters)
    
    # 3. Find external contours. Because gaps are now bridged, internal holes will be filled automatically.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(thresh)
    cv2.drawContours(filled, contours, -1, 255, -1)
    
    # 4. Apply scalpel cuts LAST so they cleanly sever the filled bone segments, re-apply vents for clean cuts
    filled[cuts_array > 0] = 0    
    filled[vents_array > 0] = 0    
    return cv2.connectedComponents(filled, connectivity=4)

def update_approved_mask():
    global combined_mask
    active_mask = np.isin(labels, list(selected_labels))
    combined_mask = (active_mask * 255).astype(np.uint8)
    if selected_labels:
        slice_approved_masks[current_idx] = combined_mask.copy()
    else:
        slice_approved_masks.pop(current_idx, None)

def invalidate_future_caches():
    # Only invalidate auto-propagated caches up to the next user keyframe
    future_keyframes = [k for k in keyframes if k > current_idx]
    stop_idx = min(future_keyframes) if future_keyframes else len(image_files)
    
    future_keys = [k for k in slice_approved_masks if current_idx < k < stop_idx]
    for k in future_keys:
        del slice_approved_masks[k]
        interpolated_slices.discard(k)

def recompute_components(auto_select_from_mask=None):
    global labels, num_labels, selected_labels, combined_mask, needs_redraw
    num_labels, labels = generate_raw_components(img, user_cuts, user_adds, user_vents)
    
    # If this slice holds a rigid interpolated mask, preserve it until edited
    if current_idx in interpolated_slices and current_idx in slice_approved_masks:
        combined_mask = slice_approved_masks[current_idx].copy()
        selected_labels = get_overlapping_labels(labels, combined_mask)
        needs_redraw = True
        return

    # FIX 2: Re-project from spatial mask instead of raw label integers so cuts/bridges
    # do not shift component IDs to unrelated bone islands
    if auto_select_from_mask is not None:
        selected_labels = get_overlapping_labels(labels, auto_select_from_mask)
    elif selected_labels and combined_mask is not None:
        ref_mask = combined_mask.copy()
        ref_mask[user_cuts > 0] = 0
        ref_mask[user_vents > 0] = 0
        selected_labels = get_overlapping_labels(labels, ref_mask)
    else:
        selected_labels = set()
        
    update_approved_mask()
    needs_redraw = True

def get_signed_distance_field(mask):
    # Calculate distance to the closest background pixel (positive inside bone)
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    # Calculate distance to the closest bone pixel (negative outside bone)
    inv_mask = cv2.bitwise_not(mask)
    outside = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 3)
    return inside - outside

def interpolate_gap(start_idx, end_idx):
    mask_A = slice_approved_masks[start_idx]
    mask_B = slice_approved_masks[end_idx]
    
    sdf_A = get_signed_distance_field(mask_A)
    sdf_B = get_signed_distance_field(mask_B)
    
    steps = end_idx - start_idx
    for i in range(1, steps):
        weight = i / steps
        # Blend the 3D distance topologies
        interp_sdf = sdf_A * (1.0 - weight) + sdf_B * weight
        interp_mask = (interp_sdf > 0).astype(np.uint8) * 255
        
        target_idx = start_idx + i
        slice_approved_masks[target_idx] = interp_mask
        interpolated_slices.add(target_idx)
        keyframes.discard(target_idx)
        
        # Clear legacy manual edits in the gap so they do not conflict
        if target_idx in slice_cuts: del slice_cuts[target_idx]
        if target_idx in slice_adds: del slice_adds[target_idx]
        if target_idx in slice_vents: del slice_vents[target_idx]

def load_current_image(propagate_from=None):
    global img, image_name, user_cuts, user_adds, user_vents, selected_labels
    image_name = image_files[current_idx]
    img = cv2.imread(os.path.join(input_dir, image_name), cv2.IMREAD_GRAYSCALE)
    
    # FIX 5: Track if persistent edits modified the target slice so selection propagates cleanly
    persisted_change = False
    if persistent_edits and propagate_from is not None and current_idx not in interpolated_slices:
        if propagate_from in slice_cuts:
            if current_idx in slice_cuts:
                slice_cuts[current_idx] = cv2.bitwise_or(slice_cuts[current_idx], slice_cuts[propagate_from])
            else:
                slice_cuts[current_idx] = slice_cuts[propagate_from].copy()
            persisted_change = True
                
        if propagate_from in slice_adds:
            if current_idx in slice_adds:
                slice_adds[current_idx] = cv2.bitwise_or(slice_adds[current_idx], slice_adds[propagate_from])
            else:
                slice_adds[current_idx] = slice_adds[propagate_from].copy()
            persisted_change = True

        if propagate_from in slice_vents:
            if current_idx in slice_vents:
                slice_vents[current_idx] = cv2.bitwise_or(slice_vents[current_idx], slice_vents[propagate_from])
            else:
                slice_vents[current_idx] = slice_vents[propagate_from].copy()
            persisted_change = True
                
    user_cuts = slice_cuts.get(current_idx, np.zeros(IMAGE_SHAPE, dtype=np.uint8)).copy()
    user_adds = slice_adds.get(current_idx, np.zeros(IMAGE_SHAPE, dtype=np.uint8)).copy()
    user_vents = slice_vents.get(current_idx, np.zeros(IMAGE_SHAPE, dtype=np.uint8)).copy()
    
    ref_mask = None
    if persisted_change and propagate_from in slice_approved_masks:
        ref_mask = slice_approved_masks[propagate_from]
    elif current_idx in slice_approved_masks:
        ref_mask = slice_approved_masks[current_idx]
    elif propagate_from is not None and propagate_from in slice_approved_masks:
        ref_mask = slice_approved_masks[propagate_from]
        
    if propagate_from is None:
        reset_view()
        
    recompute_components(auto_select_from_mask=ref_mask)

# --- UI & Interaction ---

def reset_view():
    global zoom_level, view_x, view_y
    zoom_level = min(VIEW_W / img.shape[1], VIEW_H / img.shape[0])
    view_x = (img.shape[1] - (VIEW_W / zoom_level)) / 2
    view_y = (img.shape[0] - (VIEW_H / zoom_level)) / 2

def screen_to_img(x, y):
    ix = int((x / zoom_level) + view_x)
    iy = int((y / zoom_level) + view_y)
    return max(0, min(ix, img.shape[1] - 1)), max(0, min(iy, img.shape[0] - 1))

def mouse_click(event, x, y, flags, param):
    global drawing_cut, last_cut_pt, drawing_add, last_add_pt, drawing_vent, last_vent_pt
    global is_panning, last_pan_pt, zoom_level, view_x, view_y, needs_redraw

    ix, iy = screen_to_img(x, y)

    if event == cv2.EVENT_MOUSEWHEEL:
        delta = flags >> 16
        zoom_factor = 1.15 if delta > 0 else (1 / 1.15)
        ix_before, iy_before = (x / zoom_level) + view_x, (y / zoom_level) + view_y
        zoom_level = max(0.05, min(zoom_level * zoom_factor, 50.0))
        view_x, view_y = ix_before - (x / zoom_level), iy_before - (y / zoom_level)
        return
    elif event == cv2.EVENT_MBUTTONDOWN:
        is_panning = True
        last_pan_pt = (x, y)
    elif event == cv2.EVENT_MBUTTONUP:
        is_panning = False
    elif event == cv2.EVENT_MOUSEMOVE and is_panning:
        view_x -= (x - last_pan_pt[0]) / zoom_level
        view_y -= (y - last_pan_pt[1]) / zoom_level
        last_pan_pt = (x, y)
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_label = labels[iy, ix]
        if clicked_label > 0: 
            # Clicking on an interpolated slice snaps it back to dynamic Otsu mode
            interpolated_slices.discard(current_idx)
            if clicked_label in selected_labels:
                selected_labels.remove(clicked_label)
            else:
                selected_labels.add(clicked_label)
            
            if selected_labels:
                keyframes.add(current_idx)
            else:
                keyframes.discard(current_idx)
                
            update_approved_mask()
            invalidate_future_caches()
            needs_redraw = True

    elif event == cv2.EVENT_RBUTTONDOWN:
        interpolated_slices.discard(current_idx)
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            drawing_vent = True
            last_vent_pt = (ix, iy)
            user_vents[iy, ix] = 255
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            drawing_add = True
            last_add_pt = (ix, iy)
            user_adds[iy, ix] = 255
        else:
            drawing_cut = True
            last_cut_pt = (ix, iy)
            user_cuts[iy, ix] = 255  
        needs_redraw = True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_cut and last_cut_pt is not None:
            cv2.line(user_cuts, last_cut_pt, (ix, iy), 255, thickness=1)
            last_cut_pt = (ix, iy)
            needs_redraw = True
        elif drawing_add and last_add_pt is not None:
            cv2.line(user_adds, last_add_pt, (ix, iy), 255, thickness=1)
            last_add_pt = (ix, iy)
            needs_redraw = True
        elif drawing_vent and last_vent_pt is not None:
            cv2.line(user_vents, last_vent_pt, (ix, iy), 255, thickness=1)
            last_vent_pt = (ix, iy)
            needs_redraw = True
            
    elif event == cv2.EVENT_RBUTTONUP:
        if drawing_cut:
            drawing_cut = False
            last_cut_pt = None
            if np.any(user_cuts > 0):
                slice_cuts[current_idx] = user_cuts.copy()
                keyframes.add(current_idx)
        if drawing_add:
            drawing_add = False
            last_add_pt = None
            if np.any(user_adds > 0):
                slice_adds[current_idx] = user_adds.copy()
                keyframes.add(current_idx)
        if drawing_vent:
            drawing_vent = False
            last_vent_pt = None
            if np.any(user_vents > 0):
                slice_vents[current_idx] = user_vents.copy()
                keyframes.add(current_idx)
                
        recompute_components()
        invalidate_future_caches()

# --- Initialize Windows ---
load_current_image()

# Render Separate Controls Window
controls_data = [
    ("=== MICRO-CT SEGMENTATION CONTROLS ===", ""),
    ("", ""),
    ("Left Click", "Select / Unselect a bone plate"),
    ("Right Click & Drag", "Draw standard 'scalpel' (White) to sever a suture"),
    ("Ctrl + Right Click", "Draw 'venting scalpel' (Cyan) to pop massive cavities BEFORE fill"),
    ("Shift + Right Click", "Draw 'bridge' line (Green) to fill gaps/connect bones"),
    ("Scroll Wheel", "Zoom in/out"),
    ("Middle Click & Drag", "Pan image"),
    ("", ""),
    ("'[' / ']' Keys", "Decrease/Increase LIVE Dilation by 1 px"),
    ("'{' / '}' Keys", "Decrease/Increase EXPORT Dilation by 1 px"),
    ("'v' / 'b' Keys", "Set START / END Slice boundary"),
    ("'x'", "Reset boundaries to full volume"),
    ("", ""),
    ("'e'", "BATCH EXPORT entire volume"),
    ("'s'", "SAVE CURRENT SINGLE SLICE only"),
    ("'w'", "SAVE WORKSPACE (saves all cuts & memory)"),
    ("'a'", "LOAD WORKSPACE (restores saved session)"),
    ("'j' / 'h' Keys", "Next / Previous image (Auto-propagates)"),
    ("", ""),
    ("'u'", "Undo SCALPEL & BRIDGES only on this slice"),
    ("'p'", "Toggle PERSISTENT edits (auto-copy to next slice)"),
    ("'i'", "INTERPOLATE regions from previous approved slice"),
    ("'f'", "FLIP Z-direction (Reverse dataset processing order)"),
    ("'c'", "Clear BOTH selections and cuts on this slice"),
    ("'r'", "Reset camera zoom and pan"),
    ("'m'", "Toggle Pure Mask View"),
    ("'o'", "Open a new dataset folder"),
    ("'l'", "Lock/Unlock parameter tuning"),
    ("'q'", "Quit")
]

controls_img = np.zeros((810, 750, 3), dtype=np.uint8)
y_offset = 30

for key_text, action_text in controls_data:
    if action_text == "":
        cv2.putText(controls_img, key_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    else:
        cv2.putText(controls_img, key_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(controls_img, action_text, (240, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y_offset += 25

cv2.imshow("Controls Menu", controls_img)

# Main Viewer Window
window_name = "Volumetric CC Editor"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(window_name, VIEW_W, VIEW_H)
cv2.setMouseCallback(window_name, mouse_click)

while True:
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.getWindowProperty("Controls Menu", cv2.WND_PROP_VISIBLE) < 1:
        break

    rect = cv2.getWindowImageRect(window_name)
    if rect[2] > 0 and rect[3] > 0:
        VIEW_W, VIEW_H = rect[2], rect[3]

    if needs_redraw:
        if pure_mask_view:
            display_img = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        else:
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if np.any(combined_mask > 0):
                red_overlay = display_img.copy()
                red_overlay[combined_mask > 0] = [0, 0, 255]
                blended_red = cv2.addWeighted(display_img, 0.8, red_overlay, 0.2, 0)
                display_img[combined_mask > 0] = blended_red[combined_mask > 0]

            # 20% Opacity Green for Bridge Lines
            if np.any(user_adds > 0):
                add_overlay = display_img.copy()
                add_overlay[user_adds > 0] = [0, 255, 0]
                blended_add = cv2.addWeighted(display_img, 0.8, add_overlay, 0.2, 0)
                display_img[user_adds > 0] = blended_add[user_adds > 0]

            # 20% Opacity White for Scalpel Cuts
            if np.any(user_cuts > 0):
                cut_overlay = display_img.copy()
                cut_overlay[user_cuts > 0] = [255, 255, 255]
                blended_cut = cv2.addWeighted(display_img, 0.8, cut_overlay, 0.2, 0)
                display_img[user_cuts > 0] = blended_cut[user_cuts > 0]

            # 20% Opacity Cyan for Venting Scalpels
            if np.any(user_vents > 0):
                vent_overlay = display_img.copy()
                vent_overlay[user_vents > 0] = [255, 255, 0] # Cyan in BGR
                blended_vent = cv2.addWeighted(display_img, 0.8, vent_overlay, 0.2, 0)
                display_img[user_vents > 0] = blended_vent[user_vents > 0]

        needs_redraw = False

    M = np.array([[zoom_level, 0, -view_x * zoom_level], [0, zoom_level, -view_y * zoom_level]], dtype=np.float32)
    view = cv2.warpAffine(display_img, M, (VIEW_W, VIEW_H), flags=cv2.INTER_NEAREST)

    has_active_edits = (
        (current_idx in slice_cuts)
        or (current_idx in slice_adds)
        or (current_idx in slice_vents)
        or (current_idx in interpolated_slices)
        or drawing_cut
        or drawing_add
        or drawing_vent
    )
    text_color_1 = (0, 0, 255) if has_active_edits else (255, 255, 255)
    mode_label = "MASK" if pure_mask_view else "OVERLAY"
    if current_idx in interpolated_slices:
        mode_label += " [INTERP]"

    status_1 = f"[{current_idx+1}/{len(image_files)}] {image_name} | BOUNDS: {z_min+1} to {z_max+1} | Rev: {'YES' if is_reversed else 'NO'} | Mode: {mode_label}"
    status_2 = f"Live Dil: {live_dilation_iters}px | Export Dil: {export_dilation_iters}px | Locked: {params_locked} | Persist: {'ON' if persistent_edits else 'OFF'}"
    cv2.putText(view, status_1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color_1, 1)
    cv2.putText(view, status_2, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if params_locked else (255, 255, 255), 1)

    cv2.imshow(window_name, view)
    key = cv2.waitKey(15) & 0xFF

    if key == ord("w"):
        save_workspace()

    elif key == ord("a"):
        load_workspace()
        load_current_image()    
        
    elif key == ord("e"):
        start_b, end_b = min(z_min, z_max), max(z_min, z_max)
        print(f"\n--- Starting Batch Export (Slices {start_b+1} to {end_b+1}) ---")
        os.makedirs(output_dir, exist_ok=True)
        last_valid_mask = None
        
        # Pre-allocate static arrays outside the loop to vastly improve speed
        empty_mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        export_kernel = np.ones((3, 3), np.uint8) if export_dilation_iters > 0 else None
        
        for i in range(len(image_files)):
            out_path = os.path.join(output_dir, os.path.splitext(image_files[i])[0] + ".bmp")
            
            if i < start_b or i > end_b:
                Image.fromarray(empty_mask).convert('1').save(out_path)
                continue
                
            if i in slice_approved_masks:
                raw_mask = slice_approved_masks[i]
            else:
                temp_img = cv2.imread(os.path.join(input_dir, image_files[i]), cv2.IMREAD_GRAYSCALE)
                c_array = slice_cuts.get(i, np.zeros(IMAGE_SHAPE, dtype=np.uint8))
                a_array = slice_adds.get(i, np.zeros(IMAGE_SHAPE, dtype=np.uint8))
                v_array = slice_vents.get(i, np.zeros(IMAGE_SHAPE, dtype=np.uint8))
                _, temp_labels = generate_raw_components(temp_img, c_array, a_array, v_array)
                temp_sel = get_overlapping_labels(temp_labels, last_valid_mask)
                raw_mask = (np.isin(temp_labels, list(temp_sel)) * 255).astype(np.uint8)
                # CRITICAL: Do NOT save raw_mask to slice_approved_masks here, or RAM will explode
            
            # FIX 3: Keep last_valid_mask un-dilated for accurate slice-to-slice propagation
            last_valid_mask = raw_mask.copy()
            export_mask = raw_mask.copy()
            
            if export_kernel is not None:
                export_mask = cv2.dilate(export_mask, export_kernel, iterations=export_dilation_iters)
                
            Image.fromarray(export_mask).convert('1').save(out_path)
            if i % 50 == 0:
                print(f"Exported {i+1}/{len(image_files)}...")
                
        print("--- Batch Export Complete! ---\n")
        
    elif key == ord("f"):
        is_reversed = not is_reversed
        total = len(image_files)
        image_files.reverse()
        
        def remap_dict(d):
            return {(total - 1 - k): v for k, v in d.items()}
            
        slice_cuts = remap_dict(slice_cuts)
        slice_adds = remap_dict(slice_adds)
        slice_vents = remap_dict(slice_vents)
        slice_approved_masks = remap_dict(slice_approved_masks)
        
        keyframes = set((total - 1 - k) for k in keyframes)
        interpolated_slices = set((total - 1 - k) for k in interpolated_slices)
        
        current_idx = total - 1 - current_idx
        z_min, z_max = total - 1 - z_max, total - 1 - z_min
        
        load_current_image()
        print(f"\n--- Dataset Z-direction flipped (Reversed: {is_reversed}) ---")
    
    elif key == ord("o"):
        new_dir = filedialog.askdirectory(initialdir=os.path.dirname(input_dir), title="Select New Dataset Folder")
        
        if new_dir:
            input_dir = new_dir
            dataset_name = os.path.basename(os.path.normpath(input_dir))
            output_dir = os.path.join(r"D:utomated-microct-workflow\3_workflow\annotation_testing\cc_masks", dataset_name)
            
            image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])
            if not image_files:
                print(f"Error: No valid images found in {input_dir}")
            else:
                sample_img = cv2.imread(os.path.join(input_dir, image_files[0]), cv2.IMREAD_GRAYSCALE)
                IMAGE_SHAPE = sample_img.shape
                current_idx = 0
                z_min = 0
                z_max = len(image_files) - 1
                
                # FIX 4: Reset direction and persistent state for the newly loaded dataset
                is_reversed = False
                persistent_edits = False

                # Prevent cross-contamination between datasets
                slice_cuts.clear()
                slice_adds.clear()
                slice_vents.clear()
                slice_approved_masks.clear()
                keyframes.clear()
                interpolated_slices.clear()
                
                load_current_image()
                print(f"\n--- Switched to Dataset: {dataset_name} ---")

    elif key == ord("v"):
        z_min = current_idx
        needs_redraw = True
        print(f"Start bound set to slice {z_min+1}")
    elif key == ord("b"):
        z_max = current_idx
        needs_redraw = True
        print(f"End bound set to slice {z_max+1}")
    elif key == ord("x"):
        z_min = 0
        z_max = len(image_files) - 1
        needs_redraw = True
        print(f"Bounds reset to full volume: 1 to {len(image_files)}")
    elif key == ord("j"):
        prev = current_idx
        current_idx = (current_idx + 1) % len(image_files)
        load_current_image(propagate_from=prev)
    elif key == ord("h"):
        prev = current_idx
        current_idx = (current_idx - 1) % len(image_files)
        load_current_image(propagate_from=prev)
    elif key == ord("m"):
        pure_mask_view = not pure_mask_view
        needs_redraw = True
    elif key == ord("p"):
        persistent_edits = not persistent_edits
        needs_redraw = True    
    elif key == ord("c"):
        selected_labels.clear()
        if combined_mask is not None:
            combined_mask.fill(0)
        user_cuts.fill(0)
        user_adds.fill(0)
        user_vents.fill(0)
        if current_idx in slice_cuts: del slice_cuts[current_idx]
        if current_idx in slice_adds: del slice_adds[current_idx]
        if current_idx in slice_vents: del slice_vents[current_idx]
        slice_approved_masks.pop(current_idx, None)
        keyframes.discard(current_idx)
        interpolated_slices.discard(current_idx)
        recompute_components()
        invalidate_future_caches()
    elif key == ord("i"):
        # Scan backward to find the closest user keyframe with a valid selection
        prev_idx = None
        for p in range(current_idx - 1, -1, -1):
            if p in keyframes and p in slice_approved_masks:
                prev_idx = p
                break
        # Fallback to the earliest approved slice if no explicit keyframe exists behind current_idx
        if prev_idx is None:
            for p in range(current_idx - 1, -1, -1):
                if p in slice_approved_masks:
                    prev_idx = p
                    break
        
        if prev_idx is not None and current_idx in slice_approved_masks:
            if current_idx - prev_idx <= 1:
                print("Notice: Selected keyframes are adjacent; no intermediate slices to interpolate.")
            else:
                print(f"Interpolating rigid tunnel from slice {prev_idx+1} to {current_idx+1}...")
                keyframes.add(prev_idx)
                keyframes.add(current_idx)
                interpolate_gap(prev_idx, current_idx)
                print("Interpolation complete. Scroll back with 'h' to inspect.")
                needs_redraw = True
        else:
            print("Error: Ensure you have an active selection on this slice AND a previous keyframe slice.")

    elif key == ord("u"):
        user_cuts.fill(0)
        user_adds.fill(0)
        user_vents.fill(0)
        if current_idx in slice_cuts: del slice_cuts[current_idx]
        if current_idx in slice_adds: del slice_adds[current_idx]
        if current_idx in slice_vents: del slice_vents[current_idx]
        interpolated_slices.discard(current_idx)
        recompute_components()
        invalidate_future_caches()
    elif key == ord("r"):
        reset_view()
    elif key == ord("l"):
        params_locked = not params_locked
        needs_redraw = True
    elif key == ord("]"):
        # FIX 6: Invalidate downstream caches when live dilation changes
        if not params_locked:
            live_dilation_iters += 1
            recompute_components()
            invalidate_future_caches()
    elif key == ord("["):
        if not params_locked:
            live_dilation_iters = max(0, live_dilation_iters - 1)
            recompute_components()
            invalidate_future_caches()
    elif key == ord("}"):
        if not params_locked: export_dilation_iters += 1; needs_redraw = True
    elif key == ord("{"):
        if not params_locked: export_dilation_iters = max(0, export_dilation_iters - 1); needs_redraw = True
    elif key in (ord('s'), ord('+'), ord('='), ord('-'), ord('q'), 27):
        if key == ord('s'):
            final_mask = combined_mask.copy()
            if export_dilation_iters > 0:
                kernel = np.ones((3, 3), np.uint8)
                final_mask = cv2.dilate(final_mask, kernel, iterations=export_dilation_iters)
            
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.bmp")
            Image.fromarray(final_mask).convert('1').save(out_path)
            print(f"Saved active slice to: {out_path}")
        elif key in (ord("q"), 27):
            break
        else:
            cx, cy = VIEW_W / 2, VIEW_H / 2
            ix_before, iy_before = (cx / zoom_level) + view_x, (cy / zoom_level) + view_y
            zoom_level = min(zoom_level * 1.15, 50.0) if key != ord('-') else max(zoom_level / 1.15, 0.05)
            view_x, view_y = ix_before - (cx / zoom_level), iy_before - (cy / zoom_level)

cv2.destroyAllWindows()
