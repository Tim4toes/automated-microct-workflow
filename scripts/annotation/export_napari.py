import os
import cv2
import napari
import numpy as np

def export_combined_masks(viewer, base_output_dir):
    print("\n--- Starting Mask Export ---")
    
    # Grab the Raw Image Layer
    image_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Image)]
    if not image_layers:
        print("Error: No Raw Image dataset found in the viewer!")
        return

    raw_layer = image_layers[0]
    dataset_name = raw_layer.name

    # Create the exact dataset folder in the output directory
    output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Locate original files on disk to copy exact filenames and dimensions
    input_dir = os.path.join(r"D:\automated-microct-workflow\3_workflow\annotation_testing\input_data", dataset_name)
    valid_exts = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")
    
    if os.path.exists(input_dir):
        raw_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])
        if not raw_files or len(raw_files) != raw_layer.data.shape[0]:
             print("Warning: Could not match exact filenames from disk. Using generic slice numbering.")
             raw_files = [f"slice_{i:04d}.bmp" for i in range(raw_layer.data.shape[0])]
    else:
        raw_files = [f"slice_{i:04d}.bmp" for i in range(raw_layer.data.shape[0])]

    # Grab and Combine all Labels (Mask) Layers
    labels_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Labels)]
    if not labels_layers:
        print("Error: No Labels layers found to export!")
        return

    # Merge all objects across layers into a single boolean 3D volume
    combined_3d = np.zeros(labels_layers[0].data.shape, dtype=bool)
    for layer in labels_layers:
        combined_3d |= (layer.data > 0)

    num_slices = combined_3d.shape[0]
    print(f"Merging {len(labels_layers)} mask layer(s) across {num_slices} slices into '{output_dir}'...")

    # --- NEW SETTINGS ---
    # Positive Y_SHIFT moves the mask DOWN (in pixels) to fix the top overhang
    Y_SHIFT = 1
    # Number of iterations to dilate the mask outward (1 iter = 1 pixel of expansion)
    DILATION_ITERS = 2

    # Define the 3x3 kernel used for 1-pixel dilation
    dilation_kernel = np.ones((3, 3), np.uint8)

    # Export Slice-by-Slice
    for i in range(num_slices):
        file_name = raw_files[i]
        
        orig_h, orig_w = None, None
        if os.path.exists(input_dir) and len(raw_files) == num_slices:
            raw_path = os.path.join(input_dir, file_name)
            raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
            if raw_img is not None:
                 orig_h, orig_w = raw_img.shape
        
        if orig_h is None or orig_w is None:
            orig_h, orig_w = raw_layer.data.shape[1], raw_layer.data.shape[2]

        # Convert boolean mask to 8-bit format with strictly binary 0 and 255 values
        slice_mask = (combined_3d[i, :, :] * 255).astype(np.uint8)

        if slice_mask.shape != (orig_h, orig_w):
            slice_mask = cv2.resize(slice_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Apply the downward mathematical shift to fix nnInteractive resampling bias
        if Y_SHIFT != 0:
            shifted_mask = np.zeros_like(slice_mask)
            if Y_SHIFT > 0:
                shifted_mask[Y_SHIFT:, :] = slice_mask[:-Y_SHIFT, :]
            else:
                shifted_mask[:Y_SHIFT, :] = slice_mask[-Y_SHIFT:, :]
            slice_mask = shifted_mask

        # Apply morphological dilation to puff the mask outward by 1 pixel
        if DILATION_ITERS > 0:
            slice_mask = cv2.dilate(slice_mask, dilation_kernel, iterations=DILATION_ITERS)

        out_name = os.path.splitext(file_name)[0] + ".bmp"
        out_path = os.path.join(output_dir, out_name)
        
        # Save as an 8-bit .bmp using OpenCV 
        cv2.imwrite(out_path, slice_mask)

        if (i + 1) % 50 == 0 or (i + 1) == num_slices:
            print(f"Exported {i + 1}/{num_slices} slices...")

    print(f"Export complete! Masks saved to: {output_dir}\n")