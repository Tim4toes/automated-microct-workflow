import os
import cv2
import numpy as np
import napari
import tkinter as tk
from tkinter import filedialog

# NEW: Import importlib and the correctly named module
import importlib
import export_napari_2 

# Hide the main tkinter window
tk_root = tk.Tk()
tk_root.withdraw()

def process_and_load_dataset(input_dir, viewer=None):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])

    if not image_files:
        print(f"Error: No valid images found in {input_dir}")
        return None

    # Read the first image to get true native spatial dimensions
    sample_img = cv2.imread(os.path.join(input_dir, image_files[0]), cv2.IMREAD_GRAYSCALE)
    height, width = sample_img.shape
    num_slices = len(image_files)

    print(f"Allocating memory for {num_slices} slices ({width}x{height}) in strictly 8-bit grayscale...")

    # Pre-allocate a 3D NumPy array at native resolution
    volume = np.zeros((num_slices, height, width), dtype=np.uint8)

    # Load images directly into the pre-allocated volume
    for i, file_name in enumerate(image_files):
        img_path = os.path.join(input_dir, file_name)
        slice_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        volume[i, :, :] = slice_data
        
        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1}/{num_slices} slices...")

    print("Load complete.")
    
    dataset_name = os.path.basename(os.path.normpath(input_dir))
    
    if viewer is None:
        # Initial Launch
        viewer = napari.Viewer()
        viewer.add_image(volume, name=dataset_name, colormap="gray")
        
        # Bind the open folder hotkey ('o')
        @viewer.bind_key('o')
        def open_new_dataset(v):
            new_dir = filedialog.askdirectory(title="Select New Dataset Folder")
            if new_dir:
                v.layers.clear() 
                print(f"\n--- Switched to Dataset: {os.path.basename(os.path.normpath(new_dir))} ---")
                process_and_load_dataset(new_dir, viewer=v)
                
        # Bind the export hotkey ('e')
        @viewer.bind_key('e')
        def trigger_export(v):
            # NEW: Force Python to reload the export script from the hard drive every time you press 'e'
            importlib.reload(export_napari_2)
            base_output_dir = r"D:\automated-microct-workflow\3_workflow\annotation_testing\nninteractive_masks"
            export_napari_2.export_combined_masks(v, base_output_dir)
                
        napari.run()
    else:
        # Loading into existing session
        viewer.add_image(volume, name=dataset_name, colormap="gray")

# 1. Define your initial dataset path
initial_input_dir = r"D:\automated-microct-workflow\3_workflow\annotation_testing\input_data\dataset1_sost"

if __name__ == "__main__":
    process_and_load_dataset(initial_input_dir)