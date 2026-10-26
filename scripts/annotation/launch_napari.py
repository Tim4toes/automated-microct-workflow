import os
import cv2
import numpy as np
import napari

# 1. Define your dataset path
input_dir = r"D:\automated-microct-workflow\3_workflow\annotation_testing\input_data\dataset1_sost"

# 2. Discover and sort images
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])

if not image_files:
    raise FileNotFoundError(f"No valid images found in {input_dir}")

# 3. Read the first image to get spatial dimensions
sample_img = cv2.imread(os.path.join(input_dir, image_files[0]), cv2.IMREAD_GRAYSCALE)
height, width = sample_img.shape
num_slices = len(image_files)

print(f"Allocating memory for {num_slices} slices ({width}x{height}) in strictly 8-bit grayscale...")

# 4. Pre-allocate a 3D NumPy array to prevent memory fragmentation
volume = np.zeros((num_slices, height, width), dtype=np.uint8)

# 5. Load images directly into the pre-allocated volume
for i, file_name in enumerate(image_files):
    img_path = os.path.join(input_dir, file_name)
    # Force grayscale reading (reduces RAM footprint by ~66%)
    slice_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    volume[i, :, :] = slice_data
    
    if (i + 1) % 100 == 0:
        print(f"Loaded {i + 1}/{num_slices} slices...")

print("Load complete. Launching Napari...")

# 6. Launch Napari and pass the optimized 3D volume
viewer = napari.Viewer()
viewer.add_image(volume, name=os.path.basename(input_dir), colormap="gray")
napari.run()