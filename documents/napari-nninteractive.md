# Napari & nnInteractive Setup Guide

This guide explains how to set up and use **Napari** alongside the **nnInteractive** plugin for AI-assisted 3D segmentation of micro-CT datasets. It also includes instructions for utilizing a custom Python launcher script designed to prevent memory crashes.

---

## 1. What are these tools?

*   **Napari:** A fast, interactive, multi-dimensional image viewer for Python. It allows you to explore massive 3D volumetric datasets effortlessly.
*   **napari-nninteractive:** A plugin for Napari that provides deep learning-assisted (AI) 3D segmentation. It allows you to use pre-trained models to semi-automatically segment structures (like your *Sost* knockout bone datasets) using minimal user interaction (e.g., scribbles or bounding boxes).
*   **The Custom Launcher Script (`launch_napari.py`):** By default, dragging and dropping a folder of 2D images into Napari can cause it to load them as 3-channel RGB arrays. For massive micro-CT stacks, this inflates RAM usage by ~300% and crashes the computer. Our custom script safely reads the dataset strictly as a 1-channel (8-bit) grayscale 3D volume and hands it directly to Napari, bypassing the memory spike.

---

## 2. Setting Up the Environment

To keep your deep learning packages from interfering with your standard Python tools, we will isolate them in a dedicated virtual environment named `.venv_nn`.

### Step 2a: Create and Activate the Virtual Environment
Open your Command Prompt or PowerShell and navigate to your main project folder:
```powershell
cd D:\automated-microct-workflow\
```

Create the virtual environment:
```powershell
python -m venv .venv_nn
```

Activate the environment (you must do this every time you want to use Napari):
```powershell
# On Windows Command Prompt / PowerShell:
.venv_nn\Scripts\activate
```
*(You will know it worked when your command line prompt starts with `(.venv_nn)`).*

### Step 2b: Install Dependencies
With the environment active, install Napari, its GUI backend, and our image processing tools:
```powershell
pip install "napari[all]" opencv-python numpy
```

Next, install the interactive AI segmentation plugin. Because you are running the AI inference locally on your own machine (rather than connecting to a remote server), you **must** include the `[local]` flag so pip downloads the necessary PyTorch backend:
```powershell
pip install "napari-nninteractive[local]"
```
*(Note: PyTorch is a large package, so this step may take a few minutes to complete).*

Install opencv:
```powershell
pip install "napari[all]" opencv-python numpy
```
---

## 3. The Custom Launcher Script

Create a new file in your project folder named **`launch_napari.py`**. 

> ⚠️ **CRITICAL WARNING:** Do **not** name this file `napari.py`. If you do, Python will attempt to import your script instead of the actual Napari software, resulting in a crash.

Copy and paste the following code into `launch_napari.py`. Make sure to update the `input_dir` path to point to the specific scan you want to segment:

```python
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
```

---

## 4. How to Launch and Use

Whenever you are ready to segment a new dataset:

1. Update the `input_dir` path inside `launch_napari.py` to target your new image folder.
2. Open PowerShell, navigate to your project folder, and activate the environment:
   ```powershell
   .venv_nn\Scripts\activate
   ```
3. Run the launcher script:
   ```powershell
   python launch_napari.py
   ```
4. Once Napari opens, go to the top menu bar, click **Plugins**, and select **nnInteractive**.
5. Use the nnInteractive dock widget on the right side of the screen to select your local AI model weights and begin 3D-aware segmentation!