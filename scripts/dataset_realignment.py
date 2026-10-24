# NOTE: this script is optimised for use with PC with Dual Xeon CPUs, 
# 384GB of system RAM, and an NVIDIA RTX A2000 GPU (12GB VRAM).
# 
# This script performs automated rigid 3D registration of microCT datasets.
# It replaces manual alignment in Bruker DataViewer by using SimpleITK to calculate 
# a mathematical transform based on Mattes Mutual Information. It automatically 
# traverses a batch directory structure, processing unaligned 'recon' folders and 
# outputting 'realigned' folders complete with updated, compliant Bruker metadata logs.
# OPTIMISED FOR PC2: Dual Xeon CPUs, 384GB RAM, RTX A2000.

import os
import glob
import re
import numpy as np
import SimpleITK as sitk
import datetime
import math
import argparse

# =============================================================================
# --- 1. DATA I/O (LOADING & SAVING) ---
# WHAT IT DOES: Handles reading and writing directories of 8-bit .bmp image sequences 
# into standard SimpleITK 3D tensors. 
# WHY IT IS NEEDED: Bruker scanners output individual image slices rather than 
# standard volumetric formats like NIfTI or DICOM. These functions allow the pipeline 
# to seamlessly ingest Bruker outputs and export them back in the exact same format 
# without corrupting pixel intensity scales.
# =============================================================================
def load_bmp_stack(folder_path):
    """Loads a numerical sequence of BMP files into a 3D SimpleITK image."""
    file_list = sorted(glob.glob(os.path.join(folder_path, "*.bmp")) + 
                       glob.glob(os.path.join(folder_path, "*.BMP")))
    if not file_list:
        raise FileNotFoundError(f"No BMP files found in: {folder_path}")
    
    print(f"Loading {len(file_list)} slices from {os.path.basename(folder_path)}...")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_list)
    return reader.Execute()

def save_bmp_stack(image, output_folder, prefix="aligned_"):
    """Exports a 3D SimpleITK image back to an 8-bit BMP sequence."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Ensure pixel values are clamped and cast back to unsigned 8-bit integers
    if image.GetPixelID() != sitk.sitkUInt8:
        image = sitk.Cast(sitk.Clamp(image, sitk.sitkUInt8, 0, 255), sitk.sitkUInt8)
        
    writer = sitk.ImageSeriesWriter()
    num_slices = image.GetSize()[2]
    output_names = [os.path.join(output_folder, f"{prefix}{i:04d}.bmp") for i in range(num_slices)]
    writer.SetFileNames(output_names)
    writer.Execute(image)
    print(f"Saved {num_slices} slices to: {output_folder}")

# =============================================================================
# --- 2. METADATA AUDIT TRAIL ---
# WHAT IT DOES: Modifies the original Bruker NRecon metadata file (.log). It updates 
# the slice count, filename prefixes, and explicitly appends a "CS image conversion" block 
# containing the exact 3D Euler angles used to rotate the skull.
# WHY IT IS NEEDED: CTAn relies entirely on this .log file for physical pixel size 
# (e.g., 6.97um) and calibration data to perform accurate morphometry. The appended 
# block mimics Bruker's proprietary DataViewer, ensuring 100% software compliance 
# and providing a transparent mathematical audit trail for publication review.
# =============================================================================
def update_bruker_log(original_log_path, output_log_path, new_slice_count, new_prefix, transform):
    if not os.path.exists(original_log_path):
        print(f"Warning: No .log file found at {original_log_path}. CTAn will lack metadata.")
        return

    with open(original_log_path, 'r', encoding='latin-1') as f:
        content = f.read()

    # 1. Update the overall Z-slice count in the Reconstruction block
    content = re.sub(
        r'(Sections Count\s*=\s*)\d+', 
        rf'\g<1>{new_slice_count}', 
        content
    )

    # 2. Extract rotation angles (in degrees) from the rigid transform for the audit log
    if isinstance(transform, sitk.Transform):
        # Cast to Euler3D to extract specific axis rotations
        euler_transform = sitk.Euler3DTransform(transform)
        rot_x = math.degrees(euler_transform.GetAngleX())
        rot_y = math.degrees(euler_transform.GetAngleY())
        rot_z = math.degrees(euler_transform.GetAngleZ())
    else:
        rot_x, rot_y, rot_z = 0.0, 0.0, 0.0

    current_time = datetime.datetime.now().strftime("%b %d, %Y  %H:%M:%S")
    original_log_name = os.path.basename(original_log_path)

    # 3. Build the DataViewer-compliant appended blocks
    dataviewer_blocks = f"""
[File name convention]
Filename Index Length=8
Filename Prefix={new_prefix}

[CS image conversion 1]
Converted by=SimpleITK Automated Pipeline (Replacing Dataviewer)
Conversion time={current_time}
Original configuration file={original_log_name}
Conversion description=...TRA. Rotated(deg): ({rot_x:.3f}, {rot_y:.3f}, {rot_z:.3f}).
Image scale modified=0
HU calibration modified=0
"""

    # 4. Append to the end of the file and save
    with open(output_log_path, 'w', encoding='latin-1') as f:
        f.write(content.strip() + "\n" + dataviewer_blocks)
        
    print(f"  -> Generated compliant Bruker metadata log: {os.path.basename(output_log_path)}")

# =============================================================================
# --- 3. MUTUAL INFORMATION OPTIMIZER ---
# WHAT IT DOES: Calculates a Rigid 3D transform (3 degrees rotation, 3 degrees translation)
# by maximizing the overlap of grayscale values between the new target and the fixed Atlas.
# WHY IT IS NEEDED: This replaces the manual, tedious "landmark hunting" required 
# by Bruker DataViewer. The MOMENTS initialization calculates the principal axis of 
# bone mass first, allowing the algorithm to seamlessly resolve large rotational 
# differences (like a skull mounted diagonally upside down).
# =============================================================================
def calculate_rigid_alignment(atlas_img, target_img, num_threads=None):
    """
    Computes the 3D rotation and translation matrix.
    Updated for robust alignment to handle large initial angles (e.g., ~50 degrees).
    """
    print("Calculating robust rigid 3D alignment (this may take a few minutes)...")
    
    # PC2 OPTIMIZATION: Explicitly set SimpleITK to utilize all available Xeon cores
    if num_threads:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(num_threads)
        print(f"Utilizing {num_threads} CPU threads for mathematical optimization.")
        
    atlas_f = sitk.Cast(atlas_img, sitk.sitkFloat32)
    target_f = sitk.Cast(target_img, sitk.sitkFloat32)
    
    # MOMENTS initialization correctly aligns principal axes of inertia for large initial rotations
    initial_transform = sitk.CenteredTransformInitializer(
        atlas_f, target_f, sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.08)
    registration.SetInterpolator(sitk.sitkLinear)
    
    # Multi-resolution pyramid helps the optimizer not get stuck in local minima
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.5, minStep=1e-4, numberOfIterations=300, relaxationFactor=0.75
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial_transform, inPlace=False)
    
    return registration.Execute(atlas_f, target_f)

# =============================================================================
# --- 4. VOXEL RESAMPLING ---
# WHAT IT DOES: Applies the calculated 3D rotation matrix to the unaligned target, 
# mathematically rotating it into the same coordinate space as the Atlas.
# WHY IT IS NEEDED: Mode 1 (Dynamic Z) prevents the algorithm from forcefully 
# cropping skulls that are physically longer than the reference Atlas. It projects 
# the new physical dimensions and dynamically adds empty Z-slices to preserve 
# 100% of the anatomical dataset.
# =============================================================================
def resample_dataset(target_img, atlas_img, transform, mode):
    """Resamples the dataset based on the selected mode."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0) # Pad empty space with black (air)
    resampler.SetOutputPixelType(target_img.GetPixelID())
    
    if mode == 2: 
        # Option 2: Atlas Bounding Box
        resampler.SetReferenceImage(atlas_img)
        return resampler.Execute(target_img)
        
    elif mode == 1: 
        # Option 1: Dynamic Z (Preserve length)
        inv_transform = transform.GetInverse()
        target_size = target_img.GetSize()
        
        # Calculate Z-axis physical bounds of the newly rotated target
        corners_idx = [
            [0, 0, 0], [target_size[0], 0, 0], 
            [0, target_size[1], 0], [target_size[0], target_size[1], 0],
            [0, 0, target_size[2]], [target_size[0], 0, target_size[2]], 
            [0, target_size[1], target_size[2]], [target_size[0], target_size[1], target_size[2]]
        ]
        
        z_indices = []
        for c in corners_idx:
            pt_target = target_img.TransformIndexToPhysicalPoint(c)
            pt_atlas = inv_transform.TransformPoint(pt_target)
            idx_atlas = atlas_img.TransformPhysicalPointToContinuousIndex(pt_atlas)
            z_indices.append(idx_atlas[2])
            
        min_z = int(np.floor(min(z_indices)))
        max_z = int(np.ceil(max(z_indices)))
        
        out_size = [atlas_img.GetSize()[0], atlas_img.GetSize()[1], max_z - min_z + 1]
        out_origin = atlas_img.TransformContinuousIndexToPhysicalPoint([0.0, 0.0, float(min_z)])
        
        resampler.SetSize(out_size)
        resampler.SetOutputOrigin(out_origin)
        resampler.SetOutputSpacing(atlas_img.GetSpacing())
        resampler.SetOutputDirection(atlas_img.GetDirection())
        
        return resampler.Execute(target_img)

# =============================================================================
# --- 5. MAIN BATCH LOOP & TERMINAL PARSER ---
# WHAT IT DOES: Scans the specified external drive for sample subfolders. It loads 
# the chosen reference Atlas into memory exactly once, then iteratively searches 
# each subfolder for a "recon" directory. It rigidly aligns the unaligned reconstruction, 
# outputs the new .bmp stack to an adjacent "realigned" directory, and injects 
# the updated Bruker metadata log.
# WHY IT IS NEEDED: Automates high-throughput processing and streamlines the workflow. 
# By intercepting console commands, it allows you to dynamically assign input/output 
# directories and select the biological target directly from the terminal prompt, 
# without ever having to manually open and edit the Python script.
# =============================================================================
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Automated 3D Rigid MicroCT Registration")
    
    # Require the biological target to be specified to automatically route to the correct Atlas
    parser.add_argument(
        "--target", 
        type=str, 
        required=True, 
        choices=["skull", "tibia_left", "tibia_right", "femur_left", "femur_right"],
        help="Select the biological target to route to the correct Atlas."
    )
    
    # Require the input directory path
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to the main external directory containing sample subfolders."
    )
    
    # Optional mode override
    parser.add_argument(
        "--mode", 
        type=int, 
        default=1, 
        choices=[1, 2], 
        help="1=Dynamic Z (preserves length), 2=Atlas BBox (strict crop)."
    )
    
    # PC2 OPTIMIZATION ARGUMENT
    parser.add_argument(
        "--threads", 
        type=int, 
        default=0, 
        help="Override default CPU threads used by SimpleITK. Leave at 0 to use all available."
    )
    
    args = parser.parse_args()

    # 1. Establish the default Atlas paths based on the chosen target
    base_atlas_dir = "F:/automated-microct-workflow/1_reference_files/atlast_library"
    
    if args.target == "skull":
        ATLAS_DIR = os.path.join(base_atlas_dir, "Skull_Adult")
    elif args.target == "tibia_left":
        ATLAS_DIR = os.path.join(base_atlas_dir, "Tibia_Left")
    elif args.target == "tibia_right":
        ATLAS_DIR = os.path.join(base_atlas_dir, "Tibia_Right")
    elif args.target == "femur_left":
        ATLAS_DIR = os.path.join(base_atlas_dir, "Femur_Left")
    elif args.target == "femur_right":
        ATLAS_DIR = os.path.join(base_atlas_dir, "Femur_Right")

    MAIN_DATA_DIR = args.input
    OUTPUT_MODE = args.mode

    print(f"--- INITIALIZING AUTOMATED REGISTRATION PIPELINE ---")
    print(f"Target: {args.target.upper()}")
    print(f"Input Directory: {MAIN_DATA_DIR}")
    print(f"Mode: {'Dynamic Z (Preserve Length)' if OUTPUT_MODE == 1 else 'Atlas Bounding Box (Strict Crop)'}")
    
    if not os.path.exists(ATLAS_DIR):
        print(f"\nCRITICAL ERROR: Atlas directory not found at {ATLAS_DIR}. Please ensure your library is set up.")
        exit()

    print(f"Loading Atlas into memory from {ATLAS_DIR}...")
    atlas = load_bmp_stack(ATLAS_DIR)
    
    skull_folders = [f.path for f in os.scandir(MAIN_DATA_DIR) if f.is_dir()]
    
    # PC2 OPTIMIZATION: Setup threads
    num_threads = None
    if args.threads > 0:
        num_threads = args.threads
    else:
        # Defaults to all physical cores available to the OS
        num_threads = os.cpu_count()
    
    for skull_dir in skull_folders:
        skull_name = os.path.basename(skull_dir)
        recon_dir = os.path.join(skull_dir, "recon")
        realigned_dir = os.path.join(skull_dir, "realigned")
        
        # Skip this folder if the NRecon 'recon' folder hasn't been created yet
        if not os.path.exists(recon_dir):
            print(f"\nSkipping [{skull_name}]: No 'recon' folder found.")
            continue
            
        # Skip if already realigned to prevent duplicating work on rerun
        if os.path.exists(realigned_dir) and len(os.listdir(realigned_dir)) > 0:
            print(f"\nSkipping [{skull_name}]: 'realigned' folder already populated.")
            continue

        print(f"\n========================================")
        print(f"Processing: {skull_name}")
        print(f"========================================")
        
        # Find the target Bruker .log file dynamically inside the recon directory
        target_log_files = glob.glob(os.path.join(recon_dir, "*.log")) + glob.glob(os.path.join(recon_dir, "*.LOG"))
        target_log = target_log_files[0] if target_log_files else ""
        
        try:
            target = load_bmp_stack(recon_dir)
            transform = calculate_rigid_alignment(atlas, target, num_threads=num_threads)
            realigned_img = resample_dataset(target, atlas, transform, mode=OUTPUT_MODE)
            
            # Save to the new 'realigned' folder using the specific skull name as a prefix
            prefix = f"{skull_name}_"
            save_bmp_stack(realigned_img, realigned_dir, prefix=prefix)
            
            # Copy and update the Bruker log into the realigned folder
            if target_log:
                new_log_name = f"{skull_name}_realigned.log"
                update_bruker_log(
                    original_log_path=target_log, 
                    output_log_path=os.path.join(realigned_dir, new_log_name), 
                    new_slice_count=realigned_img.GetSize()[2], 
                    new_prefix=prefix,
                    transform=transform
                )
            print(f"Successfully finished processing {skull_name}.")
            
        except Exception as e:
            print(f"ERROR processing {skull_name}: {e}")

    print("\nBatch alignment pipeline complete.")

# =============================================================================
# --- HOW TO RUN DIFFERENT REALIGNMENT BATCHES ---
# Open your terminal (CMD or PowerShell) and navigate to the scripts directory.
# 
# The script requires two mandatory arguments: --target and --input.
# 
# 1 - Realigning Mouse Skulls on Drive E (using Default Dynamic Z mode):
# python batch_dataset_realignment.py --target skull --input "E:/External_Drive/Mouse_Skulls_Project"
#
# 2 - Realigning Left Tibias on Drive D:
# python batch_dataset_realignment.py --target tibia_left --input "D:/External_Drive/Tibia_Project"
#
# 3 - Realigning Right Femurs and forcing strict cropping (Mode 2):
# python batch_dataset_realignment.py --target femur_right --input "D:/External_Drive/Femur_Project" --mode 2
#
# --- PC2 OPTIMIZATION ---
# By default, SimpleITK will try to use all available CPU cores. If you want to limit 
# the number of threads (e.g., leaving cores free for other background tasks on PC2), 
# you can use the --threads flag:
# python batch_dataset_realignment.py --target skull --input "E:/Mouse_Skulls" --threads 16
#
# --- WHAT THE MODES DO ---
# --mode 1 (Dynamic Z - Default & Recommended): 
# Mathematically rotates the target into the Atlas coordinate space, but dynamically 
# calculates the new physical bounding box to ensure 0% of the target bone is cropped. 
# Resulting folders will have different .bmp file counts depending on physical bone length.
#
# --mode 2 (Atlas BBox): 
# Strict conformity. Forces the realigned target to adopt the exact bounding box of 
# the Atlas. Any part of the target bone extending past the Atlas boundaries is 
# permanently cropped. Useful only if your downstream pipeline strictly requires 
# identically sized tensors across all samples.
# =============================================================================
