import numpy as np
import pandas as pd
import tifffile as tiff

from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing
import matplotlib.pyplot as plt

def extract_image_and_seg(image_path, seg_path):
    """
    Load the image (TIFF) and segmentation (NumPy) files.

    Returns:
        image (numpy.ndarray): The image in shape (H, W, C).
        labeled_seg (numpy.ndarray): Labeled segmentation mask.
    """
    image = tiff.imread(image_path)
    if image.shape[0] == 3:  # Check if channels are along the first axis
        image = image.transpose(1, 2, 0)  # Convert to (H, W, C)

    seg = np.load(seg_path, allow_pickle=True).item()
    seg_mask = seg['masks']
    labeled_seg = label(seg_mask)
    return image, labeled_seg

def extract_features(image, labeled_seg, dilation=5):
    """
    Extract features from each cell in the image and segmentation.
    
    This function specifically focuses on the Ly6C channel (originally red channel).
    We apply a log transform, then apply constant clipping thresholds, 
    and finally normalize to [0, 1]. We calculate ring-based intensity 
    features for each cell.

    Args:
        image (numpy.ndarray): The image in shape (H, W, C).
        labeled_seg (numpy.ndarray): Labeled segmentation mask.
        k (int, optional): Number of iterations for erosion/dilation.

    Returns:
        pandas.DataFrame: A dataframe of features for each cell.
    """
    # --- Prepare Ly6C channel ---
    ly6c_channel = image[:, :, 0].astype(float)
    ly6c_channel = np.log1p(ly6c_channel)  # Apply log transform

    # Constant clipping thresholds (previously described as quantile):
    low_thresh = 4.83
    high_thresh = 6.22
    ly6c_channel = np.clip(ly6c_channel, low_thresh, high_thresh)

    # Normalize to [0, 1]
    ly6c_channel = (ly6c_channel - ly6c_channel.min()) / (ly6c_channel.max() - ly6c_channel.min())

    # Compute reference intensities for the entire mask
    # (median or mean over all positive segmentation pixels)
    ly6c_median = np.median(ly6c_channel[labeled_seg > 0])
    ly6c_mean = np.mean(ly6c_channel[labeled_seg > 0])

    features = []

    for region in regionprops(labeled_seg, intensity_image=None):
        cell_id = region.label
        # Skip background
        if cell_id == 0:
            continue

        # Create a mask for the current cell
        cell_mask = (labeled_seg == cell_id)

        # Cell shape features
        area = region.area

        # Eroded & dilated masks
        eroded_mask = binary_erosion(cell_mask, iterations=dilation)
        dilated_mask = binary_dilation(cell_mask, iterations=dilation)

        inner_ring_mask = cell_mask & ~eroded_mask
        outer_ring_mask = dilated_mask & ~cell_mask
        inner_outer_ring_mask = inner_ring_mask | outer_ring_mask

        inside_without_inner_ring_mask = cell_mask & ~inner_ring_mask

        def compute_intensity_features(mask):
            if not np.any(mask):  
                # If empty, return zeros
                return [0, 0, 0, 0, 0, 0]
            intensities = ly6c_channel[mask]
            return [
                np.mean(intensities),
                np.max(intensities),
                np.min(intensities),
                np.std(intensities),
                np.sum(intensities > ly6c_median) / np.sum(mask),
                np.sum(intensities > ly6c_mean) / np.sum(mask),
            ]

        # Features of ring region, inside region, etc.
        features_inner_outer_ring = compute_intensity_features(inner_outer_ring_mask)
        features_inside_without_inner_ring = compute_intensity_features(inside_without_inner_ring_mask)

        # Combine
        features.append([
            cell_id, 
            area
        ] + features_inner_outer_ring + features_inside_without_inner_ring)

    return pd.DataFrame(features)

def get_area_in_micrometers(tiff_file_path):
    """
    Calculate the area of the image in square micrometers using metadata.
    """
    with tiff.TiffFile(tiff_file_path) as tif:
        tags = tif.pages[0].tags
        x_resolution = tags['XResolution'].value
        y_resolution = tags['YResolution'].value
        image_width = tif.pages[0].imagewidth
        image_length = tif.pages[0].imagelength

        x_res = x_resolution[0] / x_resolution[1]  # Pixels per micrometer
        y_res = y_resolution[0] / y_resolution[1]

        width_micrometers = image_width / x_res
        height_micrometers = image_length / y_res

        return width_micrometers * height_micrometers

def calculate_tissue_fraction(image):
    """
    Estimate the tissue fraction from the cytoplasm channel (originally green channel).
    We apply a log transform, then quantile-based clipping, then threshold.
    """
    cytoplasm_channel = image[:, :, 1].astype(float)
    cytoplasm_channel = np.log1p(cytoplasm_channel)
    cytoplasm_channel = np.clip(
        cytoplasm_channel, 
        np.quantile(cytoplasm_channel, 0.03), 
        np.quantile(cytoplasm_channel, 0.999)
    )
    cytoplasm_channel = (cytoplasm_channel - cytoplasm_channel.min()) / (
        cytoplasm_channel.max() - cytoplasm_channel.min()
    )
    return np.sum(binary_closing(cytoplasm_channel > 0.2, structure=np.ones((5, 5)))) / cytoplasm_channel.size

def highlight_cells(image, labeled_seg, cell_indices, output_file):
    """
    Highlight predicted cells in the Ly6C channel and save the visualization.

    The boundaries of all cells are shown in thin gray lines, and predicted cells
    are highlighted in thick white lines for clarity.
    """
    all_cells = find_boundaries(labeled_seg, mode='outer')
    monocytes = np.zeros_like(labeled_seg, dtype=bool)

    for monocyte_id in cell_indices:
        monocyte_mask = (labeled_seg == monocyte_id)
        monocyte_boundary = find_boundaries(monocyte_mask, mode='outer')
        monocyte_boundary_dilated = binary_dilation(monocyte_boundary, structure=np.ones((2, 2)))
        monocytes = np.logical_or(monocytes, monocyte_boundary_dilated)

    # Focus on Ly6C channel for visualization
    image_copy = image.copy()
    image_copy[:, :, 1] = 0  # Zero out other channels
    image_copy[:, :, 2] = 0

    # Log transform + clip as this usually gives a better visualization result
    ly6c_channel = image_copy[:, :, 0].astype(float)
    ly6c_channel = np.log1p(ly6c_channel)
    ly6c_channel = np.clip(
        ly6c_channel, 
        np.quantile(ly6c_channel, 0.03), 
        np.quantile(ly6c_channel, 0.999)
    ) 

    # Normalize to [0, 255]
    ly6c_channel = (ly6c_channel - ly6c_channel.min()) / (ly6c_channel.max() - ly6c_channel.min())
    ly6c_channel = (ly6c_channel * 255).astype(np.uint8)
    image_copy[:, :, 0] = ly6c_channel

    # Add boundary highlights
    image_copy[all_cells] = [100, 100, 100]
    image_copy[monocytes] = [200, 200, 200]

    plt.figure(figsize=(10, 10))
    plt.imshow(image_copy)
    plt.axis("off")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()