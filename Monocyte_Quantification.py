import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

# Import all helper functions
from helper_functions import (
    extract_image_and_seg,
    extract_features,
    get_area_in_micrometers,
    calculate_tissue_fraction,
    highlight_cells
)

def main():
    image_folder_path = "./images"
    model_path = "./models/monocyte_classifier.joblib"
    results_path = "./results.csv"

    # Find and pair TIFF files with their corresponding segmentation files.
    image_paths = []
    seg_paths = []
    base_names = []

    for file in os.listdir(image_folder_path):
        if file.endswith(".tif") or file.endswith(".tiff"):
            base_name = os.path.splitext(file)[0]
            image_path = os.path.join(image_folder_path, file)
            seg_path = os.path.join(image_folder_path, f"{base_name}_seg.npy")

            if os.path.exists(seg_path):
                image_paths.append(image_path)
                seg_paths.append(seg_path)
                base_names.append(base_name)

    print(f"Found {len(image_paths)} file pairs.")

    # Load the classifier (LogisticRegression)
    clf = joblib.load(model_path)

    results = []

    for image_path, seg_path, base_name in tqdm(
        zip(image_paths, seg_paths, base_names), total=len(image_paths)
    ):
        # Extract features
        image, labeled_seg = extract_image_and_seg(image_path, seg_path)
        features = extract_features(image, labeled_seg)

        # Fill NaN for model
        cell_indices = features.iloc[:, 0]
        features_filled = features.iloc[:, 1:].fillna(0)

        # Predict monocytes
        y_pred = clf.predict(features_filled)
        predicted_indices = cell_indices[y_pred == 1].tolist()

        # Save prediction visualization
        pred_image_path = os.path.join(image_folder_path, f"{base_name}_pred.png")
        highlight_cells(image, labeled_seg, predicted_indices, pred_image_path)

        # Calculate metrics
        total_cells = len(features)
        total_predicted_cells = len(predicted_indices)
        tissue_fraction = calculate_tissue_fraction(image)
        area = get_area_in_micrometers(image_path)

        if tissue_fraction > 0:
            monocytes_per_um2 = total_predicted_cells / (area * tissue_fraction)
        else:
            monocytes_per_um2 = 0

        # Append to results
        results.append({
            "Name": base_name,
            "#Cells": total_cells,
            "#Monocytes": total_predicted_cells,
            "Ratio": total_predicted_cells / total_cells if total_cells > 0 else 0,
            "Monocytes/µm2": monocytes_per_um2
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)

    print("Processing complete.")

if __name__ == "__main__":
    main()
