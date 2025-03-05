# Monocyte Quantification Pipeline

This repository contains the monocyte quantification pipeline accompanying the paper:\
**Examining Radiation-Induced Upregulation of Amphiregulin**

## Introduction

This pipeline is designed to **quantify the density of monocytes** in immunofluorescent (IF) images by:

1) Applying a **cell segmentation** approach (Cellpose) to detect individual cells using three channels:
- **DAPI** (Nucleus): Primarily used for nucleus detection.
- **pERGB** (Cytoplasm): Used as a cytoplasm marker (though a different cytoplasm stain can be substituted if desired).
- **Ly6C**: Used for monocyte detection.

2) Running a **logistic regression monocyte classifier** on extracted features (e.g., intensity signatures in the Ly6C channel) to identify monocytes.

3) Estimating the **tissue fraction** automatically from the cytoplasm channel, which helps correct the quantification in partially empty or artifact-heavy images.

The result is a **CSV file** listing each imges monocyte count, ratio and density, as well as a **PNG overlay** showing predicted monocytes.

The pipeline expects a set of **TIFF images** in one folder. You first run **Cellpose** on these TIFF images to generate segmentation masks, and then run this pipeline to automatically classify monocytes and produce numeric and visual results.
## Installation
**1) Clone This Repository**

```
git clone git@github.com:apiffko/IF.mono.quantif.git
```

**2) Move Into the Project Directory**

```
cd <project path>
```

**3) Create a (Recommended) Python Environment**

Python 3.7 or higher is required. Two common approaches are Conda or venv:

**Conda Example**

```
conda create --name MonoQuantif python=3.9
conda activate MonoQuantif

```

**Python venv Example**

```
python3 -m venv MonoQuantif
source MonoQuantif/bin/activate
pip install --upgrade pip
```

**4) Install Dependencies**

```
pip install -r requirements.txt
```

Note: This project requires [Cellpose](https://github.com/MouseLand/cellpose) to be installed and accessible. Follow instructions on the Cellpose repository for installation.


## Usage

**Directory Layout & Prerequisites**

1. **Images**: Store the TIFF files (multi-channel IF images) in a folder, e.g., ``"<project path>/images"`` (or adjust the image folder path in the ``Monocyte_Quantification.py`` file)
2. **Cellpose Model**: Ensure you have the custom-trained Cellpose model located in ``"<project path>/models/cellsegmentation"`` (or provide the correct path to your own model).
3. **Monocyte Classifier**: The logistic regression model (e.g., ``monocyte_classifier.joblib``) should be in ``"<project path>/models"``.

Now everything is ready to run the pipeline:

**1. Run Cellpose**
Use Cellpose to generate segmentation masks from your images:

```
cellpose \
  --dir "<project path>/images" \
  --pretrained_model "<project path>/models/cellsegmentation" \
  --chan 2 \
  --chan2 3 \
  --diameter 38.37 \
  --verbose
```
- ``--chan 2`` = Cytoplasm channel (used to detect the cell body)
- ``--chan2 3`` = DAPI channel (used to detect nuclei)

Adjust these channel numbers (and diameter) if your configuration differs. This step creates ``_seg.npy`` files in the same folder as each TIFF image.

**2. Run the Monocyte Quantification**

Navigate to the project folder and execute:
```
cd <project path>
python Monocyte_Quantification.py
```

This script will:

1. Load each TIFF and corresponding _seg.npy file.
2. Extract features from the Ly6C channel (and optional cytoplasm-based tissue mask).
3. Use the logistic regression classifier to predict which cells are monocytes.
4. Save:
  - ``results.csv``: Contains counts of total cells, predicted monocytes, ratio of monocytes, and monocytes per square micron.
  - ``_pred.png files``: Visual overlays marking the predicted monocytes.

## Notes on Usage & Hardware

- Running Cellpose segmentation can be memory-intensive for large datasets. Using a machine with sufficient RAM and GPU acceleration is highly recommended if you have a large batch of high-resolution images.
- If you need to train or retrain the Cellpose model for a different cytoplasm stain or to optimize performance on different data, consult the [Cellpose documentation](https://github.com/MouseLand/cellpose).
- The logistic regression model is custom-trained for Ly6C-based monocyte detection. If you are using a different marker, you may need to retrain your own classifier.

## Contact & Support

If you have questions or encounter issues with this pipeline, please open an Issue in this repository or contact the corresponding authors listed in the paper.

Thank you for using this tool! We hope it helps streamline your monocyte quantification tasks.
