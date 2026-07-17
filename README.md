# Where's Waldo: Small Object Detection Pipeline


This repository contains a solution for a custom small-object detection datathon based on "Where's Waldo" illustrations. 

The goal is to detect five characters (Waldo, Odlaw, Wilma, Wizard, and woof) in large, detailed drawings under severe resource constraints.

## The Challenge

Standard object detection models (such as stock YOLOv8) fail on this dataset, yielding near-zero scores. The reasons are structural:

### 1. Resolution Loss from Resizing
Illustrated images are large (typically 3000 x 2000 pixels), while target characters are tiny (often 60 x 70 pixels). Standard detectors resize inputs to a fixed size like 640 x 640 before processing. This squeezes the characters down to roughly 13 x 15 pixels, destroying the detail needed to distinguish them from background clutter.

### 2. Coarse Feature Maps
A default YOLO architecture downsamples the input through successive layers, making predictions at strides of 8, 16, and 32 pixels. An object that is only 13 pixels wide is smaller than a single cell on the coarser feature maps, making it impossible for the network to represent or locate.

### 3. Training Data Scarcity
Deep learning models typically require thousands of images. This competition provides only 58 training images in total. Standard cross-validation splits further reduce the active training set, starving the model of crucial details.

### 4. Extreme Class Imbalance
The dataset contains a severe class imbalance. While Waldo appears in almost every scene, the rarest class (woof) has only a handful of examples across the entire dataset.

## The Winning Approach

To solve these challenges, our pipeline modifies both the data ingestion and the model architecture.

### Resolution & Feature Representation
* **SAHI Tiling (Training-Time):** Instead of resizing the images, we slide a 640 x 640 window across the high-resolution images with a 25% overlap. This keeps target characters at their native pixel sizes inside each tile. We clip bounding boxes at tile boundaries, keeping them only if a minimum visibility fraction (35%) remains.
* **Custom P2 Head:** We modified the YOLOv8 neck and neck connections to add a stride-4 (P2) detection head. This allows the model to compute features and predict bounding boxes at a much higher spatial resolution.

### Managing Scarcity and Imbalance
* **Seed Ensembling:** Standard k-fold validation splits would consume 20% of our scarce 58-image dataset for validation. Instead, we train all model seeds on the full tiled training dataset, introducing ensemble diversity purely through different initialization seeds and random augmentations.
* **Rare-Class Oversampling:** Training tiles containing the rarest classes (Odlaw, Wizard, and woof) are duplicated six-fold prior to training, forcing the network to see these classes more frequently per epoch.

### Post-Processing & Validation
* **Ensembled Inference (SAHI + TTA + WBF):** During inference, we run predictions on test images using sliding window tiles at multiple scales (512, 640, and 896 pixel windows) along with horizontal flips (Test-Time Augmentation).
* **Weighted Boxes Fusion (WBF):** Instead of standard Non-Maximum Suppression (NMS) which discards overlapping boxes, we use WBF to average the coordinates of agreeing predictions weighted by their confidence scores, yielding more accurate boundary boxes.
* **Per-Class Threshold Tuning:** The competition uses a custom recall-weighted F-beta metric (beta=1.5) with localization bonuses and false-positive penalties. We re-implemented this metric locally and ran a grid search on our validation split to find optimal per-class confidence thresholds.

## Repository Structure

* **WheresWaldo.ipynb:** The complete end-to-end notebook containing the pipeline, including tiling, custom YOLOv8 configuration, multi-seed training, validation metric evaluation, WBF inference, threshold tuning, and submission generation.
