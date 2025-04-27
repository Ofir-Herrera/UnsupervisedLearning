# Unsupervised Learning: Bacteria Segmentation

This project addresses the task of segmenting *E. coli* bacteria in dense microscopy images without using any labeled data.  
The goal is to extract meaningful features from the images and apply **KMeans clustering** to create segmentation masks, which can later be used for statistical analysis or as pseudo-labels for supervised learning.

---

## 📂 Project Structure

. ├── config.py # Set hyperparameters (image name, kernels, number of clusters, etc.) ├── main.py # Main script to run the pipeline ├── image_loader.py # Load microscopy images ├── feature_extractor.py # Feature extraction (Gaussian blur difference, Laplacian, edges) ├── clusterer.py # KMeans clustering ├── visualizer.py # Visualization utilities (save plots, masks, histograms, etc.) ├── statistical_analyzer.py # Silhouette score analysis ├── utils.py # Helper functions (normalization, etc.) ├── images/ # Input images folder ├── plots/ # Output folder for generated plots and results └── README.md # This file

yaml
Copy
Edit

---

## ⚙️ How to Run
Set the hyperparameters:
Open config.py and modify the following parameters:

IMAGE: Name of the image to analyze.

KERNEL_SIZE_1 and KERNEL_SIZE_2: Kernel sizes for Gaussian blur.

N_CLUSTERS: Number of clusters for KMeans.

RANDOM_STATE: Random seed for reproducibility.

IMAGE_INDEX: Which image to load from the list.

Run main.py:
After execution, all the results (plots, segmented masks, silhouette analysis, etc.) will be saved automatically inside the plots/ folder.

A detailed project report, including background, methodology, results, and discussion, is available here:
The full project report is available [here](./Unsupervised_Learning.pdf).



Notes:
Input images must be placed inside the images/ directory.

Silhouette scores are used to determine the optimal number of clusters.

Feature extraction includes:

Gaussian blur difference

Laplacian filtering

Canny edge detection

All feature vectors are normalized to [0, 1] before clustering.

