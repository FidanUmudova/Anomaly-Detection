# Anomaly Detection System using Gaussian Distribution

This repository contains core implementations for an **Anomaly Detection** algorithm built using Python and NumPy, developed as part of machine learning coursework.

##  Implemented Functions

1. **`estimate_gaussian` (Parameter Estimation):**
   - Calculates the mean ($\mu$) and variance ($\sigma^2$) for each feature in the dataset across all training examples.
   - Utilizes vectorized NumPy operations for efficient computation along dataset columns.

2. **`select_threshold` (Threshold Optimization):**
   - Evaluates multiple threshold values ($\epsilon$) using a validation set to find the optimal cutoff for identifying outliers.
   - Computes **True Positives (tp)**, **False Positives (fp)**, **False Negatives (fn)**, precision, recall, and selects the threshold that maximizes the **$F_1$ score**.


