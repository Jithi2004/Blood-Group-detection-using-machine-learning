# Blood-Group-detection-using-machine-learning
🩸 Blood Group Prediction Using Fingerprint Analysis (HOG + SVM / RF)
A novel non-invasive machine learning–based system for predicting human blood groups using fingerprint analysis.
This project eliminates the need for traditional blood sampling and laboratory testing by leveraging the genetic relationship between fingerprint patterns and blood group traits.
🔍 Overview

Goal: Detect blood group type from fingerprint images using pattern recognition and classification algorithms.

Approach:

Extract fingerprint features using Histogram of Oriented Gradients (HOG).

Classify extracted features using Support Vector Machine (SVM) and Random Forest (RF) classifiers.

Evaluate performance using metrics like Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

⚙️ Methodology

Data Acquisition: Collect fingerprint images and corresponding blood group information.

Feature Extraction: Apply HOG for fingerprint pattern analysis.

Classification: Train and test models using SVM and Random Forest.

Evaluation: Compare models based on performance metrics.

📊 Results
Model	Accuracy
HOG + SVM	90.69%
HOG + Random Forest	87.36%

The HOG-SVM model demonstrated superior classification accuracy, confirming the potential of fingerprints as a non-invasive biomarker for blood group identification.

🧠 Key Highlights

Eliminates need for blood samples or reagents.

Reduces time, cost, and invasiveness of traditional testing.

Fully automated ML-based pipeline for biometric–genetic correlation.

🧰 Tech Stack

Python, scikit-learn, OpenCV, NumPy, Matplotlib
