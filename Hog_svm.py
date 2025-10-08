import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA

# --------------------------
# 1. DATA LOADING & FEATURE EXTRACTION
# --------------------------

dataset_path = "C:\\Users\\jithi\\OneDrive\\Desktop\\Finger_print\\300_fingeprint\\DATASET"
IMG_SIZE = (128, 128)  # Resize all images to a fixed size

features = []
labels = []

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder '{dataset_path}' not found! Ensure the dataset path is correct.")

# Iterate through blood group folders
for blood_group in os.listdir(dataset_path):
    blood_group_path = os.path.join(dataset_path, blood_group)

    if not os.path.isdir(blood_group_path):  # Ensure it's a folder
        continue

    # Process each image inside the blood group folder
    for filename in os.listdir(blood_group_path):
        file_path = os.path.join(blood_group_path, filename)

        # Ensure it's an image file (including BMP format)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        image = cv2.imread(file_path)

        if image is None:
            continue  # Skip if the image cannot be loaded

        image = cv2.resize(image, IMG_SIZE)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract HOG features
        hog_features, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys',
                              visualize=True, transform_sqrt=True)
        features.append(hog_features)
        labels.append(blood_group)  # Use folder name as label

# Convert lists to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Check if features were extracted
if features.size == 0:
    raise ValueError("No features were extracted. Ensure your dataset contains valid images.")

print("Total samples:", features.shape[0])
print("Feature vector size:", features.shape[1] if features.size > 0 else "N/A")




# --------------------------
# 2. SPLIT THE DATA: TRAINING AND TESTING
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

# --------------------------
# 3. TRAIN THE SVM CLASSIFIER
# --------------------------

svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
svm_classifier.fit(X_train, y_train)

# --------------------------
# 4. EVALUATE ON THE TEST SET
# --------------------------

y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------
# 5. CONFUSION MATRIX VISUALIZATION
# --------------------------

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# --------------------------
# 6. CLASS-WISE ACCURACY
# --------------------------

class_accuracies = {}
for label in np.unique(y_test):
    class_indices = (y_test == label)
    class_accuracy = accuracy_score(y_test[class_indices], y_pred[class_indices])
    class_accuracies[label] = class_accuracy

print("Class-wise Accuracy:")
for cls, acc in class_accuracies.items():
    print(f"{cls}: {acc:.2f}")

# --------------------------
# 7. ROC CURVE & AUC SCORE
# --------------------------

y_test_bin = label_binarize(y_test, classes=np.unique(labels))
classifier = OneVsRestClassifier(SVC(kernel="linear", probability=True, random_state=42))
classifier.fit(X_train, label_binarize(y_train, classes=np.unique(labels)))

y_score = classifier.decision_function(X_test)
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {np.unique(labels)[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# --------------------------
# 8. PRECISION-RECALL CURVE
# --------------------------

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, label=f'Class {np.unique(labels)[i]}')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# --------------------------
# 9. FEATURE VISUALIZATION WITH PCA
# --------------------------

pca = PCA(n_components=2)  # Reduce features to 2D for visualization
X_pca = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
for label in np.unique(labels):
    plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], label=label, alpha=0.7)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Blood Group Features")
plt.legend()
plt.show()
