import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

# --------------------------
# 1. DATA LOADING & FEATURE EXTRACTION
# --------------------------

dataset_path = "C:\\Users\\jithi\\OneDrive\\Desktop\\Finger_print\\300_fingeprint\\DATASET"
IMG_SIZE = (128, 128)

features = []
labels = []

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder '{dataset_path}' not found! Ensure the dataset path is correct.")

for blood_group in os.listdir(dataset_path):
    blood_group_path = os.path.join(dataset_path, blood_group)
    if not os.path.isdir(blood_group_path):
        continue
    
    for filename in os.listdir(blood_group_path):
        file_path = os.path.join(blood_group_path, filename)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        
        image = cv2.imread(file_path)
        if image is None:
            continue
        
        image = cv2.resize(image, IMG_SIZE)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        hog_features, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys',
                              visualize=True, transform_sqrt=True)
        features.append(hog_features)
        labels.append(blood_group)

features = np.array(features)
labels = np.array(labels)

if features.size == 0:
    raise ValueError("No features were extracted. Ensure your dataset contains valid images.")

print("Total samples:", features.shape[0])
print("Feature vector size:", features.shape[1] if features.size > 0 else "N/A")

# --------------------------
# 2. SPLIT THE DATA: TRAINING AND TESTING
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

# --------------------------
# 3. TRAIN THE KNN CLASSIFIER
# --------------------------
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# --------------------------
# 4. EVALUATE ON THE TEST SET
# --------------------------
y_pred = knn_classifier.predict(X_test)
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
n_classes = y_test_bin.shape[1]

probs = knn_classifier.predict_proba(X_test)

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {np.unique(labels)[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# --------------------------
# 8. FEATURE VISUALIZATION WITH PCA
# --------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
for label in np.unique(labels):
    plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], label=label, alpha=0.7)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Blood Group Features")
plt.legend()
plt.show()