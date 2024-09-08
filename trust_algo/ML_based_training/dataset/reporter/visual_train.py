# Import necessary libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
train_df = pd.read_csv('./labeled_train.csv')
test_df = pd.read_csv('./labeled_test.csv')

# Define features and labels
features = [
    'avg_rating_300', 'avg_rating_1000', 'avg_rating_3000',
    'interaction_count_300', 'interaction_count_1000', 'interaction_count_3000',
    'distance', 'same_type'
]

# Separate features and labels for the training data
X_train = train_df[features]
y_train = train_df['cluster']

# Since t-SNE was slow, we'll take a subset of the data for faster processing
X_train_subset = X_train.sample(n=500, random_state=42)
y_train_subset = y_train[X_train_subset.index]

# Define the SVM classifier with RBF kernel
svm_rbf = SVC(kernel='rbf', gamma='scale')

# Apply Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm_rbf, X_train_subset, y_train_subset, cv=skf)

# Print cross-validation performance
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}")

# Train the SVM on the full training set
svm_rbf.fit(X_train_subset, y_train_subset)

# Perform PCA for visualization (2D)
pca = PCA(n_components=2)
X_train_2d_pca = pca.fit_transform(X_train_subset)

# Create a mesh grid for plotting decision boundaries
h = .02  # step size in the mesh
x_min, x_max = X_train_2d_pca[:, 0].min() - 1, X_train_2d_pca[:, 0].max() + 1
y_min, y_max = X_train_2d_pca[:, 1].min() - 1, X_train_2d_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Project the mesh grid using PCA components
Z = svm_rbf.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Visualizing only a subset of the PCA-transformed points
subset_size = 100  # Visualize only 100 points
subset_indices = np.random.choice(len(X_train_2d_pca), subset_size, replace=False)

# Plotting the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

# Plotting a subset of the points
plt.scatter(X_train_2d_pca[subset_indices][y_train_subset.iloc[subset_indices] == 0, 0],
            X_train_2d_pca[subset_indices][y_train_subset.iloc[subset_indices] == 0, 1],
            c='red', label='Label 0', alpha=0.5)
plt.scatter(X_train_2d_pca[subset_indices][y_train_subset.iloc[subset_indices] == 1, 0],
            X_train_2d_pca[subset_indices][y_train_subset.iloc[subset_indices] == 1, 1],
            c='blue', label='Label 1', alpha=0.5)
plt.title("SVM Classification Boundary with PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
