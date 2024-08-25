import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load your dataset
df = pd.read_csv('./train.csv')  # Replace with your actual file path
# Select the features for clustering
features = [
    'avg_rating_300', 'avg_rating_1000', 'avg_rating_3000',
    'interaction_count_300', 'interaction_count_1000', 'interaction_count_3000',
    'distance', 'same_type'
]

# Fill missing values with the mean of the column
df.fillna(0, inplace=True)

X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Determine the optimal number of clusters using the Elbow method
inertia = []  # Sum of squared distances to closest cluster center (cost function J)
# Testing different numbers of clusters
K_range = range(1, 15)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)  # Append the cost function value (inertia)
# Plot the Elbow method results
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, 'bo-', markersize=8)
plt.xlabel('Number of clusters (k)')
plt.ylabel('K-means Cost Function (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.show()

# # Based on the elbow plot, choose the optimal number of clusters (let's assume it's 4)
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
# Save the labeled dataset with clusters
df.to_csv('./labeled_train.csv', index=False)
print(f"K-means clustering complete with {optimal_k} clusters. Labeled dataset saved as 'labeled_dataset.csv'.")
