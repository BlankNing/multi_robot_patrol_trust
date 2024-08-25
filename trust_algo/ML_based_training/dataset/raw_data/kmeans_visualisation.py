import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the derived features dataset
df = pd.read_csv('./derived_features_as_provider_300.csv')

# Select the features for clustering
features = [
    'avg_rating_300', 'avg_rating_1000', 'avg_rating_3000',
    'interaction_count_300', 'interaction_count_1000', 'interaction_count_3000',
    'distance', 'same_type'
]
# Fill missing values with the mean of the column
df.fillna(df.mean(), inplace=True)

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Apply K-Means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Visualize the clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
df['pca_one'] = pca_features[:, 0]
df['pca_two'] = pca_features[:, 1]
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca_one', y='pca_two', hue='cluster', data=df, palette='viridis')
plt.title('K-Means Clustering of Trustworthiness')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Save the clustered data to a new CSV file
df.to_csv('./clustered_features.csv', index=False)
print("Clustering complete and visualized. Data saved to 'clustered_features.csv'.")