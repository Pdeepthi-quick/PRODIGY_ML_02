import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

file_path = r"C:\Users\LENOVO\.spyder-py3\prodigy projects\customer\Mall_Customers.csv"  # Replace with your file path
data = pd.read_csv(file_path)
print(data.head())


print(data.columns)


data = data.dropna()  # Remove missing values if any
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]  # Select relevant features


inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)


plt.scatter(features['Annual Income (k$)'], features['Spending Score (1-100)'], 
            c=data['Cluster'], cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroids')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()


data.to_csv("clustered_customers.csv", index=False)
print("Clustered data saved as clustered_customers.csv")


