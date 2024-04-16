# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import all necessary packages.

2.Upload the appropiate dataset to perform K-Means Clustering.

3.Perform K-Means Clustering on the requried dataset.

4.Plot graph and display the clusters.

## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Sudharsanam R K
RegisterNumber: 212222040163
```

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv("/content/Mall_Customers_EX8.csv")
data

# Plot the data
X = data[['Annual Income (k$)' , 'Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel("Spending Score (1-100)")
plt.show()

# Set the number of clusters
k = 5

# Perform K-means clustering
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# Get centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroids: ")
print(centroids)
print("Label:")

# define colors for each cluster
colors = ['r', 'g', 'b', 'c', 'm']

# plotting the clusters
for i in range(k):
  cluster_points = X[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'], color=colors[i], label=f'Cluster {i+1}')

  #Find minimum enclosing circle
  distances = euclidean_distances(cluster_points, [centroids[i]])
  radius = np.max(distances)

  circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, color='k', label='Centroids')

# Set plot title and labels
plt.title('K-means Clustering')
plt.xlabel("Annual Income (k$)")
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal') # Ensure aspect ratio is equal
plt.show()
```

## Output:
## DataSet:
![image](https://github.com/SudharsanamRK/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/115523484/4c3ece4f-d137-4d0e-978b-7aae70bf3f98)

## Centroid Values:
![image](https://github.com/SudharsanamRK/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/115523484/55443d85-6367-4e75-a9f2-dfbf9c146284)

## K-Means Cluster:
![image](https://github.com/SudharsanamRK/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/115523484/d81d6faa-79fb-47a5-9464-56070cdbcf96)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
