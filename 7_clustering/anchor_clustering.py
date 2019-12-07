import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as sk_clustering


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

object_sizes = pd.read_csv("data/object_sizes.csv")
# plt.scatter(x=object_sizes["width"], y=object_sizes["height"])
X = object_sizes[["width", "height"]]

print("K-means:")

k_means_clustering_model = sk_clustering.KMeans(n_clusters=5, init="random", n_init=1)

k_means_clustering_model.fit(X)

k_means_classes = k_means_clustering_model.predict(X)

ax1.set_title("K-means")
ax1.scatter(x=object_sizes["width"], y=object_sizes["height"], c=k_means_classes, cmap="prism")

print("K-means++:")

k_means_pp_clustering_model = sk_clustering.KMeans(n_clusters=5, init="k-means++")

k_means_pp_clustering_model.fit(X)

k_means_pp_classes = k_means_pp_clustering_model.predict(X)

ax2.set_title("K-means++")
ax2.scatter(x=object_sizes["width"], y=object_sizes["height"], c=k_means_pp_classes, cmap="prism")

k_means_pp_centroids = [(int(round(x)), int(round(y))) for x, y in k_means_clustering_model.cluster_centers_]
print(k_means_pp_centroids)

plt.show()
