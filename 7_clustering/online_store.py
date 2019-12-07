import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as sk_clustering
import sklearn.preprocessing as sk_preprocessing
import scipy.cluster.hierarchy as sp_clustering


def set_printing_options():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)


set_printing_options()
users_df = pd.read_csv("data/customer_online_closing_store.csv")
# print(users_df)

users_df["return_rate"] = users_df["items_returned"] / users_df["items_purchased"]
users_df["average_price"] = users_df["total_spent"] / users_df["items_purchased"]

X = users_df[["average_price", "return_rate", "overall_rating"]]
print(X)

min_max_scaler = sk_preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

plt.title("Customer dendrogram")

linkage_method = "ward"

dendrogram = sp_clustering.dendrogram(sp_clustering.linkage(X, method=linkage_method))

agglomerative_model = sk_clustering.AgglomerativeClustering(n_clusters=4, linkage=linkage_method)

agglomerative_model.fit(X)

classes = agglomerative_model.labels_

users_df["class"] = classes

# print(users_df)

user_pivot_table = users_df.pivot_table(index="class",
                                        values=["average_price", "return_rate", "overall_rating", "customer_id"],
                                        aggfunc={"average_price": np.mean, "return_rate": np.mean,
                                                 "overall_rating": np.mean, "customer_id": len})

print(user_pivot_table)

plt.show()
