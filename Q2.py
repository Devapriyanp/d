import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


newdf = pd.read_csv(".csv")


newdf.ffill(inplace=True)
newdf.drop_duplicates(inplace=True)


newdf = pd.get_dummies(newdf)


newX = newdf.select_dtypes(include='number').iloc[:, :2]


model = KMeans(n_clusters=3)

labels = model.fit_predict(newX)


print("SSE:", model.inertia_)
print("Silhouette Score:", silhouette_score(newX, labels))


plt.figure(figsize=(6,6))
plt.scatter(newX.iloc[:,0], newX.iloc[:,1], c=labels)



plt.title("K-Means Clustering")
plt.xlabel(newX.columns[0])
plt.ylabel(newX.columns[1])
plt.show()