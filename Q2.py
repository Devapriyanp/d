import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


newdf = pd.read_csv("laptops.csv")


newdf.ffill(inplace=True)
newdf.drop_duplicates(inplace=True)


newdf = pd.get_dummies(newdf)


newX = newdf.select_dtypes(include='number').iloc[:, :5]


scaler = StandardScaler()
newX = scaler.fit_transform(newX)


newX = pd.DataFrame(newX)


model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(newX)


print("Cluster Labels:", labels)
print("SSE:", model.inertia_)
print("Silhouette Score:", silhouette_score(newX, labels))


plt.figure(figsize=(6,6))
plt.scatter(newX.iloc[:,0], newX.iloc[:,1], c=labels)


plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()