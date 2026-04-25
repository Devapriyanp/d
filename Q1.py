import pandas as pd
import matplotlib.pyplot as plt


newdf = pd.read_csv(".csv")

print("First 5 rows:\n", newdf.head())
print("\nInfo:\n")
print(newdf.info())
print("Mean:\n", newdf.mean(numeric_only=True))
print("Median:\n", newdf.median(numeric_only=True))
print("Mode:\n", newdf.mode().iloc[0])


print("Range:\n", newdf.max(numeric_only=True) - newdf.min(numeric_only=True))
print("Variance:\n", newdf.var(numeric_only=True))
print("Standard Deviation:\n", newdf.std(numeric_only=True))
print("IQR:\n", newdf.quantile(0.75, numeric_only=True) - newdf.quantile(0.25, numeric_only=True))



print("Skewness:\n", newdf.skew(numeric_only=True))
print("Kurtosis:\n", newdf.kurt(numeric_only=True))
print("\nShape:", newdf.shape)
print("\nColumns:", newdf.columns)


newdf.ffill(inplace=True)


newdf.drop_duplicates(inplace=True)


newdf_encoded = pd.get_dummies(newdf)

print("\nAfter Preprocessing:\n", newdf_encoded.head())


newdf_num = newdf_encoded.select_dtypes(include=['number']).iloc[:, :10]




newdf_num.hist(figsize=(6,6))
plt.suptitle("Histogram")
plt.show()

plt.figure(figsize=(6,6))
newdf_num.boxplot()
plt.title("Box Plot")
plt.xticks(rotation=45)
plt.show()

corr = newdf_num.corr()

plt.figure(figsize=(8,6))
plt.imshow(corr)
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Heatmap")
plt.show()


if newdf_num.shape[1] >= 2:
    plt.figure(figsize=(6,6))
    plt.scatter(newdf_num.iloc[:,0], newdf_num.iloc[:,1])
    plt.xlabel(newdf_num.columns[0])
    plt.ylabel(newdf_num.columns[1])
    plt.title("Scatter Plot")
    plt.show()



