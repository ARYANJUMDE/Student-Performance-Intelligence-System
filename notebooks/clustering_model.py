import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset

df=pd.read_csv("data\student_data.csv")

# Data Preprocessing

# Information about dataset

print(df.info())

# Mathematical analysis of dataset

print(df.describe())

# Top 5 rows of dataset

print(df.head())

# Columns in dataset

print(df.columns)

# Checking for missing values

print(df.isnull().sum())

# Removing rows with missing values

df.dropna(inplace=True)

# Converting categorical variables to numerical using one-hot encoding

df_categorical=df.select_dtypes(include="object").columns
df = pd.get_dummies(df, columns=df_categorical, drop_first=True)
df = df.astype(int)

X=df[["studytime","absences","G1","G2","failures"]]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

# Elbow method to find optimal number of clusters

inertia=[]
for k in range(1,11):
    k_means=KMeans(n_clusters=k,random_state=42)
    k_means.fit(X_scaled)
    inertia.append(k_means.inertia_)
    
# Plotting Elbow Curve

plt.plot(range(1,11),
        inertia,
        marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Fitting KMeans with optimal number of cluster by observing elbow graph

k_means = KMeans(n_clusters=6,random_state=42)
k_means.fit(X_scaled)
clusters=k_means.labels_
df["Cluster"]=clusters

# Printing the first 5 rows of the dataset with cluster labels

print(df[["studytime","absences","G1","G2","failures","Cluster"]].head())

# Evaluating the clustering using Silhouette Score

score=silhouette_score(X_scaled,clusters)
print("Silhouette Score:",score)

# Visualizing the clusters using a scatter plot

plt.figure(figsize=(10,6))
sns.scatterplot(
    x=df["G1"],
    y=df["G2"],
    hue=df["Cluster"],
    palette="deep"
)
plt.title("Student Clusters")
plt.show()

# PCA Visualisation 

pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))

sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=df["Cluster"],
    palette="deep"
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Cluster Visualization")
plt.show()


# Explained Variance Ratio

print("PCA Variance Ratio",pca.explained_variance_ratio_)

