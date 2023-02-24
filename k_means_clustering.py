#Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import the dataset
dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\19th,20th\2.K-MEANS CLUSTERING\Mall_Customers.csv")


#Removing the unnecessary attributes from the dataset to form the I.V as here for clustering algorithms, there is no D.V
X = dataset.iloc[:, [3, 4]].values


#Using the elbow method, we have to find out the optimal number of clusters
from sklearn.cluster import KMeans
#We are going to find out the optimal number of clusters & here we have to use the elbow method


wcss = []
#To plot the elbow method we have to compute WCSS for 10 different number of clusters since we gonna have 10 itterations
#We are going to write a for loop to create a list of 10 different wcss for the 10 number of clusters 
#That is why we have to initialise wcss[] & we start our loop
#We choose 1-11 becuase the 11 bound is excluded & we want 10 wcss
#How ever the first bound is included so here i = 1,2,3,4,5,6,7,8,9,10
#Now in each iteration of loop we are going to do 2 things i.e 1st we are going to fit the k-means algorithm into our data X 
#And then we are going to compute the WCSS
#Now lets fit k-means to our data X


for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
#For wcss calculation, we have a very good parameter called inertia_ . Credit goes to sklearn, that computes the sum of squares


#Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = "k-means++", random_state = 42)
y_kmeans = kmeans.fit_predict(X)


#Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "red", label = "Cluster 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "blue", label = "Cluster 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "green", label = "Cluster 3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = "cyan", label = "Cluster 4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = "magenta", label = "Cluster 5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = "yellow", label = "Centroids")
plt.title("Clusters of customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()


#To find out which customer belongs to which cluster
dataset_test = dataset
dataset_test["Cluster No.(K-Means)"] = pd.DataFrame(y_kmeans+1)
dataset_test.to_csv("New Mall Customers_k-means.csv")


#To find out the path location where exactly the new .csv file saved in our machine
import os
os.getcwd()
