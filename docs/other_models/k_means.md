# K-Means Clustering

K-means was the first clustering method.

The algorithm is straightforward:

1.  Start with K random cluster means
2.  Assign each point to the closest cluster
3.  Get new cluster means as the center of all points in the cluster
4.  Repeat 2-3 until convergence

The exact solution is too computationally difficult, so this is a more efficient solution, but it is suggested to run multiple times on different subsets of data.

Elbow chart is a way to get the appropriate number of clusters for the task. It shows number of clusters on x-axis versus explained variance on the y-axis.

For K-means, it can be produced by accessing model’s attribute `inertia_` and refitting the algorithm multiple times.