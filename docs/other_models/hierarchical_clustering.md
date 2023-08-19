# Hierarchical Clustering

Hierarchical clustering does not need the number of clusters to be specified. It is computationally expensive, as it requires pairwise distance of the entire dataset.

#code
- `scipy.cluster.hierarchy.linkage`
- `sklearn.cluster.AgglomerativeClustering`

## Agglomerative Algorithm:

1.  Assign each record its own cluster (i.e. begin with smallest possible single-record clusters)
2.  Compute dissimilarity between all the pairs of clusters
3.  Merge two least dissimilar clusters
4.  Repeat 2-3 until some convergence criterion is met or all the records are merged into a single cluster.

[[#Dissimilarity Measures|Dissimilarity]] is measured based on some distance metric. The distance is computed between all the pairs of dots in the two clusters.

## Dissimilarity Measures
### Complete Linkage

Complete linkage is given by

$$
\text{Complete linkage} = \max{d(a_i,b_j)}\ i=\overline{1,N}, j=\overline{1,M}
$$

### Single Linkage

Single linkage is given by

$$
\text{Single linkage} = \min{d(a_i,b_j)}\ i=\overline{1,N}, j=\overline{1,M}
$$

### Average Linkage

Average linkage is given by

$$
\text{Average linkage} = \frac{1}{n\cdot m}\sum_{i=1}^n{\sum_{j=1}^m{d(a_i,b_j)}}
$$

### Minimum Variance

Another dissimilarity metric is called minimum variance (or _Ward’s method_), which tends to minimize within-cluster sum of squares, like in K-Means.

## Dendrogram
Dendrogram is another way to get the appropriate number of clusters. It shows dissimilarity score between the merged clusters.

![[assets/img/Machine Learning/Unsupervised Learning Models/Hierarchial Clustering/01.png|500]]

#code `scipy.cluster.hierarchy.dendrogram`.
#sourcecode [[Hierarchial Clustering Code#Dendrogram]].
