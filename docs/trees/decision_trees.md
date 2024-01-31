# Decision Trees

Tree models, trees, decision trees, or classification and regression trees (CART) were initially developed by Leo Breiman et al. in 1984.

## Introduction

Trees are most commonly used in more powerful ensemble models: bagging ensembles [Random Forest](random_forest.md) and boosted ensembles ([Gradient Boosted Trees](gradient_boosted_trees.md)), and almost never standalone.

Tree is basically a set of if-else rules (or branches) splitting data per one feature at a time to minimize data impurity in both splits (or leaves). Trees can thus discover nonlinear patterns in data and spot important variables and interactions.

Like regression models, a single decision tree is a white box model compared to tree-based ensembles which are black box models.

## Algorithm

Decision trees available in `scikit-learn` are CART trees. The training algorithm used in CART is called **recursive partitioning** and goes as follows

1.  For each predictor variable $X_j$:
    1.  Split the dataset into two partitions for each value of $X_{i,j}$.
    2.  Measure cost function for each pair of possible partitions.
    3.  Record the value of $X_{i,j}$ of the best split (i.e. the lowest loss).
2.  Select the best split among all best splits of each variable.
3.  Repeat steps 1-2 for each of the resulting splits.

The cost function is based on one of impurity measures and given by

$$ J(k, t_k) = \frac{m_1}{m}I_1 + \frac{m_2}{m}I_2. $$

For a selected threshold $t_k$ of a particular predictor variable $k$. Here $m_i/m$ are sample fractions and $I_i$ are impurity scores.

!!! note annotate "Only one split at a time"

    Recursive partitioning is produces only binary trees. Other algorithms (e.g. ID3, C4.5; Quinlan, 1993) can make more than 2 splits at a time.

For a trained classification tree, the predicted class is simply the mode class of the leaf node. Class probability is calculated as the ratio of the number predicted class instances to all instances of the leaf node. For a regression tree, the predicted value is simply the average value of all the instances of the leaf node.

## Impurity Measures

Here are the most common impurity measures

- Gini

    $$ G_i=1 - \sum_{k=1}^n{p_{i,k}^2}. $$
    
    where $n$ is the number of classes, $p_{i,k}$ is the proportion of items labeled with class $k$.
    
- Entropy

    $$ H_i=-\sum_{k=1}^n{p_{i,k}\log_2{p_{i,k}}}. $$

- MSE (for regression trees)

## Regularization Hyperparameters

The recursive partitioning algorithm goes on until it meets one of the stop conditions listed below. These stop conditions act like regularization hyperparameters preventing the tree from overfitting (names of the corresponding parameters in `scikit-learn` are given in the brackets)

-   Number of samples at leaf is less than (`min_samples_leaf`).
-   The fraction of sum of weights at leaf is less than (`min_weight_fraction_leaf`) - for weighted samples - otherwise just fraction of number of samples.
-   Number of samples at branch is less than (`min_samples_split`).
-   Depth of the tree (i.e. the number of steps between the root and the farthest leaf) is more than (`max_depth`).
-   Number of leaves is more than (`max_leaf_nodes`).
-   Decrease of impurity of the split is less than (`min_impurity_decrease`).

## Computational Complexity

- Making a prediction with a balanced tree (which is usually true) that has $m$ branches, is $O(\log_2{m})$ and independent of the number of features.
- Training requires comparison among all $n$ features and all $m$ samples at each node $O(n \times m\log_2{m})$.

## Example

Iris classification: based on a set of 150 records with four variables (petal length, petal width, sepal length, and sepal width), predict one of three types of iris - Setosa, Virginica, or Versicolour.

```python
import matplotlib.pyplot as plt
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
    export_text,
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
model.fit(X_train, y_train)

feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
class_names = ["Setosa", "Versicolour", "Virginica"]

export_text(  # text representation
    model,
    feature_names=feature_names,
)
plot_tree(  # graph representation
	model,
	feature_names=feature_names,
	class_names=class_names,
	node_ids=True,
)
plt.show()
```
