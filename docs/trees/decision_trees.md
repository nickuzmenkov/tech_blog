# Decision Trees

- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.tree.DecisionTreeRegressor`

## Introduction

>💡 #story Tree models, trees, decision trees, or classification and regression trees (CART) were initially developed by Leo Breiman et al. in 1984.

Trees are most commonly used in more powerful ensemble models: bagging ensembles (see [[base/Machine Learning/Random Forest|Random Forest]]) and boosted ensembles (see [[Gradient Boosted Trees]]) and almost never standalone.

Tree is basically a set of if-else rules (_branches_) splitting data per one feature at a time to minimize data impurity in both splits (_leaves_). Trees can thus discover nonlinear patterns in data and spot important variables and [[Regression#Interactions|interactions]].

Like regression models, a single decision tree is a [[base/Statistics/Notation#White Box Model|White Box]] model compared to tree-based ensembles which are [[base/Statistics/Notation#Black Box Model|Black Box]] models.

## Recursive Partitioning
Decision trees available in scikit-learn are CART trees. The training algorithm used in CART is called _recursive partitioning_:

1.  For each predictor variable $X_j$:
    1.  Split the dataset into two partitions for each value of $X_{i,j}$.
    2.  Measure cost function for each pair of possible partitions.
    3.  Record the value of $X_{i,j}$ of the best split (i.e. the lowest loss).
2.  Select the best split among all best splits of each variable.
3.  Repeat 1-2 for each of the resulting splits.

The cost function is based on one of [[#Impurity Measures]] and given by

$$ J(k, t_k) = \frac{m_1}{m}I_1 + \frac{m_2}{m}I_2 $$

for a selected threshold $t_k$ of a particular predictor variable $k$. Here $m_i/m$ are sample fractions and $I_i$ are [[#Impurity Measures|impurity scores]].

> 💡 #warning Recursive partitioning is produces only binary trees. Other algorithms (e.g. ID3, C4.5; Quinlan, 1993) can make more than 2 splits at a time.

## Impurity Measures

### Gini Impurity
_Gini Impurity_ is given by

$$ G_i=1 - \sum_{k=1}^n{p_{i,k}} $$

### Entropy
_Entropy is given by_

$$ H_i=-\sum_{k=1}^n{p_{i,k}\log_2{p_{i,k}}} $$

### Mean Squared Error

[[Regression Metrics#Mean Squared Error MSE|MSE]] is used as impurity measure for numeric target variables.

## Regularization Hyperparameters
The [[#Recursive Partitioning]] algorithm goes on until it meets one of the stop conditions listed below. These stop conditions act like regularization hyperparameters preventing the tree from overfitting
-   Number of samples at leaf is less than `min_samples_leaf`.
-   The fraction of sum of weights at leaf is less than `min_weight_fraction_leaf` (for weighted samples; otherwise just fraction of number of samples).
-   Number of samples at branch is less than `min_samples_split`.
-   Depth of the tree (i.e. the number of steps between the root and the farthest leaf) is more than `max_depth`.
-   Number of leaves is more than `max_leaf_nodes`.
-   Decrease of impurity of the split is less than `min_impurity_decrease`.

## Classification
Predicted class $\hat{y}$ is simply the mode class of the leaf node.

Class probability $\hat{p}$ is calculated as the ratio of the number predicted class instances to all instances of the leaf node.

## Regression
Predicted value $\hat{y}$ is simply the average value of all the instances of the leaf node.

## #complexity
Making a prediction (i.e. _traversing_ the tree): given that decision trees are generally balanced, is $O(\log_2{m})$ and independent of the number of features.

Training requires comparison among all features and all samples at each node $O(n\times m\log_2{m})$.

## Examples
### Iris Classification #example 
Based on a set of 150 records with four variables (petal length, petal width, sepal length, and sepal width), predict one of three types of iris—_Setosa_, _Virginica_, or _Versicolour_.

#### Solution
![[assets/img/Machine Learning/Decision Trees/01.png|500]]

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
model.fit(X_train, y_train)

feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
class_names = ["Setosa", "Versicolour", "Virginica"]

plot_tree(
	model,
	feature_names=feature_names,
	class_names=class_names,
	node_ids=True,
)
plt.show()
```

Text representation is also available:

```python
>>> from sklearn.tree import export_text
>>> export_text(model, feature_names=feature_names)
|--- Petal Width <= 0.80
|   |--- class: 0
|--- Petal Width >  0.80
|   |--- Petal Length <= 4.95
|   |   |--- class: 1
|   |--- Petal Length >  4.95
|   |   |--- class: 2
```
