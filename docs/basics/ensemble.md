# Ensemble

## Voting Classifiers
It can be shown from the [[Data and Sampling#Binomial distribution|Binomial distribution]] that an ensemble of _weak learners_ (i.e. only slightly better than random gessing) can yield overwhelmingly high accuracy in combination.

> 💡 #warning This is only the case when all the models are independent, which is clearly not the case with models trained on the same data. On the contrary, they tend to make the _same mistakes_ which can further reduce the model.

So the major issue here is to get models as diverse as possible to avoid correlated errors.

### Hard Voting Classifier
_Hard voting classifier_ simply takes the [[Exploratory Data Analysis#Mode|mode]] of individual models.

### Soft Voting Classifier
_Soft voting classifier_ averages probability of each class and takes the highest probability, thus giving more weight to very confident votes.

> 💡 #warning In general, soft voting classifiers are better, but they require all models to be able to predict probabilities.

#code 
- `sklearn.ensemble.VotingClassifier`
- `sklearn.ensemble.VotingRegressor`

## Bagging and Pasting
One way to achieve model diversity is to train each predictor on different random subsets of the training data.

This generally leads to bot higher [[base/Statistics/Notation#Bias|bias]] and [[base/Statistics/Notation#Variance|variance]] for each predictor, but the ensemble often has roughly the same bias but lower variance than a single predictor trained on the entire original training set.

### Bagging
Bagging is short for [[Data and Sampling#Bootstrap|Bootstrap]] Aggregating. It denotes forming random subsets from initial dataset _with replacement_ (bootstrapping the data) and training each predictor on random subsamples.

> 💡Drawing samples with replacement means that the same instance can occur multiple times not only across multiple predictors' dataset, but also across the same dataset.

#code 
- `sklearn.ensemble.BaggingClassifier(..., bootstrap=True)`.
- `sklearn.ensemble.BaggingRegressor(..., bootstrap=True)`.

### Pasting
Similarly, pasting denotes forming random subsets of initial data _without replacement_.

#code
- `sklearn.ensemble.BaggingClassifier(..., bootstrap=False)`
- `sklearn.ensemble.BaggingRegressor(..., bootstrap=False)`

### Out-of-Bag Evaluation
Since each predictor has its own unseen part of the dataset, out-of-bag evaluation is a handy way to estimate the overall ensemble score.

For [[#Pasting]] ensemble fraction (or number) of out-of-bag samples must be provided explicitly: #code `sklearn.ensemble.BaggingClassifier(..., max_samples=0.8)`.

For [[#Bagging]] ensemble it can be shown that nearly 37% of the training data remains unseen due to replacements, hence no need for explicit data partitioning.

> 💡 #warning Out-of-bag scores (property `oob_score_`) is switched off by default. To make it accessible one must provide `oob_score=True` flag when initializing the `BaggingClassifier` or `BaggingRegressor` instance.

## Random Patches and Random Subspaces
The same logic of [[#Bagging and Pasting]] can be applied for predictor variables instead of (or along with) instances.

Both these techniques generally lead to slightly higher [[base/Statistics/Notation#Bias|bias]] but lower [[base/Statistics/Notation#Variance|variance]].

### Random Patches
Sampling both features and instances is called _Random Patches_.

#code
- `sklearn.ensemble.BaggingClassifier(..., max_samples=0.8, bootstrap=True/False, max_features=0.8, bootstrap_features=True)`
-  `sklearn.ensemble.BaggingRegressor(..., max_samples=0.8, bootstrap=True/False, max_features=0.8, bootstrap_features=True)`

### Random Subspaces
Sampling only features and drawing all instances without replacement is called _Random Subspaces_.

#code
- `sklearn.ensemble.BaggingClassifier(..., max_samples=1.0, bootstrap=False, max_features=0.8, bootstrap_features=True/False)`
-  `sklearn.ensemble.BaggingRegressor(..., max_samples=1.0, bootstrap=False, max_features=0.8, bootstrap_features=True/False)`

## Boosting
-

## Stacking
-
