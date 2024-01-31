# Random Forest

Random Forest is an [ensemble](../basics/ensemble.md) of [decision trees](decision_trees.md), generally trained via bagging or pasting methods.

## Algorithm

Decision Trees in a random forest are trained in the same way, except for at each partition step only random subset of predictors (usually $\sqrt{n}$) are considered. Random Forest has roughly all decision tree hyperparameters plus all bagging ensemble hyperparameters. Despite being built of a series of white box models random forest is a black box model.

!!! note annotate "Feature importance bonus"

    Random forest also provides feature importance scores, which can be accessed via `feature_importances_` property of a trained model. It returns predictor-wise mean decrease in the impurity score.

## Other variations

Extra trees (or Extremely Randomized Trees) takes the random forest training constraints one step further by assigning random thresholds during training instead of searching for a threshold minimizing the impurity score. This results in even higher bias and lower variance.

The API of extra trees models in `scikit-learn` is identical to that one of a random forest.
