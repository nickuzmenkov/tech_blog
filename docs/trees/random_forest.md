# Random Forest

_Random Forest_ is an [[Ensemble Learning|ensemble]] of [[Decision Trees|Decision Trees]], generally trained via [[Ensemble Learning#Bagging|bagging]] (or sometimes [[Ensemble Learning#Pasting|pasting]]).

Decision Trees in Random Forest are trained in the same way ([[Decision Trees#Recursive Partitioning|Recursive Partitioning]]) except for at each partition step only random subset of predictors (usually $\sqrt{n}$) are considered.

Random Forest has rougly all [[Decision Trees#Regularization Hyperparameters|Decision Tree]] hyperparameters plus all [[Ensemble Learning#Bagging|Bagging Ensemble]] hyperparameters.

> 💡 #warning Despite being built of a series of [[base/Statistics/Notation#White Box Model|White Box]] models random forest is a [[base/Statistics/Notation#Black Box Model|Black Box]] model.

> 💡 Random forest also provides variable importance scores, which can be accessed via `feature_importances_` property of a fitted model. It returns predictor-wise mean decrease in the [[Decision Trees#Gini Impurity|Gini Impurity]] score.

#code 
- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.ensemble.RandomForestRegressor`

## Extra-Trees
_Extra-Trees_ (or _Extremely Randomized Trees_) takes the [[#Random Forest]] training constraints one step further by assigning random thresholds during [[Decision Trees#Recursive Partitioning|Recursive Partitioning]] instead of searching for a threshold minimizing the [[Decision Trees#Impurity Measures|cost function]].

This results in even higher [[base/Statistics/Notation#Bias|bias]] and lower [[base/Statistics/Notation#Variance|variance]].

> 💡The API of Extra-Trees models in scikit-learn is identical to [[base/Machine Learning/Random Forest]].

#code 
- `sklearn.ensemble.ExtraTreesClassifier`
- `sklearn.ensemble.ExtraTreesRegressor`

## Examples
### Assessing Feature Importance #example 
Feature importance of individual pixels of MNIST dataset

![[assets/img/Machine Learning/Random Forest/01.png|500]]

#sourcecode [[Random Forest Code#Feature Importance]].
