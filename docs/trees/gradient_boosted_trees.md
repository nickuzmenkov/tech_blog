# Gradient Boosted Trees

The general idea of boosting is to combine a collection of weak learners so that each one corrects its predecessor.

The most common boosting algorithms are [[#AdaBoost]], [[#Gradient Boosting]], and [[#Stochastic Gradient Boosting]].

> 💡 #warning An important drawback of boosting algorithms is that training cannot be parallelized: hence the linear increase of the training time with the number of predictors.

## AdaBoost
> 💡 #warning The algorithm below is given only for a classification task.

_Adaptive Boosting_ (or _AdaBoost_) starts with uniform sample weight distribution assigning each sample's weight  $\boldsymbol{w}^{(i)}$ to $1 / m$.

For each predictor the error rate is given by

$$
r_j =  
	\sum_{\substack{i=1 \\ \hat{y}\ne y}}^m{\boldsymbol{w}^{(i)}} 
	\bigg/
	\sum_{i=1}^{m}{\boldsymbol{w}^{(i)}}
$$

Based on the error rate each predictor is assigned its weight $\alpha_j$

$$
\alpha_j = \eta \log{\frac{1 - r_j}{r_j}}
$$

where $\eta$ is learning rate hyperparameter ($\eta = 1$ by default).

Then instance weights are updated

$$
\boldsymbol{w}^{(i)} \leftarrow
\begin{cases}
\boldsymbol{w}^{(i)} & \text{if}\ \hat{y}^{(i)}=y^{(i)}
\\
\boldsymbol{w}^{(i)} \exp{\alpha_j} & \text{if}\ \hat{y}^{(i)} \ne y^{(i)}
\end{cases}

\quad i=\overline{1,m}
$$

and renormalized (i.e. divided by $\sum_{i=1}^m{\boldsymbol{w}^{(i)}}$)

The process is repeated until the last predictor is reached or until a perfect model is found.

Ensemble predictions are then given by the majority of the weighted votes

$$
\hat{y} = \underset{k}{\text{argmin}} \sum_{
		\substack{j=1 \\ \hat{y}=k}
	}^N{\alpha_j}
$$

where $N$ is the number of predictors.

#code 
- `sklearn.ensemble.AdaBoostClassifier`
- `sklearn.ensemble.AdaBoostRegressor`

## Gradient Boosting
-
# Continue from page 267
Gradient boosting instead casts the problem as an optimization of a cost function by fitting the model to a pseudo-residual, which has effect of training more heavily on larger residuals.

Stochastic gradient boosting in addition applies random predictor and record sampling at each step.


## Stochastic Gradient Boosting
The most widely used boosting implementation is `xgboost` (stochastic gradient boosting), which has `XGBClassifier` and `XGBRegressor` classes.
