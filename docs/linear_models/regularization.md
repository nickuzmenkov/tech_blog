# Regularization

## Bias/Variance Trade-off

Model’s generalization error can be represented as sum of three errors:

-  [[base/Statistics/Notation#Bias|Bias]]
-  [[base/Statistics/Notation#Variance|Variance]]
-   _Irreducible_ error is the part of generalization error due to the noise in the data.

Increasing model’s complexity typically decreases its bias and increases variance and vice versa.

One approach to regularization is _weight decay_ (i.e. penalizing the model for large weight values). It is used in the models below:

-   Ridge Regression
-   Lasso Regression
-   Elastic Net

!!! note
    
    Ridge Regression is a good default choice. Lasso and Elastic Net can be used instead when it’s not obvious what features are the most important hence they both tend to eliminate useless features.

## Ridge Regression

_Ridge regression_ (also called _Tikhonov regularization_) adds a _regularization term_ equal to half square of the $\ell_2$ norm of the weight vector to the $\text{MSE}$ cost function

$$ J(\boldsymbol{w})=\text{MSE}+\frac{\alpha}{2}\|\boldsymbol{w}\|_2^2=\text{MSE}+\frac{\alpha}{2}\boldsymbol{w}^T\boldsymbol{w} $$

The normal equation thus changes to

$$ \boldsymbol{\theta}=(\boldsymbol{X}^T\boldsymbol{X}+\alpha\boldsymbol{A})^{-1}\boldsymbol{X}^T\boldsymbol{y} $$

The gradient vector of the cost function changes to

$$ \nabla_{\boldsymbol{\theta}}{\text{MSE}(\boldsymbol{\theta})} = \frac{2}{m}\boldsymbol{X}^T(\boldsymbol{X}\boldsymbol{\theta}-\boldsymbol{y}) + \alpha\boldsymbol{w} $$

- `sklearn.linear_model.Ridge`
- `sklearn.linear_model.SGDRegressor(penalty="l2")`

## Lasso Regression

_Least Absolute Shrinkage and Selection Operator_ (LASSO) _Regression_ adds a _regularization term_ equal to $\ell_1$ norm of the weight vector to the $\text{MSE}$ cost function

$$ J(\boldsymbol{w})=\text{MSE}+\alpha\|\boldsymbol{w}\|_1 $$

!!! Warning

    Unlike Ridge Regression, Lasso Regression tends to eliminate the weights of the least important features.

- `sklearn.linear_model.Lasso`
- `sklearn.linear_model.SGDRegressor(penalty="l1")`

## Elastic Net

Elastic net adds two regularization terms at once corresponding to $\ell_1$ and $\ell_2$ norms of the weight vector

$$ J(\boldsymbol{w})=\text{MSE}+r\alpha\|\boldsymbol{w}\|_1+\frac{1-r}{2}\alpha\|\boldsymbol{w}\|_2^2 $$

where $r$ is the ratio between the terms (e.g. $r=1$ is equivalent to Lasso Regression, while $r=0$ is equivalent to Ridge Regression).

!!! note

    Elastic Net’s behaviour is similar to Lasso Regression.

- `sklearn.linear_model.ElasticNet(l1_ratio=0.5)`
- `sklearn.linear_model.SGDRegressor(penalty="elasticnet", l1_ratio=0.5)`