# Linear Classification

## Logistic Regression

**Logistic Regression** is a direct extension of the [Linear Regression](linear_regression.md) with its simplicity and high speed for classification tasks. Logistic Regression returns the logistic of the result

$$
\hat{p}=\sigma(\boldsymbol{\theta}^T\boldsymbol{x}),
$$

where $\sigma$ is a **standard logistic function** (or sigmoid)

$$
\sigma(t)=\frac{1}{1 + \exp{(-t)}}.
$$

Thus, the output is scaled to the $[0;1]$ range of the probability that a particular instance is positive ($y=1$).

Output of the original regression equation can be expressed as a **logit function** (or logarithm of the odds, which is the inverse of the logistic function)

$$
\boldsymbol{\theta}^T\boldsymbol{x}=\sigma^{-1}(p)=\ln{(\text{odds}(\hat{p}))}=\ln{\left(\frac{p}{1-p}\right)}
$$

The cost function for classification tasks is known as **cross entropy loss** (or log loss). In the case of binary classification, it's called **binary cross entropy** and is given by

$$
J(\boldsymbol{\theta})=-\frac{1}{m}\sum_{i=1}^m{\left(y^{(i)}\log{\hat{p}^{(i)}}+\left(1-y^{(i)}\right)\log{\left(1-\hat{p}^{(i)}\right)}\right)}.
$$

!!! example annotate "Extreme Values Example"

    Imagine that the ground truth $y=1$, while the predicted value $\hat{p} \approx 0$. The penalty for this sample is then close to $-\log{0}$, which is $\infty$. The same is true for the opposite case, where the ground truth is equal to 0 and the predicted value is close to 1.

There is no known closed-form solution for computing the value of $\boldsymbol{\theta}$ that minimizes binary cross entropy. However, just like MSE, cross entropy is a convex function. Thus, Batch Gradient Descent is guaranteed to find the optimal solution within the specified tolerance.

The cross entropy gradient $\nabla_{\boldsymbol{\theta}}{J(\boldsymbol{\theta})}$ is calculated in the same way as for [MSE](linear_regression.md#closed-form-solution) and is given by:

$$
\nabla_{\boldsymbol{\theta}}{J(\boldsymbol{\theta})}= \frac{2}{m}\boldsymbol{X}^T\left(\sigma\left({\boldsymbol{X}\boldsymbol{\theta}}\right)-\boldsymbol{y}\right).
$$

!!! note annotate "Scikit-Learn Implementation"

    The [`LogisticRegression` :octicons-link-external-16:][logistic-regression], the `scikit-learn` implementation of logistic regression, comes with the $\ell_2$ penalty turned on by default, allowing you to adjust the strength of the regularization. Read more about regularization in the [Regularization](regularization.md) manual.

!!! note annotate "Categorical Features"
    
    As with [Linear Regression](linear_regression.md), all categorical variables must first be one-hot encoded, omitting one of the classes.

## Softmax Regression

The logistic regression model can be generalized to support multiple classes directly, without the need to train one-versus-one (OvO) or one-versus-rest (OvR) classifiers. The resulting probability that an instance belongs to class $k$ is given by

$$
\hat{p}_k=\text{softmax}(\boldsymbol{s})_k=\frac{\exp{(\boldsymbol{s}_k)}}{\sum_i exp{(\boldsymbol{s}_i)}}
$$

where $\boldsymbol{s}$ is a vector containing the values of each class $k$ for the instance $\boldsymbol{x}$

$$
\boldsymbol{s}=\boldsymbol{\Theta}\boldsymbol{x}.
$$

All class-wise parameter vectors $\boldsymbol{\theta}^{(k)}$ together form a parameter matrix $\boldsymbol{\Theta}$.

Similar to binary classification, the cost function used is cross-entropy, but in a generalized version known as **categorical cross-entropy**

$$
J(\boldsymbol{\Theta})=-\frac{1}{m}\sum_{i=1}^m{\sum_{k=1}^l{y_k^{(i)}\log{\left(\hat{p}_k^{(i)}\right)}}}
$$

where $l$ is the number of classes.

The gradient vector with respect to the $\boldsymbol{\theta}^{(k)}$ parameter vector $\nabla_{\boldsymbol{\theta}^{(k)}}{J(\boldsymbol{\theta})}$ is calculated similarly to the gradient of the [MSE](linear_regression.md#closed-form-solution) and is given by

$$
\nabla_{\boldsymbol{\theta}^{(k)}}{J(\boldsymbol{\Theta})}= \frac{2}{m}\boldsymbol{X}^T\left(\text{softmax}\left({\boldsymbol{X}\boldsymbol{\theta}}\right)-\boldsymbol{y}\right).
$$

!!! note annotate "Scikit-Learn Implementation"

    The [`LogisticRegression` :octicons-link-external-16:][logistic-regression] implementation in `scikit-learn` uses One-versus-Rest wrapper for multiclass tasks instead.

[logistic-regression]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
