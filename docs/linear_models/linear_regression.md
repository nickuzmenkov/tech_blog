# Linear Regression

Linear Regression is defined by

$$
\hat{y} = \boldsymbol{\theta}^T \boldsymbol{x},
$$

where $\hat{y}$ represents the predicted value, $\boldsymbol{\theta}$ is a $(n, 1)$ vector of model weights, $\boldsymbol{x}$ is a $(n, 1)$ vector of features, and $n$ is the number of features.

The mean squared error $\text{MSE}(\boldsymbol{\theta})$ is commonly used as the cost function

$$
\text{MSE}(\boldsymbol{\theta})=\frac{1}{m} \sum_{i=1}^m{(\boldsymbol{\theta}^T \boldsymbol{x}^{(i)} - y^{(i)})^2} = \frac{1}{m}\|\boldsymbol{X}\boldsymbol{\theta}-\boldsymbol{y}\|_2^2,
$$

where $m$ represents the number of training samples, $\boldsymbol{X}$ is an $(m, n)$ matrix containing the training samples.

!!! note annotate "Euclidean Norm"

    The $\|| x \||_2$ notation represents the Euclidean (or L2) norm of the vector. Similarly, $\|| x \||_2^2$ represents the squared Euclidean norm.

There are 2 main solution approaches:

- [Closed-form solution](#closed-form-solution) computes the optimal parameters directly.
- [Gradient descent](#gradient-descent) iteratively finds near-optimal parameters by fitting either the entire training set or its batches.

Both approaches are described in the following sections.

!!! note annotate "Categorical Features"
    
    Regardless of the implementation you use, note that all categorical variables must first be one-hot encoded, omitting one of the classes.

### Closed-Form Solution

To minimize the cost function $\text{MSE}(\boldsymbol{\theta})$, we seek the point at which its gradient vanishes

$$
\nabla_{\boldsymbol{\theta}}{\text{MSE}(\boldsymbol{\theta})}=0,
$$

$$
\frac{1}{m}\nabla_{\boldsymbol{\theta}} \||\boldsymbol{X}\boldsymbol{\theta}-\boldsymbol{y}\||_2^2=0.
$$

Considering that $\||\boldsymbol{X}\||_2^2=\boldsymbol{X}^T\boldsymbol{X}$, we have

$$
\nabla_{\boldsymbol{\theta}}{(\boldsymbol{X}\boldsymbol{\theta}-\boldsymbol{y})^T(\boldsymbol{X}\boldsymbol{\theta}-\boldsymbol{y})}=0,
$$

$$
\nabla_{\boldsymbol{\theta}}{(\boldsymbol{\theta}^T\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\theta}-\boldsymbol{\theta}^T\boldsymbol{X}^T\boldsymbol{y}-\boldsymbol{y}^T\boldsymbol{X}\boldsymbol{\theta}+\boldsymbol{y}^T\boldsymbol{y})}=0.
$$

Using scalar triple product property gives

$$
\nabla_{\boldsymbol{\theta}}{(\boldsymbol{\theta}^T\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\theta}-2\boldsymbol{\theta}^T\boldsymbol{X}^T\boldsymbol{y}+\boldsymbol{y}^T\boldsymbol{y})}=0.
$$

Finally, applying the gradient gives

$$
2\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\theta}-2\boldsymbol{X}^T\boldsymbol{y}=0,
$$

$$
\boxed{\boldsymbol{\theta}=(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}}.
$$

The resultant equation is recognized as the **normal equation**.

!!! note annotate "Computational Complexity"

    Due to the computational complexity, which is about $n^3$ for finding the inverse matrix alone, the closed-form solution may be unsuitable for large $m$ and $n$ (i.e., large number of instances and features, respectively)..

??? example annotate "Example: Fit Noisy Linear Function"

    Fit the $y=3 \cdot x + 4$ equation given 100 noisy samples.

    The problem can be solved directly with `numpy` with a little trickery:
    
    ```python
    import numpy as np
    
    np.random.seed(42)
    size = 100
    
    noise = 0.25 * np.random.randn(size, 1)
    x1 = np.random.rand(size, 1)
    x2 = np.ones_like(x1)  # (1)
    X = np.c_[x1, x2]
    y = 3 * x1 + 4 + noise

    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    print(f"The equation is: y={theta[0, 0]:.2f} * x + {theta[1, 0]:.2f}.")
    ```

    1. Although there is only one argument to the function, we need to use 2 features (one being the argument and the other being all ones) to find the intercept.

    Or more traditionally, with `scikit-learn`:

    ```python
    import numpy as np
    from sklearn.linear_model import LinearRegression

    np.random.seed(42)
    size = 100
    
    noise = 0.25 * np.random.randn(size, 1)
    X = np.random.rand(size, 1)
    y = 3 * X + 4 + noise
    
    model = LinearRegression()
    model.fit(X, y)
    print(f"The equation is: y={model.coef_[0, 0]:.2f} * x + {model.intercept_[0]:.2f}.")
    ```

### Gradient Descent

**Gradient descent** is a generic optimization algorithm capable of finding optimal solutions to a wide range of problems.

!!! note annotate "Mind Scaling"

    When using gradient descent, all inputs must be on the same scale for good convergence.

Gradient descent begins with random initialization of model parameters and tweaks the model’s parameter vector $\boldsymbol{\theta}$ opposite to the gradient of the cost function at each iteration

$$ \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}}{\text{MSE}(\boldsymbol{\theta})} $$

where $\eta$ is the learning rate.

The gradient vector $\nabla_{\boldsymbol{\theta}}{\text{MSE}(\boldsymbol{\theta})}$ is given by

$$ \nabla_{\boldsymbol{\theta}}{\text{MSE}(\boldsymbol{\theta})}= \begin{pmatrix} \frac{\partial}{\partial \theta_0} \text{MSE}(\boldsymbol{\theta}) \\ \vdots \\ \frac{\partial}{\partial \theta_n} \text{MSE}(\boldsymbol{\theta}) \end{pmatrix} = \frac{2}{m}\boldsymbol{X}^T(\boldsymbol{X}\boldsymbol{\theta}-\boldsymbol{y}) $$

(it was derived [here](https://www.notion.so/Regression-366aa3c9e2dc4d3f91dad0034473a215)). Thus, partial derivatives are given by

$$ \frac{\partial}{\partial \theta_j}\text{MSE}(\boldsymbol{\theta})=\frac{2}{m}\sum_{i=1}^m{\left(\boldsymbol{\theta}^T\boldsymbol{x}^{(i)}-y^{(i)}\right)x_j^{(i)}} $$

The iterations continue until the norm of the gradient becomes less than the specified tolerance value $\epsilon$

$$ \|\nabla_{\boldsymbol{\theta}}{\text{MSE}(\boldsymbol{\theta})}\|<\epsilon $$

The cost function $\text{MSE}(\boldsymbol{\theta})$ is convex, meaning that gradient descent is guaranteed to find the global minimum within the specified tolerance.

There are 3 types of Gradient Descent:

- **Batch gradient descent** (shown above): the entire training set is used to compute the gradient. Guaranteed to find optimal parameters for convex cost functions. Gets slow with large datasets and lots of features and does not support out-of-core computations.
- **Mini-batch gradient descent**: a single batch is used to compute the gradient. Scale well to huge datasets and supports out-of-core computations.
- **Stochastic gradient descent**: a single instance is used to compute the gradient. May require learning rate schedule for good convergence. Scale well to huge datasets and supports out-of-core computations.

!!! note annotate "Computational Complexity"

    The complexity of gradient descent is $O(m \times n)$, which is drastically lower than that of a [closed-form solution](#closed-form-solution).

??? example annotate "Example: Fit Noisy Linear Function"

    Fit the $y=3 \cdot x + 4$ equation given 100 noisy samples.

    Batch and mini-batch gradient descent implementations are not directly available in scikit-learn. However, it's easy to implement:

    ```python
    import numpy as np
    
    np.random.seed(42)
    size = 100
    
    noise = 0.25 * np.random.randn(size, 1)
    x1 = np.random.rand(size, 1)
    x2 = np.ones_like(x1)  # (1)
    X = np.c_[x1, x2]
    y = 3 * x1 + 4 + noise

    eta = 0.1
    max_iter = 1000
    m, n = X.shape
    
    theta = np.random.randn(n, 1)
    
    for _ in range(max_iter):
        grad = 2 / m * np.dot(X.T, np.dot(X, theta) - y)
        theta -= eta * grad

    print(f"The equation is: y={theta[0, 0]:.2f} * x + {theta[1, 0]:.2f}.")
    ```

    Stochastic gradient descent is available directly:
    
    ``` py
    import numpy as np
    from sklearn.linear_model import SGDRegressor

    np.random.seed(42)
    size = 100
    
    noise = 0.25 * np.random.randn(size, 1)
    X = np.random.rand(size, 1)
    y = 3 * X + 4 + noise
    
    model = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
    model.fit(X, y)
    print(f"The equation is: y={model.coef_[0]:.2f} * x + {model.intercept_[0]:.2f}.")
    ```

Mini-batch gradient descent is not available in scikit-learn.

## Other types of regression

### Polynomial Regression

Linear models, such as [`LinearRegression` :octicons-link-external-16:][linear-regression] and [`SGDRegressor` :octicons-link-external-16:][sgd-regressor], can be extended to higher powers of input features with [`PolynomialFeatures` :octicons-link-external-16:][polynomial-features]. It returns all possible permutations of a given degree, for example, $x_1^2$, $x_1 x_2$, and $x_2^2$ for two features $x_1$, $x_2$ and a degree of 2.

### Spline Regression

Spline regression can produce even more sophisticated relations by using a series of polynomial segments joined at specified knots. It is available with [`SplineTransformer` :octicons-link-external-16:][spline-transformer]. The resulting segments can then be fit into regression model of one’s choice

!!! example

    See the [example :octicons-link-external-16:][spline-example] of using polynomial regression with spline interpolation in `scikit-learn` documentation.

### Generalized Linear Models

The process of specifying knots in splines can be automated using **Generalized Additive Models** (GAM) regression. GAMs are available in the [`pygam` :octicons-link-external-16:][pygam] package.

!!! example

    See the [example :octicons-link-external-16:][pygam-example] of GAM regression in `pygam` documentation.

[linear-regression]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
[sgd-regressor]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
[polynomial-features]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
[spline-transformer]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html
[spline-example]: https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html#sphx-glr-auto-examples-linear-model-plot-polynomial-interpolation-py
[pygam]: https://pygam.readthedocs.io/en/latest/
[pygam-example]: https://pygam.readthedocs.io/en/latest/notebooks/quick_start.html#Fit-a-Model
