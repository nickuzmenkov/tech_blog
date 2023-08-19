# Support Vector Machines

## Classification
Similarly to [[Linear Models#Logistic Regession|Logistic Regression]], the score of SVM classifier is a dot product of instance’s feature vector and the model’s weight vector plus intercept

$$
\text{Decision function}=\boldsymbol{w}^T\boldsymbol{x}+b
$$

Yet no function is applied to it, so there’s no “probability” score in SVM. The prediction is drawn straight from the decision function

$$
\hat{y}= 
\begin{cases} 
	0 & \text{if} & \boldsymbol{w}^T\boldsymbol{x}+b<0,
	\\ 
	1 & \text{if} & \boldsymbol{w}^T\boldsymbol{x}+b\ge0
\end{cases}
$$

💡Unlike most `sklearn` models, SVC doesn’t have the `predict_proba` method since there’s no _probability_ sense assigned to the decision function.

Under the hood SVM tries to get the widest possible margin between the two classes.

Instances outside the margin do not affect the decision boundary while instances at the margin edge determine the boundary. Those are called _support vectors_.

For data with only a single feature the train objective can be visualized as finding the the least possible slope so that the decision function is greater than $1$ for all positive instances and less than $-1$ for all negative instances:

![[assets/img/Machine Learning/Support Vector Machines/01.png|600]]

#sourcecode [[Support Vector Machines Code#Widest Margin Search]].

Considering whether instances are allowed to violate the $[-1;1]$ margin or not there’s 2 main approaches:

-   _hard margin classification_: instances are not allowed to violate the margin. Only applicable for _linearly separable_ classes.
-   _soft_ _margin_ _classification_: instances are allowed to violate the margin. Number of violations is minimized along with maximizing the margin width.

Hard margin linear SVM classifier objective can be expressed as a constrained optimization problem

$$ \begin{aligned} \underset{\boldsymbol{w},b}{\text{minimize}} \quad & \frac{1}{2}\|\boldsymbol{w}\|_2^2 \\ \text{subject to} \quad & t^{(i)}(\boldsymbol{w}^T\boldsymbol{x}^{(i)}+b) \ge 1,\ i=\overline{1,m} \end{aligned} $$

where

$$ t^{(i)}= \begin{cases} -1 & \text{if} & y^{(i)}=0, \\ +1 & \text{if} & y^{(i)}=1, \\ \end{cases} $$

Similarly, soft margin linear SVM classifier objective can be expressed as

$$ \begin{aligned} \underset{\boldsymbol{w},b,\zeta}{\text{minimize}} \quad & \frac{1}{2}\|\boldsymbol{w}\|_2^2 + C\sum_{i=1}^m{\zeta^{(i)}}\\ \text{subject to} \quad & t^{(i)}(\boldsymbol{w}^T\boldsymbol{x}^{(i)}+b) \ge 1 - \zeta^{(i)},\ i=\overline{1,m} \end{aligned} $$

where $\zeta^{(i)}\ge0$ is a _slack variable_, $C$ is a regularization hyperparameter (inverse proportional to the $\ell_2$ penalty strength).

SGD formulation can be expressed as minimizing the _hinge cost function_ plus $\ell_2$ penalty:

$$ J(\boldsymbol{w},b)=C\sum_{i=1}^m\max{\left(0,1-t^{(i)}\left(\boldsymbol{w}^T\boldsymbol{x}^{(i)}+b\right)\right)} + \frac{1}{2}\boldsymbol{w}^T\boldsymbol{w} $$

#code
-   `sklearn.svm.LinearSVC` (fastest for linear tasks)
-   `sklearn.svm.SVC(kernel="linear")` (supports kernel trick, slower on linear tasks than `LinearSVC`)
-   `sklearn.linear_model.SGDClassifier(loss="hinge", alpha=1 / m / C)` (the slowest option, yet supports out-of-core training. Recommended for big datasets)


## Regression
Similarly to Linear Regression model, SVM regression equation is given by

$$ \hat{y}=\boldsymbol{w}^T\boldsymbol{x}+b $$

SVM regressor tries to fit as many instances as possible within the specified $\epsilon$ range. Training instances within the range do not affect the model.

#code
-   `sklearn.svm.LinearSVR` (fastest for linear tasks)
-   `sklearn.svm.SVR` (supports kernel trick, slower on linear tasks than `LinearSVR`)
-   `sklearn.linear_model.SGDRegressor(loss="hinge", alpha=1 / m / C)` (supports out-of-core and online training, slowest among all)

## Non-Linear Models
### Polynomial Features
Adding higher degrees and permutations of the predictor variables is the same as for [[Linear Models#Polynomial Regression|polynomial regression]].

### Gaussian Radial Basis Function
Gaussian _Radial Basis Function_ (RBF) is given by

$$ \phi_\gamma(\boldsymbol{x},\ell)=\exp{\left(-\gamma\|\boldsymbol{x}-\ell\|^2\right)} $$

where $\ell$ is the landmark, $\gamma$ (inverse proportional to bell width) is regularization parameter.

You can think of landmark being every instance in the dataset. 

For #example take dummy data with one feature ($x$) with linearly non inseparable classes and replace it with two features ($x_1,\ x_2$) of $\phi_\gamma(\boldsymbol{x},4)\ \phi_\gamma(\boldsymbol{x},7)$, respectively. In the new space classes are linearly separable.

![[assets/img/Machine Learning/Support Vector Machines/03.png|700]]

#sourcecode [[Support Vector Machines Code#Gaussian Radial Basis Function]].

## Quadratic Programming
The above stated optimization problems (i.e. convex quadratic optimization problems with linear constraints) belong to _Quadratic Programming_ (QP) problems. Depending other conditions (i.e. kernel type) either the _primal_ or the dual _problem_ is solved.

### Primal Problem
The _primal problem_ is given by

$$ \begin{aligned} \underset{\boldsymbol{p}}{\text{minimize}} \quad & \frac{1}{2}\boldsymbol{p}^T\boldsymbol{H}\boldsymbol{p} + \boldsymbol{f} ^T\boldsymbol{p}\\ \text{subject to} \quad & \boldsymbol{A}\boldsymbol{p}\le \boldsymbol{b} \end{aligned} $$

### Dual Problem
The _dual problem_ is given by

$$ \begin{aligned}

\underset{\boldsymbol{\alpha}}{\text{minimize}} \quad & \frac{1}{2}\sum_{i=1}^m{\sum_{j=1}^m{\boldsymbol{\alpha}^{(i)}\boldsymbol{\alpha}^{(j)}}\boldsymbol{t}^{(i)}\boldsymbol{t}^{(j)}\boldsymbol{X}^{(i)^T}\boldsymbol{X}^{(j)}}

\\

\text{subject to} \quad & \boldsymbol{\alpha}^{(i)}\ge 0,\ i=\overline{1,m}

\end{aligned} $$

then the model’s weight vector and bias can be expressed through the $\boldsymbol{\alpha}$ vector.

> 💡 $\boldsymbol{\alpha}^{(i)}\ne0$ only for support vectors.

## Kernelized SVM

Let’s start with an example. Let $\phi(\boldsymbol{x})$ be a second-degree polynomial transformation. E.g. for a two-dimensional feature vector $\boldsymbol{x}$

$$ \phi(\boldsymbol{x})=\phi\left( \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} \right) = \begin{pmatrix} x_1^2 \\ \sqrt{2}x_1x_2\\ x_2^2 \end{pmatrix} $$

<aside> 💡 The exact formula of the $\phi(\boldsymbol{x})$ is of no importance here and can be safely omitted.

</aside>

Now let’s calculate the dot product of two arbitrary 2-dimensional vectors $\boldsymbol{a}, \boldsymbol{b}$ transformed by $\phi$

$$ \phi(\boldsymbol{a})^T\phi(\boldsymbol{b})= \begin{pmatrix} a_1^2 \\ \sqrt{2}a_1a_2\\ a_2^2 \end{pmatrix}^T \begin{pmatrix} b_1^2 \\ \sqrt{2}b_1b_2\\ b_2^2 \end{pmatrix} = a_1^2b_1^2+2\sqrt{2}a_1a_2b_1b_2+a_2^2b_2^2=\left(a_1b_1+a_2b_2\right)^2=\left( \begin{pmatrix} a_1 \\ a_2 \end{pmatrix}^T \begin{pmatrix} b_1 \\ b_2 \end{pmatrix} \right)^2 $$

$$ \boxed{ \phi(\boldsymbol{a})^T\phi(\boldsymbol{b})= \left(\boldsymbol{a}^T\boldsymbol{b}\right)^2} $$

i.e. the dot product of transformed vectors equals to the square dot product of the original vectors, known as second-degree _polynomial kernel_.

Similarly, other kernel functions are given by

$$ \begin{aligned} \text{Linear:} \quad & K(\boldsymbol{a},\boldsymbol{b})=\boldsymbol{a}^T\boldsymbol{b} \\ \text{Polynomial:} \quad & K(\boldsymbol{a},\boldsymbol{b})=\left(\gamma\boldsymbol{a}^T\boldsymbol{b} + r\right)^d \\ \text{Gaussian RBF:} \quad & K(\boldsymbol{a},\boldsymbol{b})=\exp\left(-\gamma\left\|\boldsymbol{a} - \boldsymbol{b}\right\|^2\right) \\ \text{Sigmoid:} \quad & K(\boldsymbol{a},\boldsymbol{b})=\tanh{\left(\gamma\boldsymbol{a}^T\boldsymbol{b} + r\right)} \end{aligned} $$

_Kernel_ is a function capable of computing the dot product $\phi(\boldsymbol{a})^T\phi(\boldsymbol{b})$ based only on the original vectors and not the transformation. This allows to just substitute the kernel expression into the [[#Dual Problem| dual problem]] equation

$$ \phi(\boldsymbol{X}^{(i)})^T\phi(\boldsymbol{X}^{(j)}) \rightarrow K(\boldsymbol{X}^{(i)^T}\boldsymbol{X}^{(j)}) $$

without applying the transformation (e.g. in case of polynomial transformation, this won’t cause the combinatorial explosion of the number of features; let alone Gaussian RBF function which maps the original feature vectors into infinite dimensional space).

#complexity
-   `LinearSVR`,`LinearSVC`: $O(m \times n)$
-   `SVR`, `SVC`: $O(m^2 \times n)$ to $O(m^3\times n)$
-   `SGDRegressor`, `SGDClassifier`: $O(m\times n)$

## Examples
### Moons classification #example
Moons synthetic data looks like this

![[assets/img/Machine Learning/Support Vector Machines/02.png|500]]

Code (`LinearSVC` + `PolynomialFeatures`):

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import plotly.graph_objects as go
import numpy as np

X, y = make_moons(n_samples=150, noise=0.15, random_state=42)

model = Pipeline(
	[
		("features", PolynomialFeatures(degree=3)),
		("scale", StandardScaler()),
		("fit", LinearSVC(C=10)),
	]
)
model.fit(X, y)

x1 = np.linspace(X[:, 0].min(), X[:, 0].max())
x2 = np.linspace(X[:, 1].min(), X[:, 1].max())

X_test = np.meshgrid(x1, x2)
X_test = np.c_[X_test[0].reshape((-1,)), X_test[1].reshape((-1,))]

y_test = model.decision_function(X_test)

go.Figure(
	data=(
		go.Scatter(
			x=X[:, 0],
			y=X[:, 1],
			mode="markers",
			marker=dict(color=y, symbol=y, colorscale="Tropic"),
		),
		go.Heatmap(
			z=y_test.reshape(50, 50),
			x=x1,
			y=x2,
			zsmooth="best",
			colorscale="Tropic",
			colorbar=dict(title_text="Decision function"),
		),
		go.Contour(
			z=y_test.reshape(50, 50),
			x=x1,
			y=x2,
			showscale=False,
			colorscale="Greys",
			contours_coloring="lines",
			contours=dict(start=0, end=0),
		),
	),
	layout=dict(
		width=600,
		height=400,
		margin=dict(b=10, t=10, l=10, r=10),
		plot_bgcolor="rgba(0,0,0,0)",
		xaxis=dict(visible=False),
		yaxis=dict(visible=False),
	),
).write_image("newplot.png", scale=2)
```

Similarly, with `SVC` and kernel trick (ceteris paribus):

```python
...

model = Pipeline(
	[
		("scale", StandardScaler()),
		("fit", SVC(kernel="poly", degree=3, coef0=1, C=5)),
	]
)

...
```
