# Introduction  

- $n$ is the number of features plus bias term
-   $m$ is the number of instances
-   $*^{(i)}$ are instance’s indices
-   $\boldsymbol{x}$ is the instance’s feature vector (with $x_0$ always equal to $1$)
-   $\boldsymbol{X}$ is the $(m \times n)$ training matrix
-   $\boldsymbol{w}$ is the models’ parameter (weight) vector, containing feature weights $w_i$
-   $\boldsymbol{\theta}$ is the model’s parameter vector, containing the bias (intercept) term $\theta_0$ and $(n-1)$ feature weights $w_i$
-   $\boldsymbol{y}$ is the vector of $m$ target values $y^{(i)}$
-   $\|*\|_2$ is $L^2$ norm


### Expected Value
For a random variable with $n$ possible outcomes $x_i$ with associated probabilities $p_i$ the expected value $E[X]$ is given by

$$
E[X] = \sum_{i=1}^n{x_ip_i}
$$
For a random variable with infinitely many outcomes and probability density function $f$ the expected value is given by

$$
E[X] = \int_{-\infty}^{\infty}{xf(x)\ dx}
$$

### Central Moment
The _Central Moment_ (or _moment about the mean_) of degree $k$ is given by

$$
\mu_k = E[(X-E[X])^k] = \int_{-\infty}^{\infty}{(x-\mu)^kf(x)\ dx}
$$

where $E[X]$ is the [[#Expected Value|Expected Value]].

Obviously, the zeroth central moment $\mu_0$ is equal to $1$, the first central moment $\mu_1$ is equal to $0$, and the second central moment is equal to variance $\sigma^2$. 

### Standardized Moment
The _Standardized Moment_ of degree $k$ is given by

$$
\tilde{\mu}_k = \frac{\mu_k}{\sigma^k} 
$$

### Skewness
_Skewness_ is the third [[#Standardized Moment|Standardized Moment]] and is given by

$$
\tilde{\mu}_3 = \frac{E[(X-E[X])^3]}{\sigma^3}
$$

Zero skewness indicates perfectly symmetric distribution, while positive skewness indicates higher density of smaller values and vice versa.

![[assets/img/Statistics/Notation/01.png|400]]

#code `pandas.Series.skew`.

### Kurtosis
_Kurtosis_ is the fourth [[#Standardized Moment|Standardized Moment]] and is given by

$$
\tilde{\mu}_4 = \frac{E[(X-E[X])^4]}{\sigma^4}
$$

Kurtosis determines the plateau width of the distribution. The higher the kurtosis the narrower the plateau and vice versa.

![[assets/img/Statistics/Notation/02.png|400]]

#code `pandas.Series.kurt`.

### CDF
Cumulative Distribution Function

### PMF
Probability Mass Function

### RVs
Random Variables


### Z-Score
The process of standardizing is essentially subtracting the mean and dividing it by standard deviation. This value is also called a *z-score*.


### Rectangular Data
Rectangular data is essentially a table of any relational database.


### Degrees of Freedom
In a nutshell, this is just the sample size (number of observations) minus number of explanatory variables.


### Frequency Table
Frequency table is essentially just the histogram represented as a table.


### Contingency Table
_Contingency table_ is a table of frequency distribution of the variables typically with an extra row and column representing the totals. For #example:

| Action   | Headline A|Headline B|Headline C|Total |
| -------- | ---------- | ---------- | ---------- | ----- |
| Click    | 14         | 8          | 12         | 34    |
| No click | 986        | 992        | 988        | 2,966 |
| Total    | 1,000      | 1,000      | 1,000      | 3,000 |

### Norm
Norms are functions mapping vectors to non-negative values. The $L^p$ norm is given by

$$ \|x\|_p=\left(\sum_{i}{|x_i|^p}\right)^{1/p} $$

for $p \in \mathbb{R}, p \geq 1$.

### Manhattan Distance
Manhattan distance is given by

$$
\text{Distance}=\sqrt{\sum_{i=1}^n{(x_i-y_i)^2}}
$$

### Euclidean Distance
Euclidean distance is given by

$$
\text{Distance}=\sum_{i=1}^n|x_i-y_i|^2
$$

### Bias
_Bias_ is the part of generalization error due to wrong assumptions (e.g. data linearity).

### Variance
_Variance_ is the part of generalization error due to the model’s excessive sensitivity to small variations in data (e.g. high-degree polynomials).

### White Box Model
_White Box Model_ is easy to interpret and draw insights from given model’s parameters.

### Black Box Model
Unlike [[#White Box Model]], _Black Box Moel_ does not provide intuitive results.
