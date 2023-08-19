# Regression

## Parameters Confidence Intervals
Confidence intervals for regression coefficients:

1.  Draw $2n$ records from the original dataset (with replacement)
2.  Fit regression model and record the coefficients
3.  Repeat steps 1-2 $R$ times
4.  For $(1-2p)$ confidence interval, get $p$ and $(1-p)$ level quantiles.

## Stepwise Regression
_Stepwise regression_ is the process of searching for the optimal set of predictor variables.

> 💡 #story It’s common to use the principle of _Occam’s razor_ when building a regression model: a simpler model (i.e. with fewer predictor variables) should be used in preference to a more complicated model, ceteris paribus.

### Vanilla Predictor Selection
Adding and dropping predictor variables based on adjusted R-squared or AIC metrics and statistical significance as you go.

### All Subset Regression
_All subset regression_: searching through each possible combination of predictor variables. Computationally expensive and most prone to overfitting.

### Forward Selection
 _Forward selection_: start with no predictors at all (constant model) and add them one by one _greedily_ (i.e. those having the largest contribution to the metric). The process stops when the contribution is no longer statistically significant.
 
### Backward Selection
_Backward regression_ (_backward elimination_): start with all available predictor variables (i.e. the full model) and take away those which aren’t statistically significant until all the predictor variables are statistically significant.

### Penalized Regression
An alternate approach to stepwise regression is penalized regression: instead of eliminating predictors, it applies the $\ell_1$ penalty (_lasso regression_) or $\ell_2$ penalty (_ridge regression_) thus reducing coefficients of insignificant or highly correlated predictors.

## Categorical Variables in Regression

Categorical variables are also called _factor variables_. Binary variables are also called _indicator variables_.

Factor variables are typically translated to _dummy variables_ or _one-hot encoded_ before fitting into the regression model. This is done via `pandas.get_dummies(series, drop_first=True)`.

>💡 One of the categories must be omitted to avoid multicollinearity, i.e. fitting a model with dependent variables. E.g. if factor variable $x$ can take ($p$) values $A$, $B$, or $C$, then if both $x_A$ and $x_B$ are false, $x_C$ must be true. Thus, only $x_A$ and $x_B$ (i.e. $p-1$, which is degrees of freedom) dummy variables should be kept.

Factor variables with _too many_ categories can be translated to numeric variables by mapping them to mean/median of the target (e.g. city id can be translated to median city house price) or residual from the target given by model ignoring this variable. Similarly, we can reduce the number of categories by grouping the obtained numeric values, producing new categories. This can be done via `pandas.qcut(series, q=4, labels=["s", "m", "l", "xl"])`.

Ordinal variables (or _ordered factor variables_) can be treated as numeric variables (or assigned to those according to order) in most cases. This helps to preserve the order.

## Interpreting the Regression Equation

In statistics, regression is mainly a tool for explanatory modeling, not for prediction. This can be of value for Data Science applications too.

### Correlated predictors
_Correlated predictors_ can skew the coefficients to nonsense (i.e. total space, number of living rooms, and number of bathrooms)

### Multicollinearity
_Multicollinearity_ is the extreme case for correlated variables. It is usually the case when the model is fitted with the same predictor added multiple times by mistake, $p$ instead of $(p-1)$ dummy variables, or nearly perfectly correlated predictors.

### Confounding variables
_Confounding_ variables are mistakenly omitted variables of high significance. Ignoring them introduces randomness in coefficients and leads to incorrect conclusions.

### Interactions
_Interactions_ are combinations (usually products) of predictor variables (or _main effects_). Adding meaningful interactions (e.g. house total space and price region) to the model can significantly improve scores and explainability.

>💡 Searching for the proper interaction of predictors can be challenging. Interactions are selected either based on prior experience or via stepwise regression approaches. Another way to account for interactions is switching to a non-linear model, e.g. decision trees, random forests, or XGBoost. Those search for optimal interactions automatically.

## Regression Diagnostics
### Outliers in Residuals
A general approach to detect outliers is based on boxplot or z-score (i.e. residual divided by the standard error of all residuals). Outliers of residuals can help spot anomalies in data (e.g. units mismatch, misspellings, column swap, etc.), including fraud.

### Influential values
Influential values have high _leverage_ on regression, i.e. excluding these records would significantly change regression coefficients.

![[assets/img/Statistics/Regression/01.png|300]]

There are a few metrics to determine the influence of a single record:

#### Hat Values
Given that $\hat{Y}=HY$, where $H$ is the hat matrix, diagonal values (_hat-values_) larger than $2(p+1)/n$ indicate high leverage records.

#### Cook's Distance
 _Cook’s distance_ larger than $4/(n-p-1)$ indicates high leverage records.

#### Bubble Plot
A _bubble plot_ is a scatter plot of hat-values versus residual z-scores with dot size equal to Cook’s distance.

![[assets/img/Statistics/Regression/02.png|300]]

#sourcecode [[Regression Code#Bubble Plot]].

### Heteroskedasticity
_Heteroskedasticity_ is a difference in residual variance across the range of predicted values, i.e. variance of the residual depends on the predicted value. It can be assessed visually from a scatter plot of predictions versus residuals with spline smoothing

![[assets/img/Statistics/Regression/03.png|400]]

#sourcecode [[Regression Code#Heteroskedacticity Plot]].

### Residuals Distribution
The distribution of residuals, which is a subject of interest for statisticians exclusively, can actually tell much about the model quality. That is, normally distributed residuals indicate that the model is complete, whereas the opposite is a clear sign that the model is missing something.

### Mean Squared Error (MSE)
MSE is given by

$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^m{
	\left(
		\boldsymbol{y}^{(i)}-\boldsymbol{\hat{y}}^{(i)}
	\right)^2
	}
$$

### Root Mean Squared Error (RMSE)
RMSE is given by

$$
\text{RMSE} = \sqrt{
	\frac{1}{m} \sum_{i=1}^m{
		\left(
			\boldsymbol{y}^{(i)}-\boldsymbol{\hat{y}}^{(i)}
		\right)^2
	}
}
$$

### Residual Standard Error (RSE)
RSE is given by

$$
\text{RSE} = \sqrt{
	\frac{1}{m - n - 1} \sum_{i=1}^m{
		\left(
			\boldsymbol{y}^{(i)}-\boldsymbol{\hat{y}}^{(i)}
		\right)^2
	}
}
$$

the only difference is that the denominator is the degrees of freedom (i.e. number of records minus number of predictor variables minus one) instead of number of records.

### Coefficient of determination (R-squared)
R-squared is given by

$$ 
R^2 = 1 - \frac{\sum_{i=1}^m{
		\left(
			\boldsymbol{y}^{(i)}-\boldsymbol{\hat{y}}^{(i)}
		\right)^2
		}
	}
	{\sum_{i=1}^m{
		\left(
			\boldsymbol{y}^{(i)}-\boldsymbol{\overline{y}}^{(i)}
		\right)^2
	}
}
$$

The denominator is proportional to the target variance. Hence R-squared indicates the fraction of target variance accounted in the model.

### Adjusted R-squared
Adjusted R-squared penalizes the model with too many predictor variables:

$$
R_{adj}^2 = 1 - \left( 1 - R^2 \right) \frac{m - 1}{m - n - 1}
$$

### Akaike’s information criterion (AIC)
AIC takes into account model’s complexity

$$
\text{AIC} = 2n + m \log{(\text{MSE})}
$$

so that $n$ more predictor variables would be penalized by $2n$.

### T-Statistic
The t-statistic is calculated for each model's parameter 

$$
t_b=\frac{
		\boldsymbol{\theta}^{(i)}
	}
	{\text{SE}\left(\boldsymbol{\theta}^{(i)}\right)}
$$

where $\text{SE}$ is [[base/Statistics/Data and Sampling#Standard Error|Standard Error]] of parameter. It can be assessed similarly to [[base/Statistics/Regression#Parameters Confidence Intervals| parameters confidence intervals]] by [[Data and Sampling#Bootstrap|bootstrapping]] the data and refitting the model.

Since t-statistic is the mirror image of the p-value, the higher the value of t-statistic, the more statistically significant the predictor variable is.