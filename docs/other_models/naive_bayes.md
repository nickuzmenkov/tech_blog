# Naive Bayes

#code `sklearn.naive_bayes.MultinomialNB`.

## Bayes’ Theorem
The theorem is stated mathematically as follows

$$ P(A|B)=\frac{P(A)P(B|A)}{P(B)} $$

where $P(A|B)$ is a _conditional probability_, i.e. probability of event $A$ given that event $B$ happened.


## Naive Bayes Algorithm
Applying the Bayes’ theorem to a dataset, the value of interest is $P(Y=i|X_1...X_p)$, i.e. the probability of $Y$ being either 0 or 1 ($i$) given the set of predictor variables, can be calculated as follows:

$$ P(Y=i|X_1...X_p)=\frac{P(Y=i)P(X_1...X_p|Y=i)}{P(Y=0)P(X_1...X_p|Y=0) + P(Y=1)P(X_1...X_p|Y=1)} $$

>💡 Given even a relatively small number of predictors—say, five—it would require an overwhelmingly large dataset to find at least a couple of records with the same state of predictors $X_1...X_p$ to get conditional probabilities $P(X_1...X_p|Y=i)$.

Under the assumption of conditional independence among predictor variables (_naive assumption_) all the mutual probabilities can be factorized:

$$ P(X_1...X_p|Y=i)=\prod_{j=1}^p{P(X_j|Y=i)} $$

Changing the above equation into:

$$ P(Y=i|X_1...X_p)=\frac{P(Y=i)\prod_{j=1}^p{P(X_j|Y=i)}}{P(Y=0)\prod_{j=1}^p{P(X_j|Y=0)} + P(Y=1)\prod_{j=1}^p{P(X_j|Y=1)}} $$


### Supported Predictors
The Bayesian classifier works only with categorical predictors. To apply naive Bayes to numerical predictors, one of two approaches must be taken:

-   Bin and convert the numerical predictors to categorical predictors
-   Use a probability model (e.g. normal distribution) for estimating the conditional probabilities $P(X_j|Y=i)$

>💡 #warning Categorical variables for naive Bayes must be one-hot encoded without omitting.

## Examples
### Bayes Theorem #example
There’s a rare subspecies of a beetle, 0.1% of the total population. 98% of rare subspecies have a special pattern on the back, compared to only 5% members of the common subspecies. How likely is the beetle having the pattern to be rare?

#### Solution

Given that:

$$ P(\text{Rare})=0.001 \\ P(\text{Common})=0.999 \\ P(\text{Pattern|Rare})=0.98 \\ P(\text{Pattern|Common })=0.05 $$

According to the Bayes’ theorem:

$$ P(\text{Rare|Pattern}) = \frac{P(\text{Rare})P(\text{Pattern|Rare})}{P(\text{Pattern})} $$

Expanding the $P(\text{Pattern})$:

$$ P(\text{Rare|Pattern}) = \frac{P(\text{Rare})P(\text{Pattern|Rare})}{P(\text{Common})P(\text{Common|Pattern}) + P(\text{Rare})P(\text{Rare|Pattern})} $$

Substitute the values:

$$ P(\text{Rare|Pattern}) = \frac{0.001 \cdot 0.98}{0.999\cdot 0.05 + 0.001\cdot 0.98} = 0.019 $$

That is, 1.9%.
