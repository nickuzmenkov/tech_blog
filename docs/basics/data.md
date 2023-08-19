# Data and Sampling

## Central Limit Theorem
Means of samples drawn from a population will be normally distributed around the population mean, even if the population itself is not normally distributed. The sample standard deviation will be roughly the same as that of the population.

### Standard error
*Standard error* is standard deviation of the normal distribution of sample means:

$$
\text{SE} = \frac{\sigma}{\sqrt{n}}
$$

### Standard error on difference
*Standard error on differrence* in means:

$$
\text{SE (diff)}=\sqrt{\frac{\sigma_x^2}{n_x} + \frac{\sigma_y^2}{n_y}}
$$

It is greater than one for each sample but lower than their sum. This corresponds to larger uncertainty when drawing several samples from the population.

## Commong Sampling Bias Types
### Selection bias
It’s usually hard to draw a proper random sample from a population. Assessment of non-random sample characteristics can lead to dramatically incorrect conclusions.

> 💡 #story The *Literary Digest* magazine surveyed more than 10 million of its subscribers, as well as car and phone owners (those contacts were publicly available) all over the country to predict the winner of the presidential election in 1936. Results had proposed Landon’s win with 57% votes. Instead, Roosevelt won with 60% votes. The poll was biased towards wealthier people (those who have magazine subscriptions, cars, and phones), who hence were more likely to vote Republican.

### Publication bias
Assuming there are a lot of repeated studies all over the world each having multiple predictor variables vast search effect, some may find statistically significant patterns just by chance. Those reporting positive findings and/or correlations are *more likely* to be published.

### Recall bias
People tend to create false memories based on their thoughts in present. This is why longitudinal studies are preferred to do cross-sectional studies.

### Survivorship bias
For instance, average class test scores typically rise from freshman to senior year, as the least scoring students are being consistently kicked off.

### Healthy user bias
For instance, people who take vitamins regularly are healthier at least because they tend to care about themselves.

## Bootstrap
*Bootstrap* is a process of replicating the original sample to form a hypothetical population to estimate standard errors and confidence intervals for the sample’s characteristics.

The same can be done by sampling with replacement (#code `sklearn.resample`) instead of replicating.

Confidence intervals can be assessed numerically by drawing the characteristic of interest from $R$ random resamples of the original sample and getting level $p$ and $(1-p)$ quantiles. 

Bootstrap is also widely used for training model ensembles, e.g. each model is trained with bootstrap samples, then predictions averaged for inference, producing *bagging* (bootstrap aggregating) or _pasting_ ensemble.

> 💡 #warning Bootstrap is not used to compensate for a small sample size, create new data, or impute the existing data. It merely gives the idea about samples' statistics drawn from population like the original sample


## Common Distributions
### Normal Distribution
> 💡 Assess how the sample is close to the normal distribution via #code `scipy.stats.probplot`.

### Student’s t-distribution
A family of distributions resembling the normal distribution, but with thicker tails. It’s widely used for sample means, regression parameters, and the like.

$$
\overline{x} \pm t_{n-1}(p) \cdot \frac{s}{\sqrt{n}}
$$

### Binomial distribution
The frequency distribution of the number of successes ($x$) in a given number of trials ($n$) with specified probability of success in each trial ($p$).
    
Distribution function

$$
f(x, n, p) = 
\begin{pmatrix}
    n \\
    x \\
\end{pmatrix}
p^x (1 - p)^{n-x} = \frac{n!}{x!(n-x)!}p^x (1 - p)^{n-x}
$$

Mean

$$
\overline{x}=n \cdot p
$$

Variance

$$
\sigma^2=n\cdot p \cdot (1-p)
$$

💡 For a large number of trials and $p$ close to 0.5 binomial distribution can be approximated by the normal distribution.

#code
- Probability of *exactly* $x$ successes in $n$ trials is given by `scipy.stats.binom.pmf(x=x, n=n, p=0.5)`
- Probability of $x$ *or fewer* successes in $n$ trials `scipy.stats.binom.cdf(x=x, n=n, p=0.5)`
  
### Chi-squared distribution
#empty 

### F-distribution
#empty 

### Poisson Distribution
The frequency distribution of the number of events in sampled units of time or space.

Distribution function

$$
f(k)=\exp{(-\mu)} \frac{\mu^k}{k!}
$$

Mean

$$
\overline{x} = \mu
$$

Variance

$$
\sigma^2 = \mu
$$

#code `scipy.stats.poisson.rvs(mu=2, size=100)`.
    
### Exponential Distribution
The frequency distribution of the time or distance from one event to the next event.
    
Distribution function

$$
f(x)=\exp{(-x)}=\lambda\cdot \exp{(-\lambda \cdot x)}
$$

> 💡 In Poisson and exponential distributions, lambda must stay the same, which is usually not the case in real life. However, if the period over which lambda changes is much longer than the typical interval between events, one can divide time or space into chunks where lambda is roughly constant. When this is not the case, one must switch to Weibull distribution.

#code `scipy.stats.expon.rvs(scale=0.2, size=100)`.

### Weibull Distribution
A generalized version of exponential distribution allows event rates to change over time.
 
#code `scipy.stats.weibull_min.rvs(c=1.5, scale=5000, size=100)`.
