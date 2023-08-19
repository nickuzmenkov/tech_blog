# Statistical Tests

## A/B Testing
#empty Add here a detailed description from the “Naked Statistics”.

## Statistical Significance and P-Values
Significance tests are eliminating two major problems:
-   Humans tend to underestimate the scope of natural random effects ("black swan" effect)
-   Humans tend to misinterpret random events as significant patterns

### P-Value
Given a chance that the null hypothesis is true, the _p-value_ is the probability of results obtained.

### Significance Level
_Significance level_ $\alpha$ is the upper bound for the likelihood of observing (at least as extreme) pattern of data if the null hypothesis is true.

>💡 The correct way to use these notions is, e.g.: “Reject the null hypothesis at the .05 level".

### Type I Error
Type I error (false positive): concluding effect is real when it's a product of chance.

### Type II Error
Type II error (false negative): concluding effect is a product of chance when it's real.

## Permutation Tests
Permutation tests are more of a hands-on way to estimate whether the observed difference of the value in an A/B test is a product of chance.

This algorithm is not available in any of python libraries, yet it’s easy to implement:

1.  Combine the results from all treatment groups
2.  Draw new samples and record new values of interest
3.  Repeat step 2 $R$ times
4.  Assess results distribution (via histogram, bar, etc.)

There are other variations as well:

-   An exhaustive permutation test (i.e. drawing conclusions from all possible permutations)
-   Bootstrap permutation tests (with replacement)

> 💡 T-test is a statistic-based replacement for permutation test: #code `scipy.stats.ttest_ind(x, y, equal_var=False, permutations=1)`

## Vast Search Effect and Multiple Testing
The more predictor variables you test, the higher the chance of type I error. This is called the _vast search effect_. E.g. probability of observing one or more type I error when assessing 20 predictor variables at .05 significance level:

$$ P=1 - 0.95^{20} = 1 - 0.36=64 \% $$

The same is true for multiple statistical comparisons (e.g. is A different from B, C, D, etc.?) and multiple models or, in general, _multiple testing_.

> 💡 This can be summarized in the quote: “_If you torture the data long enough, sooner or later it will confess_”.

In order to minimize false discovery rate, different techniques can be applied:

-   adjustment of the p-value (typically divided by the number of comparisons or models)
-   hold-out set (typically for regression tasks)

### ANOVA
_Analysis of variance_ (ANOVA) is a statistical test to measure the significance of differences among multiple treatment groups. The value of interest measured must be numeric.

ANOVA is used when we are interested if the overall variance between _all_ the treatment groups is a product of chance. That is, no attention is paid to variance between individual treatment groups.

This algorithm is not available in any of python libraries, yet it’s easy to implement. It differs from the permutation test just by one additional step:

1.  Combine all the data together
2.  Draw new samples and record new values of interest
3.  Record the variance between results for all the groups
4.  Repeat the steps (2-3) $R$ times
5.  The p-value is the fraction of results exceeding the original variance.

>💡 A statistic-based replacement for ANOVA is to f-statistic.

## Pearson’s Chi-Squared Test
Chi-square test is designed to test the observed distribution of categorical data against the expected (usually uniform) distribution. It is widely used for independence tests.

This test requires data in a form of $r \times c$ _contingency table_, i.e. table of frequency distribution of the variables. Degrees of freedom can be calculated like:

$$ \text{dof} = (r - 1)(c - 1) $$

The chi-square statistic is then computed as follows:

$$ \chi^2 = \sum_i^r \sum_j^c \frac{(O - E)^2}{E} $$

where $O$ is the observed value, and $E$ is the expected value.

>💡 #warning When value counts are extremely low (the rule of thumb is five or fewer), it’s best to use permutation test or exhaustive permutation test—_Fisher’s exact test_.

## Multi-Arm Bandit Algorithms
Unlike classic [[#A B Testing|A/B tests]], _multi-arm bandit_ _algorithms_ allow adjusting size of treatment groups on the fly to maximize outcomes.

Typically used in web testing and other applications where the main point is not to prove that the distinctions between the treatments are statistically significant but to maximize the outcomes (e.g. conversion rates).

One of possible algorithms may be like this:

-   Randomly assign a new specimen to any of the treatment groups with probability of $\varepsilon$
-   Assign a new specimen to the highest scoring treatment group with probability of $(1-\varepsilon)$
-   Adjust $\varepsilon$ to the results accordingly

> 💡 When $\varepsilon=0$ the algorithm becomes a classic A/B test, and when $\varepsilon=1$ the algorithm becomes _greedy_ (i.e. always choosing the best option based on fixed number of previous scores).

Another approach is called _Thompson’s sampling_. It uses Bayesian theorem to maximize the probability of choosing the best treatment for each specimen.

## Power and Sample Size
There are four dependent characteristics in a test:

-   Sample size
-   Effect size
-   Power
-   Significance level

Typically the characteristic of interest is the sample size.

### Effect size
_Effect size_ is the minimum difference of the value of interest between two treatment groups to be proved as statistically significant.

### Power
_Power_ is the probability of detecting a given effect size within a sample of given size and variability.


## Examples
### Simple Statistical Significance Test #example 
Do printers have higher diastolic blood pressure levels than farmers?

| Occupation|Observations|Mean blood pressure (mmHg) | Std (mmHg) |
| ---------- | ------------ | -------------------------- | ---------- |
| Printer    | 72           | 86                         | 8.5        |
| Farmer     | 48           | 82                         | 8.2        |

#### Solution
Set up the null hypothesis and the alternate hypothesis. _Null hypothesis_: there is no difference in diastolic blood pressure levels between printers and farmers. _Alternate hypothesis_: printers and farmers have different blood pressure.

Set up significance level ($\alpha$) of .05 for rejecting the null hypothesis.

According to [[Data and Sampling#Standard error on difference|this formula]] standard error for difference in means is:

$$ \text{SE (diff)} = \sqrt{\frac{8.5^2}{72} + \frac{8.2^2}{48}} = 1.55 $$

The observed difference in means in SE scale is:

$$ z = \frac{86 - 82}{1.55} = 2.58 $$

The p-value is given by `2 * (1 - scipy.stats.norm.cdf(2.58))`: 0.01.

Thus the null hypothesis can be rejected at .05 level.

### Permutation Test #example
Does any of the headlines A, B, or C really attract readers the most?

| Action   | Headline A|Headline B|Headline C|Total |
| -------- | ---------- | ---------- | ---------- | ----- |
| Click    | 14         | 8          | 12         | 34    |
| No click | 986        | 992        | 988        | 2,966 |
| Total    | 1,000      | 1,000      | 1,000      | 3,000 |

#### Solution
The null hypothesis is that actions are uniformly distributed. If so

| Action   | Headline A|Headline B|Headline C|Total |
| -------- | ---------- | ---------- | ---------- | ----- |
| Click    | 11.33      | 11.33      | 11.33      | 34    |
| No click | 988.67     | 988.67     | 988.67     | 2,966 |
| Total    | 1,000      | 1,000      | 1,000      | 3,000 |

Pearson's residual for each row

| Action   | Headline A|Headline B|Headline C|Total |
| -------- | ---------- | ---------- | ---------- | ----- |
| Click    | 0.792      | -0.990     | 0.198      | 0     |
| No click | -0.085     | 0.106      | -0.021     | 0     |
| Total    | 0          | 0          | 0          | 0     | 

Compute the chi-squared statistic for the entire table

$$
\chi = \sum_i^r \sum_j^c R^2=1.67
$$

Make a permutation set of 34 positives (clicks) and 2,966 negatives (no clicks).
Apply the steps above to random $R$ random permutations and draw the $\chi$ values.
The p-value is the fraction of results exceeding the original $\chi$ value. In this case:

$$ p=0.48 $$

Thus the null hypothesis _cannot be rejected_ at .05 level.

### Pearson’s Chi-Squared Test #example 1
The problem statement is the same as for [[#Permutation Test example| headline example]] above.

#### Solution
```python
>>> from scipy.stats import chi2_contingency
>>> observed = [
>>>     [14, 8, 12, 34], 
>>>     [986, 992, 988, 2966],
>>>     [1000, 1000, 1000, 3000],
>>> ]
>>> chisq, pvalue, dof, expected = chi2_contingency(observed)
>>> round(chisq, 2)
1.67
>>> round(pvalue, 2)
0.43
>>> dof
2
>>> expected
array([[  11.33333333,   11.33333333,   11.33333333,   34.        ],
       [ 988.66666667,  988.66666667,  988.66666667, 2966.        ],
       [1000.        , 1000.        , 1000.        , 3000.        ]])
```


### Pearson’s Chi-Squared Test #example 2
Did the researcher fabficate her results? 
Here’s 315 interior digits drawn from the paper, (i.e. excluding the first and the last digits, which are usually not random):

| Digit     | 0   | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   |
| --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frequency | 14  | 71  | 7   | 65  | 23  | 19  | 12  | 45  | 53  | 6   | 

BTW, this is a real #story.

#### Solution
Hence chi-squared test requires data to be at least 2 dimensional with at least 2 rows and 2 columns, we need to reformat the initial table

| Digit       | 0   | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | Total |
| ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- |
| Present     | 14  | 71  | 7   | 65  | 23  | 19  | 12  | 45  | 53  | 6   | 315   |
| Not present | 301 | 244 | 308 | 250 | 292 | 296 | 303 | 270 | 262 | 309 | 2,835 |
| Total       | 315 | 315 | 315 | 315 | 315 | 315 | 315 | 315 | 315 | 315 | 3,150 | 

Apply a chi-squared test for this contingency table

```python
>>> from scipy.stats import chi2_contingency
>>> observed = [
>>>     [14, 71, 7, 65, 23, 19, 12, 45, 53, 6, 315],
>>>     [301, 244, 308, 250, 292, 296, 303, 270, 262, 309, 2835],
>>>     [315, 315, 315, 315, 315, 315, 315, 315, 315, 315, 3150],
>>> ]
>>> chisq, pvalue, dof, expected = chi2_contingency(observed)
>>> f"{p:.2e}"
'1.94e-30'
```

### Sample Size Estimation #example
Set up an A/B test for adds. The current add (the control group) has conversion rate of 1.1%. It’s supposed that a new add should be at least 10% better than the old one (i.e. at least 1.21%). Given that, how many samples must be taken to observe statistically significant difference at least 80% of the time?

#### Solution
This can be done via `statsmodels` package:

```python
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import TTestIndPower

effect_size = proportion_effectsize(0.0121, 0.011)
analysis = TTestIndPower()
result = analysis.solve_power(
    effect_size=effect_size,
    alpha=0.05,
    power=0.8,
    alternative='larger',
)
print(round(result))
```

The result is 116602.