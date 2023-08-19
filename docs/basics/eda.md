# Exploratory Data Analysis

>💡 #story John W. Tukey first introduced EDA as a mandatory step before statistical inference in his paper “The future of Data Analysis” in 1962.

#code for advanced plots
-   Hexagonal binning `pandas.DataFrame.plot.hexbin`
-   Contour plot `seaborn.kdeplot`
-   Violin plot `seaborn.violinplot`

### Data Types

-   Numeric
    -   Continuous
    -   Discrete
-   Categorical
    -   Binary (literally true/false)
    -   Ordinal (e.g. sizes: S, M, L, XL, etc.)

## Central Tendency Estimation

### Mean
### Weighted mean
_Weighted mean_ is given by

$$
\overline{x}_w=\frac{\sum_{i=1}^n w_i x_i}{\sum_{i=1}^n w_i}
$$

### Trimmed mean 
Trimmed mean is mean of a population with $p$ largest and $p$ smallest values omitted

$$
\overline{x}=\frac{\sum_{i=p+1}^{n-p}x_{(i)}}{n - 2p}
$$

### Median
Median is given by

#empty
$$
$$

### Weighted median
Weighted median is given by

#empty
$$
$$

###  Mode
Mode is simply most frequent value. Only applicable for categorical data.

## Variability Metrics
### Mean Absolute Deviation

$$ \text{Mean absolute deviation}=\frac{\sum_{i=1}^n |x_i-\overline{x}|}{n} $$

### Variance
_Variance_ is given by

$$
\sigma^2=\frac{\sum_{i=1}^n (x_i -\overline{x})^2}{n-1}
$$

### Standard Deviation
_Standard deviation_ is given by

$$
\sigma=\sqrt{\frac{\sum_{i=1}^n (x-\overline{x})^2}{n-1}}
$$

### Median Absolute Deviation from the Median
_Median Absolute Deviation from the Median_ is given by

#empty
$$
$$

>💡 The median absolute deviation from the median is prone to outliers (like the [[#Median|Median]] itself), unlike [[#Mean Absolute Deviation|Mean Absolute Deviation]], [[#Variance|Variance]], and [[#Standard Deviation|Standard Deviation]].


###  Interquartile Range
_Interquartile Range_ (_IQR_) is a difference between the 25th and the 75th percentiles.

## Exploring Data Distribution
### Histogram
Histograms and percentiles are used for the same purpose, roughly speaking. The core distinction is that histograms have an uneven number of samples but even bin width and vice versa.

### Density Plot
A density plot can be treated as either a smoothed version of a histogram or an approximation of the distribution function.

#code `pandas.DataFrame.plot.density`.

### Correlation
Most widespread correlation metric is the Pearson’s correlation coefficient:

$$
r=\frac{\sum_{i=1}^n(x-\overline{x})(y-\overline{y})}{(n-1)s_xs_y}
$$

It’s highly prone to outliers and uncovers only linear correlations.

## Examples
### Correlation Matrix #example
![[assets/img/Statistics/Exploratory Data Analysis/01.png|500]]

#sourcecode [[Exploratory Data Analysis Code#Correlation Matrix]].
