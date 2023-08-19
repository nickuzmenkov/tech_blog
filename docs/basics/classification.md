# Classification

### Binary classification
#empty

### Multiclass classification
[[Naive Bayes|Naive Bayes]], [[Linear Models#Gradient Descent|SGD]], and [[Random Forest|Random Forest]] classifiers have native support for multiclass classification tasks.

On the contrary, [[Linear Models#Logistic Regression|Logistic Regression]] and [[Support Vector Machines#Classification|SVC]] do not support it. However, these models can  be scaled to handle multiclass classification tasks as well using either _one-versus-the-rest_ (_OvR_) or _one-versus-one_ (_OvO_) approach.

> 💡This is exactly what sklearn implicitly does for such models.

#### One-Versus-the-Rest (OvR)
Train class detectors for each class (e.g. class A, not class A). For inference choose the highest probability among all models. Requires $N$ models to be fitted with the eintire dataset.

#code  `sklearn.multiclass.OneVsRestClassifier`.

#### One-Versus-One (OvO)
Train models to distinguish between each pair of classes. For inference choose the class winning maximum number of duels. Requires $N(N-1)/2$ models to be fitted with $2/N$ fraction of the dataset.

#code   `sklearn.multiclass.OneVsOneClassifier`.

### Multilabel classification
#empty

### Multioutput classification
_Multioutput classification_ is generalization of multilabel classification to more than one class for each label.

## Strategies for Imbalanced Data
When one of the classes in the dataset is rare (e.g. fraud, churn, cancer, etc.) there’s a decent probability of observing similar records in the majority class just by chance. This is likely to disrupt the whole training process.

In this case one should also concern about the correct choice of metrics, since monitoring wrong metrics (e.g. accuracy) can yield satisfying results even if the model only predicts the majority class all the time.

> 💡 Stratified KFold is handy for multiclass classification problems (especially for those with imbalanced data) hence it makes sure each class is represented like in the original distribution.

### Undersampling
_Undersampling_ the majority class is usually applied when there’s a lot of data available.

### Oversampling
_Oversampling_ the minority class is applied when the opposite is true. Underrepresented class’ records can be bootstrapped forming a larger sample.

### Weighted Samples
_Applying weights_ is an option in both cases. Weights are selected so that the sum of weights of both classes are equal (e.g. if $p$ is the fraction of the minority class, then either the minority records can be weighted as $1/p$ or the majority records can be weighted as $p$).


## Examples
### Multiclass Classification #example 
![[assets/img/Statistics/Classification/01.png|700]]

#sourcecode [[Classification Code#Iris Multiclass Classification]].

### Multiloutput Classification #example
![[assets/img/Statistics/Classification/02.png|700]]

#sourcecode [[Classification Code#MNIST Denoising]]

### Accuracy
Accuracy is simply a fraction of all correct predictions

$$
\text{Accuracy}=\frac{\text{TP}+\text{TN}}{n}
$$

where $\text{TP}$ is the total number of true positives and $\text{TN}$ is the total number of true negatives.

#code `sklearn.metrics.accuracy_score(y_true, y_pred)`.

### Confusion Matrix
Confusion matrix is a way to present results in a form of table with rows of true values and columns of predicted values, giving all possible true/false rates. For #example:

| True\Predicted | A   | B   |
| -------------- | --- | --- |
| A              | 13  | 2   |
| B              | 3   | 12  |

> 💡 #warning Confusion matrix is not necessarily symmetrical and thus cannot be cut along diagonal like the correlation matrix.

#code `sklearn.metrics.confusion_matrix(y_true, y_pred)`.

### Precision
Precision is the accuracy of the positive predictions (i.e. the ratio of correctly predicted positives to all predicted positives):

$$
\text{Precision}=\frac{\text{TP}}{\text{TP}+\text{FP}}
$$

A trivial perfect-precision-model is that one predicting a single positive making sure it’s correct (i.e. the one of the highest confidence score).

Can be drawn from confusion matrix: true positives in the column divided by the sum of the column. For #example from the [[#Confusion Matrix|above Confusion Matrix]] 

$$
\text{Precision}=12/(12+2)=0.86
$$

(i.e. when model predicts class B it is correct 86% of the time).

#code `sklearn.metrics.precision_score(y_true, y_pred)`.

### Recall
Recall (or *true positive rate*) is the ratio of correctly predicted positives to actual positives:

$$
\text{Recall}=\frac{\text{TP}}{\text{TP}+\text{FN}}
$$

A trivial perfect-recall-model is that one always predicting positives.

Can be drawn from confusion matrix: true positives in the row divided by the sum of the row. E.g. from the [[#Confusion Matrix|above Confusion Matrix]]

$$
\text{Recall}=12/(12+3)=0.80
$$

(i.e. model detected 80% of class B instances).

#code `sklearn.metrics.recall_score(y_true, y_pred)`.

### Specificity
Similarly to recall, specificity (or *true negative rate*) is the ratio of correctly predicted negatives to actual negatives:

$$
\text{Specificity}=\frac{\text{TN}}{\text{TN}+\text{FP}}
$$

There’s no direct `sklearn` implementation for specificity. But the required values can be drawn from the [[#Confusion Matrix]]. 

### F1-Score
F1-score is a way to measure precision-recall trade-off as a *harmonic mean* between them:

$$
F_1=\frac{2}{\text{Precision}^{-1} + \text{Recall}^{-1}} = \frac{\text{TP}}{\text{TP} + 0.5(\text{FP} + \text{FN})}
$$

F1-score is computed as the harmonic mean (and not the regular mean) because it gives much more weight to low values. It thus can be high only if both precision and recall are high.

F1-score, however, does not take into account true negative rate.

> 💡 A more generic $F_\beta$ score applies additional weights, valuing one of precision or recall more than the other.

#code `sklearn.metrics.f1_score(y_true, y_pred)`.

## Precision/Recall Trade-Off
Precision and recall are mirroring metrics. Think about this:

![[assets/img/Statistics/Classification Metrics/01.png|600]]

- As we increase the threshold there are fewer positives of higher confidence (those are less likely to be false positives), hence higher precision.
- As we decrease the threshold, more positives are predicted, raising the chance to predict all actual positives (and false positives), hence higher recall.

In most cases, either high precision or high recall is required most:

- Adult content filter (positive is *safe*): high precision is required, low recall is acceptable.
- Shoplifter identifier (positive is *suspect*): high recall is required, low precision is acceptable.

There are 2 visual representations for precision/recall trade-off: precision-recall (PR) curve and receiver operating curve (ROC). Both can give insights and help to choose a proper threshold based on precision and recall relative importance.

>💡 It's generally advisable to choose PR curve when the positive class is rare or when you care more about false positives. Otherwise it is suggested to use ROC.


### PR (Precision-Recall) Curve
There are two ways to plot the PR curve. One is precision and recall versus the probability (`predict_proba`) or decision function (`decision_function`) threshold:

![[assets/img/Statistics/Classification Metrics/03.png|500]]
The other way is to plot precision versus recall directly:
![[assets/img/Statistics/Classification Metrics/02.png|700]]

#code `sklearn.metrics.precision_recall_curve`.
#sourcecode [[Classification Metrics Code#PR Curve]].

### ROC (Receiver Operating Characteristics) Curve
ROC curve shows $1-\text{Specificity}$ (also *called false positive rate*) versus recall (t*rue positive rate*):

![[04.png|500]]

The analytic way to compute ROC is like this:

1. Sort records descending by predicted probability.
2. Compute cumulative specificity and recall on sorted records.

#code `sklearn.metrics.roc_curve`.
#sourcecode [[Classification Metrics Code#ROC Curve]].

### ROC AUC (Area Underneath the Curve)
A quantitative metric associated with the ROC curve is the *area underneath the curve* (AUC or ROC AUC). AUC is equal to 1.0 for a perfect classifier and to 0.5 for a random classifier.

>💡 Thus ROC is drawn for all possible thresholds, ROC AUC is threshold independent.

#code `sklearn.metrics.roc_auc_score`.
