# K-Nearest Neighbors

K-nearest neighbors (KNN) is a data-based approach for both classification and regression.

The KNN algorithm is pretty easy:

-   Find $K$ records that have all similar features as the new record.
-   For classification, return the mode class of those records.
-   For regression, return the mean target value of those records.

The hyperparameter K is usually selected between 1 and 20.

Similarity is measured using one of possible distance metrics, e.g. [[base/Statistics/Notation#Manhattan Distance|Manhattan]] or [[base/Statistics/Notation#Euclidean Distance|Euclidean]].

#code 
- `sklearn.neighbors.KNeighborsRegressor`
- `sklearn.neighbors.KNeighborsClassifier`

Inputs must be normalized with one of possible ways:
-   subtract mean and divide by standard deviation (standard scaler)
-   subtract minimum value and divide by range (`MinMaxScaler`)
-   subtract median and divide by interquartile range (robust to outliers)

The idea is to put all predictors on the same scale

> 💡 #warning Only Linear and Logistic Regression requires to drop one of the categories of one-hot encoded factor variables. Other techniques do not.
