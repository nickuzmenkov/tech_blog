# Linear Discriminant Analysis

>  💡 #story Discriminant analysis is the earliest statistical classifier; it was introduced by R. A. Fisher in 1936 in an article published in the _Annals of Eugenics_ journal.

While discriminant analysis encompasses several techniques, the most commonly used is _linear discriminant analysis_ (LDA).

LDA searches for optimal linear decision boundaries between the classes so that sum of squares _between_ the groups is as high as possible while the sum of squares _within_ the groups is as low as possible.

Because of LDA decision boundaries can be used as new axes, LDA is also widely used for dimensionality reduction.

>  💡 Despite LDA is designed for normally distributed numerical predictors, it can be used with slightly non-normal distributed and binary variables as well.

### Synthetic Data #example 
Classify random samples of two normal distributions with mean 3 (y=0) and 5 (y=1):

![[assets/img/Machine Learning/Linear Discriminant Analysis/01.png|600]]

#### Solution
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plotly.graph_objects as go
import numpy as np

SIZE = 50

X1 = np.random.normal(loc=2, size=(SIZE, 2))
X2 = np.random.normal(loc=5, size=(SIZE, 2))

y1 = np.zeros(SIZE)
y2 = np.ones(SIZE)

X = np.r_[X1, X2]
y = np.r_[y1, y2]

model = LinearDiscriminantAnalysis()
model.fit(X, y)
predict = model.predict_proba(X)

center = np.mean(model.means_, axis=0)
coef = -float(model.scalings_[0] / model.scalings_[1])
intercept = float(center[1] - center[0] * coef)

go.Figure(
    data=(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(
                size=10,
                color=predict[:, 0],
                colorscale="Tropic",
                colorbar=dict(title="class probability"),
                showscale=True,
            ),
        ),
        go.Scatter(
            x=[np.min(X[:, 0]), np.max(X[:, 0])],
            y=[
                (np.min(X[:, 0]) - intercept) / coef,
                (np.max(X[:, 0]) - intercept) / coef,
            ],
            line=dict(width=3, color="black"),
            mode="lines",
        ),
    ),
    layout=dict(
        height=500,
        width=700,
        title_text="LDA Example",
        xaxis=dict(title_text="x1"),
        yaxis=dict(title_text="x2"),
        font=dict(size=16),
        showlegend=False,
    ),
).write_html("1.html")
```
