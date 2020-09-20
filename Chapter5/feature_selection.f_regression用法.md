# feature_selection.f_regression用法



```python
sklearn.feature_selection.f_classif(X, y)
```

**Parameters**

**X**

{array-like, sparse matrix} shape = [n_samples, n_features]

The set of regressors that will be tested sequentially.

**y**

array of shape(n_samples)

The data matrix.

**center**

*rue, bool,*

If true, X and y will be centered.

<br />



**Returns**

**F**

array, shape = [n_features,]

F values of features.

**pval**

array, shape = [n_features,]

p-values of F-scores.