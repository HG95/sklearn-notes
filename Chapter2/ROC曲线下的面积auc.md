# ROC曲线下的面积auc

```python
sklearn.metrics.auc(x, y)
```

Compute Area Under the Curve (AUC) using the trapezoidal rule

**Parameters**:

- `x` array, shape = [n]
  x coordinates. These must be either monotonic increasing or monotonic decreasing.

- `y` array, shape = [n]
  y coordinates.

**Returns**:

- auc

## Examples

```python
>>> import numpy as np
>>> from sklearn import metrics
>>> y = np.array([1, 1, 2, 2])
>>> pred = np.array([0.1, 0.4, 0.35, 0.8])
>>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
>>> metrics.auc(fpr, tpr)
0.75
```



```python
from sklearn.metrics import roc_curve,auc

y_true = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
y_score = [0.31689620142873609, 0.32367439192936548, 0.42600526758001989, 0.38769987193780364, 0.3667541015524296, 0.39760831479768338, 0.42017521636505745, 0.41936155918127238, 0.33803961944475219, 0.33998332945141224]

fpr, tpr, thresholds = roc_curve(y_true, 
                                 y_score, 
                                 drop_intermediate=False)

auc(fpr,tpr)

# 0.9047619047619048
```

