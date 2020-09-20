# sklearn.metrics.roc_auc_score用法

计算AUC (Area Under Curve) 面积的类 `sklearn.metrics.roc_auc_score  `

直接根据真实值（必须是二值）、预测值（可以是**0/1,也可以是proba值）计算出auc值，中间过程的roc计算省略**。

```python
sklearn.metrics.roc_auc_score ( y_true, 
                                y_score, 
                                average=’macro’, 
                                sample_weight=None, 
                                max_fpr=None
                            )
```

- `y_true` :array, shape = [n_samples] or [n_samples, n_classes]
  真实的标签
- `y_score` :array, shape = [n_samples] or [n_samples, n_classes]
   预测得分，可以是正类的估计概率、置信值或者分类器方法 “decision_function” 的返回值；
- `average` :string, [None, ‘micro’, ‘macro’ (default), ‘samples’, ‘weighted’]
- `sample_weight`  : array-like of shape = [n_samples], optional



```python
from sklearn.metrics import roc_auc_score as AUC
area = AUC(y,clf_proba.decision_function(X))
```

