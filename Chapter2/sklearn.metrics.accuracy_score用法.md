# sklearn.metrics.accuracy_score用法

**准确率accuracy** :所有的测量点到真实值非常接近。与测量点的偏差有关。

```python
sklearn.metrics.accuracy_score(y_true, 
                               y_pred, 
                               normalize=True, 
                               sample_weight=None
                              )
```

**Parameters：**

- **y_true** ：**1d array-like, or label indicator array / sparse matrix**
- **y_pred** ：**1d array-like, or label indicator array / sparse matrix**
- **normalize**：**bool, optional (default=True)**
  如果为False，则返回正确分类的样本数。 否则，返回正确分类的样本的分数。
- **sample_weight**： **array-like of shape (n_samples,), default=None**



**Returns**：

- **score**：**float**

<br>

**coding**

```python
>>> from sklearn.metrics import accuracy_score
>>> y_pred = [0, 2, 1, 3]
>>> y_true = [0, 1, 2, 3]
>>> accuracy_score(y_true, y_pred)
0.5
>>> accuracy_score(y_true, y_pred, normalize=False)
2
```



