# feature_selection.f_classif 用法

F 检验，又称 ANOVA，方差齐性检验，是用来捕捉每个特征与标签之间的线性关系的过滤方法。它即可以做回归也可以做分类，因此包含`feature_selection.f_classif`（F检验分类）和`feature_selection.f_regression`（F检验回归）两个类。其中F检验分类用于标签是离散型变量的数据，而 F检验回归用于标签是连续型变量的数据  。

和卡方检验一样，这两个类需要和类`SelectKBest`连用，并且我们也可以直接通过输出的统计量来判断我们到底要设置一个什么样的 K 。需要注意的是，F检验在数据服从正态分布时效果会非常稳定，因此如果使用F检验过滤，我们**会先将数据转换成服从正态分布的方式**  。

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

<br />

**Returns**

**F**

array, shape = [n_features,]

The set of F values.

**pval**

array, shape = [n_features,]

The set of p-values.



