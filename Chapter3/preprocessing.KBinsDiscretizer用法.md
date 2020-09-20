# preprocessing.KBinsDiscretizer用法

将连续型变量划分为分类变量的类，能够将连续型变量排序后按顺序分箱后编码  .

```python
class sklearn.preprocessing.KBinsDiscretizer(n_bins=5, 
                                             encode='onehot', 
                                             strategy='quantile'
                                            )
```

**Parameters**

| 参数     | 含义&输入                                                    |
| -------- | ------------------------------------------------------------ |
| n_bins   | 每个特征中分箱的个数，默认5，一次会被运用到所有导入的特征    |
| encode   | 编码的方式，默认“onehot”<br /> "onehot"：做哑变量，之后返回一个稀疏矩阵，每一列是一个特征中的一个类别，含有该 类别的样本表示为1，不含的表示为0 <br />“ordinal”：每个特征的每个箱都被编码为一个整数，返回每一列是一个特征，每个特征下含 有不同整数编码的箱的矩阵 <br />"onehot-dense"：做哑变量，之后返回一个密集数组。 |
| strategy | 用来定义箱宽的方式，默认"quantile" <br />"uniform"：表示等宽分箱，即每个特征中的每个箱的最大值之间的差为<br /> (特征.max() - 特征.min())/(n_bins)<br /> "quantile"：表示等位分箱，即每个特征中的每个箱内的样本数量都相同 <br />"kmeans"：表示按聚类分箱，每个箱中的值到最近的一维k均值聚类的簇心得距离都相同 |

**Attributes**

**`n_bins_`**

每个特征的数量



```python
from sklearn.preprocessing import KBinsDiscretizer
X = data.iloc[:,0].values.reshape(-1,1)

# array([[1],
#        [0],
#        [1],
#        ...,
#        ...,
#        [7],
#        [6],
#        [9]], dtype=int64)

est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
est.fit_transform(X)
#查看转换后分的箱：变成了一列中的三箱
set(est.fit_transform(X).ravel())
est = KBinsDiscretizer(n_bins=3, encode='onehot', strategy='uniform')
#查看转换后分的箱：变成了哑变量
est.fit_transform(X).toarray()
# array([[1., 0., 0.],
#        [0., 1., 0.],
#        [1., 0., 0.],
#        ...,
#        [0., 1., 0.],
#        [1., 0., 0.],
#        [0., 1., 0.]])
```

