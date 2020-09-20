# feature_selection.chi2用法

方差挑选完毕之后，我们就要考虑下一个问题：相关性了。我们希望选出与标签相关且有意义的特征，因为这样的特征能够为我们提供大量信息。如果特征与标签无关，那只会白白浪费我们的计算内存，可能还会给模型带来噪音。

在sklearn当中，我们有三种常用的方法来评判特征与标签之间的相关性：卡方，F检验，互信息  



**卡方过滤是专门针对离散型标签（即分类问题）的相关性过滤**。卡方检验类`feature_selection.chi2`计算每个非负特征和标签之间的卡方统计量，并依照卡方统计量由高到低为特征排名。再结合`feature_selection.SelectKBest`这个可以输入”评分标准“来选出前K个分数最高的特征的类，我们可以借此除去最可能独立于标签，与我们分类目的无关的特征  

另外，如果卡方检验检测到某个特征中所有的值都相同，会提示我们使用方差先进行方差过滤。并且，刚才我们已经验证过，当我们使用方差过滤筛选掉一半的特征后，模型的表现时提升的。因此在这里，我们使用threshold=中位数时完成的方差过滤的数据来做卡方检验（如果方差过滤后模型的表现反而降低了，那我们就不会使用方差过滤后的数据，而是使用原数据）  

```python
sklearn.feature_selection.chi2(X, y)
```

**Parameters**

**`X`**

{array-like, sparse matrix} of shape (n_samples, n_features)*

Sample vectors.

**`y`**

array-like of shape (n_samples,)*

Target vector (class labels).

<br>

**Returns**

**`chi2`**

array, shape = (n_features,)*

chi2 statistics of each feature.

**`pval`**

array, shape = (n_features,)*

p-values of each feature.



卡方检验的本质是推测两组数据之间的差异，其检验的原假设是”两组数据是相互独立的”。卡方检验返回卡方值和P值两个统计量，其中卡方值很难界定有效的范围，而p值，我们一般使用0.01或0.05作为显著性水平，即p值判断的边界，具体我们可以这样来看  

| P值      | <=0.05或0.01             | >0.05或0.01                |
| -------- | ------------------------ | -------------------------- |
| 数据差异 | 差异不是自然形成的       | 这些差异是很自然的样本误差 |
| 相关性   | 两组数据是相关的         | 两组数据是相互独立的       |
| 原假设   | 拒绝原假设，接受备择假设 | 接受原假设                 |

<br>

**Examples**

**首先import包和实验数据：**

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
 
#导入IRIS数据集
iris = load_iris()
iris.data#查看数据
```

```
array([[ 5.1,  3.5,  1.4,  0.2],
       [ 4.9,  3. ,  1.4,  0.2],
       [ 4.7,  3.2,  1.3,  0.2],
       [ 4.6,  3.1,  1.5,  0.2],
       [ 5. ,  3.6,  1.4,  0.2],
       [ 5.4,  3.9,  1.7,  0.4],
       [ 4.6,  3.4,  1.4,  0.3],
```

**使用卡方检验来选择特征**

```python
model1 = SelectKBest(chi2, k=2)
# 选择k个最佳特征
# iris.data是特征数据，iris.target是标签数据，该函数可以选择出k个特征 
model1.fit_transform(iris.data, iris.target)

```

结果输出为：

```
array([[ 1.4,  0.2],
       [ 1.4,  0.2],
       [ 1.3,  0.2],
       [ 1.5,  0.2],
       [ 1.4,  0.2],
       [ 1.7,  0.4],
       [ 1.4,  0.3],

```



可以看出后使用卡方检验，选择出了后两个特征。如果我们还想查看卡方检验的p值和得分

```python
model1.scores_  #得分
```

得分输出为：

```
array([ 10.81782088, 3.59449902, 116.16984746, 67.24482759])
```

可以看出后两个特征得分最高，与我们第二步的结果一致；

```python
model1.pvalues_  #p-values
```

p值输出为：

```
array([ 4.47651499e-03, 1.65754167e-01, 5.94344354e-26, 2.50017968e-15])
```

可以看出后两个特征的p值最小，置信度也最高.