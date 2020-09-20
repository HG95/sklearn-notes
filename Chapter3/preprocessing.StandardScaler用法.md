# preprocessing.StandardScaler用法

## `preprocessing.StandardScaler`  

当数据(x)按均值(μ)中心化后，再按标准差(σ)缩放，数据就会服从为均值为0，方差为1的正态分布（即标准正态分布），而这个过程，就叫做数据标准化(Standardization，又称Z-score normalization)，公式如下：  
$$
x^{*}=\frac{x-\mu}{\sigma}
$$


函数：

```python
class sklearn.preprocessing.StandardScaler(copy=True, 
                                           with_mean=True, 
                                           with_std=True
                                          )
```



属性：

**`scale_`**：标准差

**`mean_`**：均值

**`var_`**： 方差



## StandardScaler和MinMaxScaler选哪个？  

大多数机器学习算法中，会选择StandardScaler来进行特征缩放，因为MinMaxScaler对异常值非常敏。在PCA，聚类，逻辑回归，支持向量机，神经网络这些算法中，StandardScaler往往是最好的选择。

MinMaxScaler在不涉及距离度量、梯度、协方差计算以及数据需要被压缩到特定区间时使用广泛，比如数字图像处理中量化像素强度时，都会使用MinMaxScaler将数据压缩于[0,1]区间之中。

建议先试试看StandardScaler，效果不好换MinMaxScaler  