# preprocessing.MinMaxScaler用法

## `preprocessing.MinMaxScaler  `

当数据(x)按照最小值中心化后，再按极差（最大值 - 最小值）缩放，数据移动了最小值个单位，并且会被收敛到 [0,1] 之间，而这个过程，就叫做数据归一化(Normalization，又称Min-Max Scaling)。注意，Normalization是归一化，不是正则化，真正的正则化是regularization，不是数据预处理的一种手段。归一化之后的数据服从正态分布，公式如下： 
$$
x^{*}=\frac{x-\min (x)}{\max (x)-\min (x)}
$$


在sklearn当中，我们使用preprocessing.MinMaxScaler来实现这个功能。MinMaxScaler有一个重要参数，feature_range，控制我们希望把数据压缩到的范围，默认是[0,1]  .



函数：

```python
class sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), 
                                         copy=True
                                        )
```



属性：

**`min_`** :   ， 计算方式 min - X.min(axis=0) * self.scale_

**`scale_`** : 每个特征的相对缩放比例，计算方式 (max - min) / (X.max(axis=0) - X.min(axis=0))

**`data_min_`** ：最小值

**`data_max_`** : 最大值

**`ata_range_`** : 数据的范围 ，计算方式 data_max_ - data_min_



源码：<a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html?highlight=minmaxscaler#sklearn.preprocessing.MinMaxScaler" target="_blank">sklearn.preprocessing.MinMaxScaler</a>

```python
from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

import pandas as pd
pd.DataFrame(data)

#实现归一化
scaler = MinMaxScaler() #实例化
scaler = scaler.fit(data) #fit，在这里本质是生成min(x)和max(x)
result = scaler.transform(data) #通过接口导出结果
result

result_ = scaler.fit_transform(data) #训练和导出结果一步达成
scaler.inverse_transform(result) #将归一化后的结果逆转

#使用MinMaxScaler的参数feature_range实现将数据归一化到[0,1]以外的范围中
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

scaler = MinMaxScaler(feature_range=[5,10]) #依然实例化
result = scaler.fit_transform(data) #fit_transform一步导出结果
result

#当X中的特征数量非常多的时候，fit会报错并表示，数据量太大了我计算不了
#此时使用partial_fit作为训练接口
#scaler = scaler.partial_fit(data)
```

